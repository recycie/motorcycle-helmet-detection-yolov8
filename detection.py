import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
import time
import imutils
import numpy as np
import streamlink
import hashlib
import threading
import json
import jwt
import configparser
from datetime import datetime, timedelta, timezone
from flask import Flask, jsonify, request, Response, g
from flask_cors import CORS
from utils.mysql import *
from utils.model import Model, letterbox, non_max_suppression, calculate_iou
from utils.sort import *
from utils.predict_seg import YOLO_SEGMENT

tracker = Sort()
app = Flask(__name__, static_folder="runs")
CORS(app)

def create_default_config(config_file):
    config = configparser.ConfigParser()

    config['app'] = {
        'secret_key': 'catxcatxcat'
    }

    config['motorcycle'] = {
        'file': 'models/motorcycle.41.best14kv8n.onnx',
        'confidence': 0.4,
        'iou': 0.5
    }

    config['helmet'] = {
        'file': 'models/best-helmet8l.onnx',
        'confidence': 0.4,
        'iou': 0.45
    }

    config['person'] = {
        'file': 'models/segment_driver-v8n.onnx',
        'confidence': 0.4,
        'iou': 0.45
    }

    config['database'] = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': '',
        'dbname': 'helmet'
    }

    config['server'] = {
        'host': '0.0.0.0',
        'port': 8001,
        'debug': True,
        'threaded': True,
        'use_reloader': False
    }

    # Write the default configuration to the file
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    print(f"Created default config file: {config_file}")

def load_config(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    return config

def check_model_file(file_path, model_name):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file for {model_name} not found: {file_path}")

COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'gray': (128, 128, 128),
    'purple': (128, 0, 128),
    'orange': (0, 165, 255),
    'pink': (203, 192, 255),
    'brown': (42, 42, 165),
    'teal': (128, 128, 0),
    'lime': (0, 255, 0),
    'olive': (0, 128, 128),
    'navy': (128, 0, 0),
    'maroon': (0, 0, 128),
    'gold': (0, 215, 255),
    'silver': (192, 192, 192),
    'indigo': (130, 0, 75)
}

def generate(resolution):
    while True:
        try:
            if not status:
                break

            if steamFrame is None:
                time.sleep(0.03)
                continue

            resizeFrame = steamFrame
            if resolution:
                resizeFrame = imutils.resize(resizeFrame, resolution)

            flag, encodedImage = cv2.imencode(".jpg", resizeFrame)
            
            if not flag:
                time.sleep(0.03)
                continue
            
            # Use bytes() for more explicit byte conversion
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')
        
        except Exception as e:
            print("Image reading failed:", e)
            time.sleep(0.1)
            continue

@app.route("/api/v1/steam")
def videosteam():
    try:
        res = request.args.get("res", type=int)
        if res in resolution:
            res = resolution[resolution.index(res)]
        else:
            res = resolution[0]  # Default to the smallest resolution if not valid

        return Response(generate(res), mimetype="multipart/x-mixed-replace; boundary=frame")
    
    except Exception as e:
        print("Error in videosteam endpoint:", e)
        # Provide a JSON error response with a 500 status code
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Not Found", "message": str(error)}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error", "message": str(error)}), 500

@app.route("/api/v1/monitor", methods=['GET', 'POST'])
def getMonitor():
    try:
        
        if request.method == 'GET':
            cmd = "SELECT * FROM monitor"

            data = DATABASE.select(cmd)
            if not data:
                return jsonify({"status": "error", "msg": "Monitor is Empty."})
            
            return jsonify({"status": "success", "msg": "ok", "data": data})
        elif request.method == 'POST':
            data = request.get_json()

            name = data.get('name')
            description = data.get('description')
            source_type = data.get('url')

            if not os.path.isfile(source_type):
                return jsonify({"status": "error", "msg": "File not found."})

            cmd = "INSERT INTO `monitor`(`name`, `description`, `url`, `status`) VALUES (%s, %s, %s, 1)"
            data = DATABASE.insert(cmd, (name, description, source_type))
            if not data:
                return jsonify({"status": "error", "msg": f""})
            
            return jsonify({"status": "success", "msg": "ok"})
        return jsonify({"status": "error", "msg": "Method Not Allow"}), 401
    except Exception as e:
        return jsonify({"status": "error", "msg": "Internal Server Error", "error": str(e)}), 500

@app.route("/api/v1/detection/<string:method>", methods=['GET'])
def toggle(method: str):
    global status, monitor_id

    method = method.lower()

    if request.method != "GET":
        return jsonify({"status": "error", "msg": "Method Not Allowed"})

    if method == "start":
        
        if not request.args.get("monitor_id"):
            return jsonify({"status": "error", "msg": "Parameter 'monitor_id' is missing."})

        if status:
            return jsonify({"status": "error", "msg": "Detection is already running."})

        monitor_id = int(request.args.get("monitor_id"))
        cmd = "SELECT url FROM monitor WHERE id = %s"
        data = DATABASE.select(cmd, (monitor_id,))

        if not data:
            return jsonify({"status": "error", "msg": f"Monitor ID {monitor_id} not found."})

        source = data[0][0]
        if not os.path.isfile(source):
            return jsonify({"status": "error", "msg": f"File {source} not found."})

        print("Initializing...")
        start_time = time.time()
        modelInitialize()
        print(f"{time.time() - start_time:.2f} sec Done.")

        threading.Thread(target=main, args=(source,), daemon=True).start()

        status = None

        # Wait for the status to be updated
        while status is None:
            time.sleep(0.3)
        
        if status:
            return jsonify({"status": "success", "msg": "Detection has started."})
        else:
            return jsonify({"status": "error", "msg": "Failed to start detection."})

    elif method == "stop":
        if status is None:
            return jsonify({"status": "success", "msg": "Detection is already stopped."})
        
        status = False
        return jsonify({"status": "success", "msg": "Detection has been stopped."})

    return jsonify({"status": "error", "msg": "Invalid method specified."})

@app.route("/api/v1/confidences", methods=['GET'])
def set_confidence():
    global status, helmet_conf, helmet_iou, motorcycle_conf, motorcycle_iou, person_conf, person_iou

    if request.method == "GET":
        helmet_conf = request.args.get("helmet_conf", type=float) if request.args.get("helmet_conf", type=float) else helmet_conf
        helmet_iou = request.args.get("helmet_iou", type=float) if request.args.get("helmet_iou", type=float) else helmet_iou
        motorcycle_conf = request.args.get("motorcycle_conf", type=float) if request.args.get("motorcycle_conf", type=float) else motorcycle_conf
        motorcycle_iou = request.args.get("motorcycle_iou", type=float) if request.args.get("motorcycle_iou", type=float) else motorcycle_iou
        person_conf = request.args.get("person_conf", type=float) if request.args.get("person_conf", type=float) else person_conf
        person_iou = request.args.get("person_iou", type=float) if request.args.get("person_iou", type=float) else person_iou

    return jsonify({
            "helmet_conf": helmet_conf, 
            "helmet_iou": helmet_iou, 
            "motorcycle_conf": motorcycle_conf, 
            "motorcycle_iou": motorcycle_iou, 
            "person_conf": person_conf, 
            "person_iou": person_iou
        })

@app.route('/api/v1/status')
def get_status():
    def status_data():
        while True:
            data = {"status": status}
            yield f"data: {json.dumps(data)}\n\n"
            time.sleep(0.5)
    return Response(status_data(), content_type='text/event-stream')

@app.route('/api/v1/result')
def get_result():
    def result_data():
        while True:
            yield f"data: {json.dumps(detected_data)}\n\n"
            time.sleep(0.1)
    return Response(result_data(), content_type='text/event-stream')

@app.route("/api/v1/data/detection/<string:method>", methods=['GET'])
def getDetectionData(method):
    if request.method != "GET":
        return jsonify({"data": None})

    try:
        start_time = request.args.get("start")
        end_time = request.args.get("end")
        monitor_id = request.args.get("monitor_id")
        limit_value = request.args.get("limit")
        sort = request.args.get("sort")
        did = request.args.get("did")
        hour = request.args.get("h")
        day = request.args.get("d")
        month = request.args.get("m")

        # Add conditions to the query
        conditions = []
        params = []

        # Base query based on method
        if method == "all":
            cmd = "SELECT motorcycle_id, helmet_score, person_score, helmet, driver, DATE_FORMAT(created_at, '%Y/%m/%d %H:%i:%S') created_at, id FROM detection"
        elif method == "monitor":
            cmd = "SELECT * FROM monitor"
        elif method == "bbox" and did:
            cmd = "SELECT * FROM bbox WHERE detection_id = %s"
            params.append(did)
        elif method == "summary":
            cmd = """
                SELECT COUNT(DISTINCT motorcycle_id) AS motorcycle,
                       COUNT(id) AS person,
                       SUM(CASE WHEN helmet = 1 THEN 1 ELSE 0 END) AS helmet,
                       SUM(CASE WHEN helmet = 0 THEN 1 ELSE 0 END) AS no_helmet,
                       ROUND(AVG(CASE WHEN helmet_score != 0 THEN helmet_score ELSE NULL END), 3) AS helmet_accuracy,
                       ROUND(AVG(CASE WHEN person_score != 0 THEN person_score ELSE NULL END), 3) AS person_score_accuracy,
                       SUM(CASE WHEN driver = 1 AND helmet = 1 THEN 1 ELSE 0 END) AS driver_helmet,
                       SUM(CASE WHEN driver = 1 AND helmet = 0 THEN 1 ELSE 0 END) AS driver_no_helmet,
                       SUM(CASE WHEN driver = 0 AND helmet = 1 THEN 1 ELSE 0 END) AS non_driver_helmet,
                       SUM(CASE WHEN driver = 0 AND helmet = 0 THEN 1 ELSE 0 END) AS non_driver_no_helmet
                FROM detection
            """
        elif method == "chart":
            if day == "_":
                startFormat = '%Y-%m-%d 00:00:00'
            elif hour == "_":
                startFormat = '%Y-%m-%d %H:00:00'
            elif month == "_":
                startFormat = '%Y-%m'
            else:
                return jsonify({"data": None})

            cmd = f"""
                SELECT 
                    SUM(CASE WHEN helmet = 1 THEN 1 ELSE 0 END) AS helmet,
                    SUM(CASE WHEN helmet = 0 THEN 1 ELSE 0 END) AS no_helmet,
                    DATE_FORMAT(created_at, '{startFormat}') AS d
                FROM detection
            """
        else:
            return jsonify({"data": None})

        if (start_time and end_time):
            conditions.append("created_at BETWEEN %s AND %s")
            params.extend([f"{start_time} 00:00:00", f"{end_time} 23:59:59"])

        if monitor_id and monitor_id.isnumeric():
            conditions.append("monitor_id = %s")
            params.append(monitor_id)

        if conditions:
            cmd += " WHERE " + " AND ".join(conditions)

        if method == "chart":
            cmd += f" GROUP BY DATE_FORMAT(created_at, '{startFormat}') ORDER BY DATE_FORMAT(created_at, '{startFormat}')"
        elif sort and sort.lower() in ['asc', 'desc']:
            cmd += f" ORDER BY created_at {sort.upper()}"

        if limit_value and limit_value.isnumeric():
            cmd += f" LIMIT %s"
            params.append(limit_value)

        
        data = DATABASE.select(cmd, params)

        return jsonify({"data": data})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"data": None})

def generate_token():
    expiry = datetime.now(timezone.utc) + timedelta(days=3650)

    # Create JWT token (permanent token)
    token = jwt.encode({'user': 'APIPERMANENTTOKEN', 'exp': expiry}, app.config['SECRET_KEY'], algorithm='HS256')

    # # Save token to a file
    # with open('token.json', 'w') as f:
    #     f.write(token)

    # return token.decode('UTF-8')

def generate_id(base_int, length):
    # Get the current timestamp
    current_time = time.time()
    combined = str(base_int) + str(current_time)
    hashed_id = hashlib.sha256(combined.encode()).hexdigest()[:length]
    return hashed_id

def modelInitialize():
    global motorcycle_model, helmet_model, rider_model
    motorcycle_model = Model(motorcycle_file)
    helmet_model = Model(helmet_file)
    rider_model = YOLO_SEGMENT(person_file)

    # Init
    img_test = cv2.imread("data/test2.jpg")
    rider_model.predict(img_test)
    img = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
    img, ratio, dwdh = letterbox(img, auto=False)

    motorcycle_model.predict(img)
    helmet_model.predict(img)

def fill_background(img, bbox):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    x, y, w, h = bbox
    mask[y:h, x:w] = 255

    # สร้าง inverted mask เพื่อเลือกส่วนของรูปที่ไม่ได้อยู่ใน bounding box
    inverted_mask = cv2.bitwise_not(mask)

    image = np.copy(img)

    # กำหนดสีพื้นหลังในส่วนที่ไม่ได้อยู่ใน bounding box
    image[inverted_mask > 0] = (255, 255, 255)  # เปลี่ยนสีพื้นหลังที่คุณต้องการ
    return image

def fill_bbox(img, bbox):
    image = img.copy()
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLORS['black'], -1)
    return image

def draw_box(image, box, color=COLORS['red'], thickness=1):
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness)

def person_segment(img, id):
    global detected_data
    image = img.copy()
    boxs, scores, classids, masks = rider_model.predict(img, conf=0.5, iou=0.45)

    motorcycle_id = generate_id(id, 10)
    
    # motorcycle_id = str(time.time())

    for box, score, cls, mask in zip(boxs, scores, classids, masks):
        box = np.array(box)
        person_box = box.astype(np.int32).tolist()
        img_fill = img.copy()

        # mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
        # img_fill[mask_resized == 0] = 0
        img_fill[mask == 0] = 0

        person_score = round(float(score), 3)
        # FNF = f"runs/images/{score}_{IMAGE_NAME}"

        helmet_result = helmet_detection(img_fill)
        if cls == 0:
            driver = 1
            draw_box(image, person_box, COLORS['green'], 2)
        else:
            driver = 0
            draw_box(image, person_box, COLORS['red'], 2)

        cv2.putText(image, f"{person_score}", (person_box[2], person_box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['black'], thickness=1)


        ddt = []
        # Helmet check
        if len(helmet_result) > 0:
            helmet_score = helmet_result[1]
            helmet_bbox = helmet_result[0]
            hashed = generate_id(id, 24)

            for i in tracking_box:
                if i["id"] == id:
                    i["helmet"] = True
                    ddt.append([motorcycle_id, helmet_score, person_score, driver, 1])

                    # DIR_IMAGE = f"runs/images/{temp_folder}/helmet"
                    # DIR_IMAGE = f"runs/images/helmet"

                    draw_box(image, helmet_bbox, COLORS['yellow'], 3)
                    cv2.putText(image, f"{person_score}", (helmet_bbox[2], helmet_bbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['black'], thickness=1)
                    DATABASE.insert_detection(hashed, motorcycle_id, monitor_id, driver, 1, helmet_score, person_score)
                    DATABASE.insert_bbox(hashed, helmet_bbox[0], helmet_bbox[1], helmet_bbox[2], helmet_bbox[3], person_box[0], person_box[1], person_box[2], person_box[3])
        else:
            hashed = generate_id(id, 24)
            for i in tracking_box:
                if i["id"] == id and i["helmet"] != True:
                    i["helmet"] = False
                    ddt.append([motorcycle_id, 0, person_score, driver, 0])

                    # DIR_IMAGE = f"runs/images/{temp_folder}/nohelmet"
                    # DIR_IMAGE = f"runs/images/nohelmet"

                    DATABASE.insert_detection(hashed, motorcycle_id, monitor_id, driver, 0, 0, person_score)
                    DATABASE.insert_bbox(hashed, 0, 0, 0, 0, person_box[0], person_box[1], person_box[2], person_box[3])

    DIR_IMAGE = f"runs/images"
    if not os.path.exists(DIR_IMAGE):
        os.makedirs(DIR_IMAGE)

    IMAGE_NAME = motorcycle_id + ".jpg"
    FN = f"{DIR_IMAGE}/{IMAGE_NAME}"
    cv2.imwrite(FN, image)
    FN = f"{DIR_IMAGE}/raw-{IMAGE_NAME}"
    cv2.imwrite(FN, img)

    for dd in ddt:
        detected_data = dd
        time.sleep(0.5)

def helmet_detection(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, ratio, dwdh = letterbox(image, auto=False)
    output = helmet_model.predict(image)
    helmet_pred = non_max_suppression(output, helmet_conf, helmet_iou)
    helmet_bbox = []
    for i, DET in enumerate(helmet_pred): 
        if len(DET):
            for *xyxy, score, cls_id in reversed(DET):
                cls_id = int(cls_id)
                score = round(float(score), 3)
                box = np.array(xyxy)
                box -= np.array(dwdh*2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
                data = [box, score]
                helmet_bbox.append(data)

    if len(helmet_bbox) > 0:
        helmet_bbox = helmet_bbox[max(range(len(helmet_bbox)), key=lambda i: helmet_bbox[i][1])]

    return helmet_bbox

def main(url):
    global steamFrame, motorcycle_count, helmet_count, tracking_box, status

    cap = None
    try:
        # Set up video capture
        stream_url = streamlink.streams(url)["best"].url if "youtube" in url else url
        cap = cv2.VideoCapture(stream_url)
        
        if not cap.isOpened():
            raise RuntimeError("Failed to open video capture.")

        # Get frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        count_area = [frame_width // 3, 0, frame_width // 3 + 1, frame_height]

        while True:
            try:
                ref, frame = cap.read()
                if not ref or status == False:
                    print("Reading steam failed.")
                    steamFrame = None
                    status = False
                    break

                status = True
                frame = imutils.resize(frame)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image, ratio, dwdh = letterbox(image, auto=False)
                
                # Perform object detection
                start_time = time.time()
                output = motorcycle_model.predict(image)
                pred = non_max_suppression(output, motorcycle_conf, motorcycle_iou)
                processing_time = (time.time() - start_time) * 1000

                for DET in pred:
                    if len(DET) == 0:
                        continue

                    track_ids = tracker.update(DET)
                    for coords in track_ids.tolist():
                        tbox = (np.array(coords[:4]) - np.array(dwdh * 2)) / ratio
                        tbox = tbox.round().astype(int).tolist()
                        tbox[1] = max(0, tbox[1] - int(tbox[1] * 0.05))
                        id = int(coords[4])

                        for *xyxy, score, cls_id in reversed(DET):
                            cls_id = int(cls_id)
                            score = f"{score:.3f}"
                            box = (np.array(xyxy) - np.array(dwdh * 2)) / ratio
                            box = box.round().astype(int).tolist()
                            box[1] = max(0, box[1] - int(box[1] * 0.01))

                            iou = calculate_iou(tbox, box)
                            if iou > 0.8:
                                found = next((i for i, mt in enumerate(tracking_box) if mt['id'] == id), None)

                                if found is not None:
                                    tracking_box[found]["update_tick"] = time.time()
                                    tracking_box[found]["bbox"] = box

                                    if tracking_box[found]['save'] is None:
                                        if calculate_iou(box, count_area) > 0:
                                            motorcycle_count += 1
                                            tracking_box[found]['save'] = True

                                            cropped_image = frame[box[1]:box[3], box[0]:box[2]]
                                            if tracking_box[found]['helmet'] is None:
                                                tracking_box[found]['helmet'] = "Processing"
                                                threading.Thread(target=person_segment, args=(cropped_image, tracking_box[found]['id']), daemon=True).start()

                                    color = COLORS.get('green') if tracking_box[found]['helmet'] else COLORS.get('yellow') if tracking_box[found]['helmet'] == "Processing" else COLORS.get('red') if tracking_box[found]['helmet'] is False else COLORS.get('silver')
                                    text = f"{tracking_box[found]['id']} {score} {'Helmet' if tracking_box[found]['helmet'] else 'No Helmet'}"
                                    cv2.rectangle(frame, (tracking_box[found]['bbox'][0], tracking_box[found]['bbox'][1]), (tracking_box[found]['bbox'][2], tracking_box[found]['bbox'][3]), color, 2)
                                    cv2.putText(frame, text, (tracking_box[found]['bbox'][2], tracking_box[found]['bbox'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS['gold'], 2)

                                if found is None:
                                    tracking_box.append({
                                        "id": id,
                                        "class": 1,
                                        "score": 0,
                                        "save": None,
                                        "helmet": None,
                                        "bbox": box,
                                        "update_tick": time.time()
                                    })
                                    if len(tracking_box) > 10:
                                        tracking_box.pop(0)

                # Display metrics
                fps = int(1000 / processing_time)
                cv2.putText(frame, f"{motorcycle_count} Motorcycle", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 230, 255), 3)
                cv2.putText(frame, f"{helmet_count} Helmet", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 230, 255), 3)
                cv2.putText(frame, f"{motorcycle_count - helmet_count} No Helmet", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 230, 255), 3)
                cv2.putText(frame, f"{fps} fps", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS['orange'], 2)
                cv2.rectangle(frame, (count_area[0], count_area[1]), (count_area[2], count_area[3]), (125, 110, 255), 2)

                steamFrame = frame.copy()

            except Exception as e:
                print("Error during frame processing:", e)
                steamFrame, status = None, False

    except Exception as e:
        print("Error setting up video capture or during processing:", e)
        steamFrame, status = None, False

    finally:
        if cap is not None:
            cap.release()

if __name__ == "__main__":
    config = load_config('config.ini')

    steamFrame = None
    save_vid = False
    status = False
    detected_data = []
    tracking_box = []
    motorcycle_count = 0
    helmet_count = 0
    monitor_id = 0
    resolution = [240, 360, 480, 720, 1080]

    motorcycle_model = None
    helmet_model = None
    rider_model = None

    # Configure Flask app
    app.config['SECRET_KEY'] = config['app']['secret_key']

    # Motorcycle Model Configuration
    motorcycle_file = config['motorcycle']['file']
    check_model_file(motorcycle_file, "motorcycle")
    motorcycle_conf = float(config['motorcycle']['confidence'])
    motorcycle_iou = float(config['motorcycle']['iou'])

    # Helmet Model Configuration
    helmet_file = config['helmet']['file']
    check_model_file(helmet_file, "helmet")
    helmet_conf = float(config['helmet']['confidence'])
    helmet_iou = float(config['helmet']['iou'])

    # Person Model Configuration
    person_file = config['person']['file']
    check_model_file(person_file, "person")
    person_conf = float(config['person']['confidence'])
    person_iou = float(config['person']['iou'])

    # Database Configuration
    DATABASE = Database(
        h=config['database']['host'],
        u=config['database']['user'],
        p=config['database']['password'],
        db=config['database']['dbname']
    )

    # Run the Flask app
    app.run(
        host=config['server']['host'],
        port=int(config['server']['port']),
        debug=config.getboolean('server', 'debug'),
        threaded=config.getboolean('server', 'threaded'),
        use_reloader=config.getboolean('server', 'use_reloader')
    )