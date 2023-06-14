import os
import shutil
import cv2
import pafy
import time
import imutils
import threading
import numpy as np
import webbrowser
import torch
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from flask import Flask, jsonify, request, render_template, Response
# from waitress import serve
# from imutils.video import VideoStream
from flask_cors import CORS
from utils.model import Model, letterbox, non_max_suppression, get_iou
from utils.sort import *
from utils.mysql import *

tracker = Sort()
app = Flask(__name__)
CORS(app)

def main(url):
    global steamFrame

    if "youtube" in url:
        video = pafy.new(url)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)
    else:
        cap = cv2.VideoCapture(url)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)

    if save_vid:
        record_ = cv2.VideoWriter('runs/videos/output.avi', 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            30, size)
    
    motorcycle_count = 0
    helmet_count = 0

    print("start main")
    while True:

        ref, frame = cap.read()
        s = time.time()

        #Resize image
        frame = imutils.resize(frame)

        #Create center line
        (height, width, cb) = frame.shape
        x = int(width/2)
        count_area = [x, 0, x+1, height]

        origin_img = frame.copy()
        image = frame.copy()

        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im = np.ascontiguousarray(im[0:1,...]/255)
        
        output = model.predict(im)
        pred = non_max_suppression(output, CONFIDENCE, IOU_THRESHOLD)
        
        tracking_box = []
        for i, DET in enumerate(pred): 
            if len(DET):
                #Class Filter
                midx = torch.where(DET[:, -1] == removeClass[0])[0]
                hidx = torch.where(DET[:, -1] == removeClass[1])[0]

                # delete the rows at the indices
                if DET.shape[0] == len(midx):
                    motorcycle_det = torch.tensor([])
                else:
                    motorcycle_det = torch.index_select(DET, 0, torch.tensor([i for i in range(DET.shape[0]) if i not in midx]))

                if DET.shape[0] == len(hidx):
                    helmet_det = torch.tensor([])
                else:
                    helmet_det = torch.index_select(DET, 0, torch.tensor([i for i in range(DET.shape[0]) if i not in hidx]))

                try:
                    #SORT (Simple Online and Realtime Tracker)
                    track_ids = tracker.update(motorcycle_det)
                    for i in range(len(track_ids.tolist())):
                        coords = track_ids.tolist()[i]

                        #Resize the image to original size
                        box = np.array(coords[:4])
                        box -= np.array(dwdh*2)
                        box /= ratio
                        box = box.round().astype(np.int32).tolist()
                        id = int(coords[4])

                        #ID Filter
                        mTrack = [element for element in tracked_box if element['id'] == id]
                    
                        if mTrack:
                            mTrack[0]["bbox"] = box
                            mTrack[0]["box"] = {
                                    'x1': box[0], 
                                    'x2': box[2],
                                    'y1': box[1],
                                    'y2': box[3]
                                }
                            mTrack[0]["update_tick"] = time.time()
                            tracking_box.append(mTrack[0])
                        else:
                            motorcycleData = {
                                "id": id,
                                "class": 1,
                                "score": 0,
                                "save": None,
                                "helmet": None,
                                "bbox": box,
                                "box":{
                                    'x1': box[0], 
                                    'x2': box[2],
                                    'y1': box[1],
                                    'y2': box[3]
                                },
                                "update_tick": time.time()
                            }
                            tracking_box.append(motorcycleData)
                except:
                    pass
                
                for i, d in enumerate(tracking_box):
                    B1 = {
                        'x1': count_area[0], 
                        'x2': count_area[2],
                        'y1': count_area[1],
                        'y2': count_area[3]
                    }
                    SCORE = get_iou(d['box'], B1)
                    if SCORE > 0 and not d['save']:
                        motorcycle_count += 1
                        tracking_box[i]['save'] = True

                        # tracker_frame[tracking_box[i]['id']] = origin_img.copy()
                        # cv2.rectangle(tracker_frame[tracking_box[i]['id']], (d["bbox"][0], d["bbox"][1]), (d["bbox"][2], d["bbox"][3]), (255, 0, 255), 2)
                        
                        #Cropped&Save image with Bounding box
                        cropped_image = origin_img[d['bbox'][1]:d['bbox'][3], d['bbox'][0]:d['bbox'][2]]
                        FN = f"runs/images/{d['id']}.png"
                        cv2.imwrite(FN, cropped_image)
                        print(f'{FN} Saved.')

                    for *xyxy, score, cls_id in reversed(helmet_det):
                        if score < 0.85: continue
                        cls_id = int(cls_id)
                        box = np.array(xyxy)
                        box -= np.array(dwdh*2)
                        box /= ratio
                        box = box.round().astype(np.int32).tolist()
                        helmetBox = {
                            'x1': box[0], 
                            'x2': box[2],
                            'y1': box[1],
                            'y2': box[3]
                        }
                        iou_score = get_iou(d["box"], helmetBox)

                        if iou_score != 0 and tracking_box[i]["helmet"] is None and tracking_box[i]["save"] == True:
                            tracking_box[i]["helmet"] = True
                            tracking_box[i]["update_tick"] = time.time()
                            helmet_count += 1

                        if iou_score != 0 and tracking_box[i]["helmet"] == True and tracking_box[i]["save"] == True:
                            N = f"runs/images/helmet/{d['id']}.png"
                            if not os.path.isfile(N):
                                shutil.move(f"runs/images/{d['id']}.png", N)
                                print(f"Move to {N}")

                    cv2.rectangle(frame, (d["bbox"][0], d["bbox"][1]), (d["bbox"][2], d["bbox"][3]), (255, 0, 255), 2)
                    if tracking_box[i]["helmet"]:
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 255), 2)
                        cv2.putText(frame, f"{tracking_box[i]['id']} Helmet", (d["box"]["x2"], d["box"]["y1"]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=2)
                    else:
                        cv2.putText(frame, f"{tracking_box[i]['id']} No Helmet", (d["box"]["x2"], d["box"]["y1"]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, thickness=2)

        tracked_box = tracking_box
        cv2.putText(frame, f"{motorcycle_count} Motorcycle", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 230, 255), thickness=3)
        cv2.putText(frame, f"{helmet_count} Helmet", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 230, 255), thickness=3)
        cv2.putText(frame, f"{motorcycle_count-helmet_count} No Helmet", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 230, 255), thickness=3)
        cv2.rectangle(frame, (count_area[0], count_area[1]), (count_area[2], count_area[3]), (125, 110, 255), 2)

        steamFrame = frame.copy()
        if save_vid:
            record_.write(frame)
        e = (time.time() - s)*100
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if save_vid:
        record_.release()
    cap.release()
    os._exit(0)

def generate():
    global steamFrame, lock
    while True:
        try:
            if steamFrame is None:
                time.sleep(0.03)
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", steamFrame)
            if not flag:
                time.sleep(0.03)
                continue
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        except:
            print("Image reading failed.")
            time.sleep(3)
            continue

@app.route("/api/steam")
def steam():
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

# @app.route("/")
# def index():
# 	return render_template("index.html")

if __name__ == "__main__":
    save_vid = False
    steamFrame = None
    lock = False
    color = [150,255,255]
    removeClass = [0.0, 1.0]
    tracked_box, tracker_ = [], []

    CONFIDENCE = 0.75
    IOU_THRESHOLD = 0.8

    model_file = r"..\model\best[hm]30l.onnx"
    # url = r"data\cctv_th.mp4"
    URL = "https://www.youtube.com/watch?v=qgNTbBn0JCY"

    #Model Load 
    model = Model(model_file)

    #Start 'main' function with threading
    t = threading.Thread(target=main, args=(URL,))
    t.daemon = True
    t.start()

    #Run flask server
    # webbrowser.open('http://127.0.0.1:8001/')
    app.run(host="0.0.0.0", port=8001, debug=True, threaded=True, use_reloader=False)
