from ultralytics import YOLO
from ultralytics.utils.ops import scale_masks, scale_image
import cv2
import random
import time
import numpy as np
from flask import Flask, jsonify, request, render_template, Response

app = Flask(__name__)

labels = ['H', 'NH']
steamFrame = None
lock = False
label = None

def generate():
    while True:
        if steamFrame is None:
            time.sleep(0.03)
            continue
        (flag, encodedImage) = cv2.imencode(".jpg", steamFrame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route("/app")
def steam():
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/update")
def update():
    global label
    label = int(request.args.get("label"))
    print(f"Label is: {label}")
    return jsonify({"status": label})

@app.route("/")
def index():
	return render_template("index.html")

def start_app():
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True, use_reloader=False)

def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    
    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    # color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined
            

def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

        # cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        # cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



# Load a model
model = YOLO('yolov8n-seg.pt')
imgPath = r'data/test2.jpg'


class_names = ['person']
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]

results = model.predict(imgPath, classes=[0], save_txt=True)

for r in results:
    img = r.orig_img
    h, w = r.orig_shape
    boxes = r.boxes  # Boxes object for bbox outputs
    masks = r.masks  # Masks object for segment masks outputs
    probs = r.probs  # Class probabilities for classification outputs

    if masks is not None:
        masks = masks.data.cpu()
        print(masks.data.cpu().numpy())
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            img = overlay(img, seg, colors[int(box.cls)], 0.7)
            
            xmin = int(box.data[0][0])
            ymin = int(box.data[0][1])
            xmax = int(box.data[0][2])
            ymax = int(box.data[0][3])
            
            plot_one_box([xmin, ymin, xmax, ymax], img, colors[int(box.cls)], f'{class_names[int(box.cls)]} {float(box.conf):.3}')

            steamFrame = img

            while True:
                if label is not None:
                    if label == 3:
                        label = None
                        break
                    txt = f"{label} {box}"
                    label_data.append(txt)
                    label = None
                    break
                time.sleep(0.1)
