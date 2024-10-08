import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8m-seg.pt")
    model.train(data="datasets/data.yaml", epochs=300, batch=-1, device=0, imgsz=448)
    model.val()
    success = model.export(format="onnx", imgsz=320, opset=12)


