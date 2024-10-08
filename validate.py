from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("models/segment_driver-v8n.pt")

    # Validate with a test dataset
    metrics = model.val(data="datasets/data.yaml", imgsz=320)
    
    print(f'p_curve: {metrics.box.p_curve}')  # p_curve
    print(f'r_curve: {metrics.box.r_curve}')  # p_curve
    print(f'Presision: {metrics.box.p[0]}')  # Presision
    print(f'Recall: {metrics.box.r[0]}')  # Recall
    print(f'F1: {metrics.box.f1[0]}')  # F1
    print(f'mAp: {metrics.box.map}')  # mAp
    print(f'Inference: {metrics.speed["inference"]:.2f}ms')  # Inference
