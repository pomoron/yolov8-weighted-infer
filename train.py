from ultralytics import YOLO

# model = YOLO('yolov8l-seg.yaml')  # build a new model from YAML
# model = YOLO('./runs/segment/a14-orthonormal/weights/epoch40.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8l-seg.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8l-seg.yaml').load('yolov8l.pt')  # build from YAML and transfer weights

# Run MODE mode using the custom arguments ARGS (guess TASK)
results = model.train(data='custom.yaml', epochs=100, batch=6, save_period=5, name='a14-p', imgsz=640, amp=False, verbose=True)