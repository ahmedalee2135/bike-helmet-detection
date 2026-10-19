from ultralytics import YOLO

# Model load karo (nano = fastest)
model = YOLO('yolov8n.pt')

# Training shuru karo
model.train(
    data=r'dataset\data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    name='helmet_train'
)

# Evaluate karo
metrics = model.val()
print(f"\n=== RESULTS ===")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")