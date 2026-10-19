import os
from ultralytics import YOLO

# Trained model load karo
model = YOLO(r'models\best.pt')

# Test folder se pehli image uthao
test_images_dir = r'dataset\test\images'
images = os.listdir(test_images_dir)
first_image = os.path.join(test_images_dir, images[0])
print(f"Testing on: {images[0]}")

# Predict karo
results = model(first_image, conf=0.5)

# Output folder banao (agar nahi hai)
os.makedirs('output', exist_ok=True)

# Result save karo
results[0].save(r'output\result1.jpg')
print("✅ Result saved: output/result1.jpg")

# Classes dikhao
print("\nClasses:", model.names)

# Detections dikhao
print("\nDetections:")
for r in results:
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]
        print(f"  {label}: {conf:.2%}")