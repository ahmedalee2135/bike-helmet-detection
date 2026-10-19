import cv2
import os
import time
from ultralytics import YOLO

# ============ SETTINGS ============
MODEL_PATH = r'models\best.pt'
VIDEO_PATH = r'test_videos\video.mp4'
OUTPUT_PATH = r'output\result_video_pro.mp4'
CONF_THRESHOLD = 0.65      # Zyada strict (false positive kam)
IOU_THRESHOLD = 0.45       # Overlap kam
PROCESS_EVERY_N = 2        # Har 2nd frame process (jhatke kam)
DISPLAY_WIDTH = 960        # Display size (chhota = fast)
# ==================================

model = YOLO(MODEL_PATH)

if not os.path.exists(VIDEO_PATH):
    print(f"Video nahi mili: {VIDEO_PATH}")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {width}x{height} @ {fps}fps")
print(f"Total frames: {total_frames}")

os.makedirs('output', exist_ok=True)
os.makedirs('violations', exist_ok=True)

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

frame_count = 0
processed_count = 0
violation_count = 0
last_annotated = None
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Frame skipping - har PROCESS_EVERY_N frame process karo
    if frame_count % PROCESS_EVERY_N != 0:
        # Pichla result use karo (smooth playback)
        if last_annotated is not None:
            out.write(last_annotated)
        else:
            out.write(frame)
        continue

    processed_count += 1

    # Detection chalao - strict thresholds
    results = model(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=640,
        verbose=False,
        agnostic_nms=True  # Class overlap fix
    )

    # Violation check
    has_violation = False
    detected_labels = []

    for r in results:
        # Har box ka confidence check karo
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            detected_labels.append(f"{label}:{conf:.2f}")

            # Sirf high-confidence violations count karo
            if label == 'Without Helmet' and conf > 0.7:
                violation_count += 1
                has_violation = True

    # Annotated frame
    annotated = results[0].plot()

    # Info overlay
    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time) * PROCESS_EVERY_N
    prev_time = curr_time

    # Top-left: Stats
    cv2.rectangle(annotated, (0, 0), (350, 100), (0, 0, 0), -1)
    cv2.putText(annotated, f"FPS: {fps_display:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(annotated, f"Violations: {violation_count}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Violation alert
    if has_violation:
        cv2.rectangle(annotated, (width-320, 10), (width-10, 70), (0, 0, 200), -1)
        cv2.putText(annotated, "! VIOLATION !", (width-300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    last_annotated = annotated

    # Progress
    if processed_count % 20 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.1f}% | Detections: {', '.join(detected_labels[:3])}")

    out.write(annotated)

    # Display (resize for smooth view)
    display = cv2.resize(annotated, (DISPLAY_WIDTH, int(DISPLAY_WIDTH * height / width)))
    cv2.imshow('Helmet Detection - Professional', display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nUser stopped.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print()
print("=" * 60)
print(f"✅ Video saved: {OUTPUT_PATH}")
print(f"📊 Frames processed: {processed_count}/{frame_count}")
print(f"🚨 Total violations: {violation_count}")
print("=" * 60)