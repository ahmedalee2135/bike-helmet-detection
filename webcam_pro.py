import cv2
import time
import os
from ultralytics import YOLO
from collections import deque

# ================== SETTINGS ==================
MODEL_PATH = r'models\best.pt'
CONF_THRESHOLD = 0.95          # Zyada strict - false positive fix
IOU_THRESHOLD = 0.45           # Overlap kam
PROCESS_EVERY_N = 2            # Har 2nd frame process
DISPLAY_WIDTH = 960            # Display size
MIN_BOX_AREA_RATIO = 0.02      # Chhote boxes ignore
VIOLATION_CONF = 0.95          # Violation ke liye zyada strict
# ==============================================

# Folders banao
os.makedirs('output', exist_ok=True)
os.makedirs('violations', exist_ok=True)

# Model load karo
print("Model load ho raha hai...")
model = YOLO(MODEL_PATH)
print(f"Model loaded: {model.names}")

# Webcam open karo
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Webcam nahi khul raha")
    exit()

# Log file
log_file = open('violations_log.txt', 'a')
log_file.write(f"\n\n=== Session Started: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

print()
print("=" * 60)
print("🎥 Live Helmet Detection - Professional Version")
print("=" * 60)
print("Controls:")
print("  Q = Quit")
print("  S = Screenshot save")
print("=" * 60)
print()

# Variables
prev_time = time.time()
fps_buffer = deque(maxlen=30)
frame_count = 0
processed_count = 0
violation_count = 0
last_annotated = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame read nahi hua")
        break

    frame_count += 1

    # Frame skipping (smooth playback)
    if frame_count % PROCESS_EVERY_N != 0:
        if last_annotated is not None:
            display = cv2.resize(last_annotated, (DISPLAY_WIDTH, int(DISPLAY_WIDTH * 720 / 1280)))
            cv2.imshow('Live Helmet Detection', display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'output/webcam_{ts}.jpg', last_annotated)
                print(f"📸 Screenshot: output/webcam_{ts}.jpg")
        continue

    processed_count += 1

    # Detection chalao
    results = model(
        frame,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=640,
        verbose=False,
        agnostic_nms=True
    )

    # Violation check
    has_violation = False
    frame_area = frame.shape[0] * frame.shape[1]

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Box area check (chhote boxes ignore)
            xyxy = box.xyxy[0].cpu().numpy()
            box_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])

            if box_area / frame_area < MIN_BOX_AREA_RATIO:
                continue

            # Violation with high confidence
            if label == 'Without Helmet' and conf > VIOLATION_CONF:
                has_violation = True
                violation_count += 1

                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f'violations/violation_{ts}.jpg', frame)
                log_file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - Violation (conf: {conf:.2f})\n")
                log_file.flush()
                print(f"🚨 VIOLATION detected (conf: {conf:.2f})")

    # Annotated frame
    annotated = results[0].plot()

    # FPS calculate
    curr_time = time.time()
    fps_buffer.append(1 / (curr_time - prev_time))
    prev_time = curr_time
    avg_fps = sum(fps_buffer) / len(fps_buffer) * PROCESS_EVERY_N

    # Top-left overlay (dark background)
    cv2.rectangle(annotated, (0, 0), (320, 120), (0, 0, 0), -1)
    cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Frames: {frame_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated, f"Violations: {violation_count}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Violation alert (top-right)
    if has_violation:
        cv2.rectangle(annotated, (annotated.shape[1]-320, 10),
                     (annotated.shape[1]-10, 80), (0, 0, 200), -1)
        cv2.putText(annotated, "! VIOLATION !",
                    (annotated.shape[1]-310, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    last_annotated = annotated

    # Display (resize for smooth playback)
    display = cv2.resize(annotated, (DISPLAY_WIDTH, int(DISPLAY_WIDTH * 720 / 1280)))
    cv2.imshow('Live Helmet Detection', display)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n🛑 Q pressed - Closing...")
        break
    elif key == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f'output/webcam_{ts}.jpg', annotated)
        print(f"📸 Screenshot saved: output/webcam_{ts}.jpg")

# Cleanup
cap.release()
cv2.destroyAllWindows()
log_file.write(f"=== Session Ended: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
log_file.close()

# Summary
print()
print("=" * 60)
print("✅ Session Complete")
print("=" * 60)
print(f"📊 Total frames: {frame_count}")
print(f"📊 Processed frames: {processed_count}")
print(f"🚨 Total violations: {violation_count}")
print(f"📝 Log saved: violations_log.txt")
print("=" * 60)