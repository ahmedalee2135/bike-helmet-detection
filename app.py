from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import os
import cv2
import time
import numpy as np
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

MODEL_PATH = r'models\best.pt'
model = YOLO(MODEL_PATH)
print(f"Model loaded: {model.names}")

MODEL_INFO = {
    'name': 'YOLOv8n',
    'mAP50': 96.99,
    'mAP50_95': 69.61,
    'precision': 95.88,
    'recall': 93.54,
    'dataset_size': 9268,
    'epochs': 50
}

CONF_THRESHOLD = 0.65
IOU_THRESHOLD = 0.45
VIDEO_CONF = 0.55
VIDEO_IOU = 0.30

webcam = None


def generate_frames():
    global webcam
    webcam = cv2.VideoCapture(0)
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    webcam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        success, frame = webcam.read()
        if not success:
            break

        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                       imgsz=416, verbose=False)
        annotated = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html', model_info=MODEL_INFO)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_webcam')
def stop_webcam():
    global webcam
    if webcam is not None:
        webcam.release()
        webcam = None
    cv2.destroyAllWindows()
    return jsonify({'success': True})


def put_text_with_shadow(img, text, pos, color, font_scale=0.7, thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness, cv2.LINE_AA)


@app.route('/detect_image', methods=['POST'])
def detect_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'input_{timestamp}.jpg'
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        start_time = time.time()
        results = model(input_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
        inference_time = (time.time() - start_time) * 1000

        with_helmet = 0
        without_helmet = 0
        confidences = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]

                if label == 'With Helmet':
                    with_helmet += 1
                    confidences.append(conf)
                elif label == 'Without Helmet':
                    without_helmet += 1
                    confidences.append(conf)

        output_filename = f'result_{timestamp}.jpg'
        output_path = os.path.join(RESULT_FOLDER, output_filename)
        results[0].save(output_path)

        total = with_helmet + without_helmet
        helmet_rate = (with_helmet / total * 100) if total > 0 else 0
        avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0

        return jsonify({
            'success': True,
            'result_image': f'/static/results/{output_filename}',
            'with_helmet': with_helmet,
            'without_helmet': without_helmet,
            'total': total,
            'helmet_rate': round(helmet_rate, 1),
            'avg_confidence': round(avg_confidence, 1),
            'inference_time': round(inference_time, 1)
        })
    except Exception as e:
        print(f"ERROR in detect_image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/detect_video', methods=['POST'])
def detect_video():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'input_{timestamp}.mp4'
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            fps = 25

        output_filename = f'result_{timestamp}.mp4'
        output_path = os.path.join(RESULT_FOLDER, output_filename)

        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        tracked_riders = []
        total_with_helmet = 0
        total_without_helmet = 0

        confidences = []
        frame_count = 0
        start_time = time.time()

        DISTANCE_THRESHOLD = 100
        FORGET_AFTER = 150
        MIN_BOX_AREA = 1000

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            results = model(frame, conf=VIDEO_CONF, iou=VIDEO_IOU,
                           imgsz=640, verbose=False, agnostic_nms=True)

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]

                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < MIN_BOX_AREA:
                        continue

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    confidences.append(conf)

                    matched = False
                    for rider in tracked_riders:
                        if (rider['label'] == label and
                            abs(rider['x'] - cx) < DISTANCE_THRESHOLD and
                            abs(rider['y'] - cy) < DISTANCE_THRESHOLD and
                            (frame_count - rider['last_seen']) < FORGET_AFTER):
                            rider['x'] = cx
                            rider['y'] = cy
                            rider['last_seen'] = frame_count
                            matched = True
                            break

                    if not matched:
                        tracked_riders.append({
                            'x': cx, 'y': cy,
                            'label': label,
                            'last_seen': frame_count
                        })
                        if label == 'With Helmet':
                            total_with_helmet += 1
                        elif label == 'Without Helmet':
                            total_without_helmet += 1

            annotated = results[0].plot()

            put_text_with_shadow(annotated, f"Frame: {frame_count}/{total_frames}",
                                (20, 40), (0, 255, 0))
            put_text_with_shadow(annotated, f"With Helmet: {total_with_helmet}",
                                (20, 75), (0, 255, 136))
            put_text_with_shadow(annotated, f"Without Helmet: {total_without_helmet}",
                                (20, 110), (255, 8, 68))
            put_text_with_shadow(annotated, f"Total: {total_with_helmet + total_without_helmet}",
                                (20, 145), (0, 212, 255))

            out.write(annotated)

        cap.release()
        out.release()

        processing_time = time.time() - start_time
        total = total_with_helmet + total_without_helmet
        helmet_rate = (total_with_helmet / total * 100) if total > 0 else 0
        avg_confidence = (sum(confidences) / len(confidences) * 100) if confidences else 0

        return jsonify({
            'success': True,
            'result_video': f'/static/results/{output_filename}',
            'with_helmet': total_with_helmet,
            'without_helmet': total_without_helmet,
            'total': total,
            'helmet_rate': round(helmet_rate, 1),
            'avg_confidence': round(avg_confidence, 1),
            'frames_processed': frame_count,
            'processing_time': round(processing_time, 1)
        })
    except Exception as e:
        print(f"ERROR in detect_video: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open('http://127.0.0.1:5000')

    threading.Timer(1.5, open_browser).start()
    app.run(debug=False, host='0.0.0.0', port=5000)