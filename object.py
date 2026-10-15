import cv2
import numpy as np
import time
import json
from pathlib import Path

import mysql.connector
from flask import Flask, Response, render_template, jsonify

try:
    from config import db_config
except ImportError:
    raise SystemExit(
        "config.py not found. Copy config.example.py to config.py and fill in "
        "your MySQL credentials."
    )

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

# Load YOLO
WEIGHTS_FILE = BASE_DIR / "yolov4.weights"
if not WEIGHTS_FILE.is_file():
    raise SystemExit(
        "yolov4.weights is missing. Run 'python download_weights.py' first."
    )

net = cv2.dnn.readNet(str(WEIGHTS_FILE), str(BASE_DIR / "yolov4.cfg"))
with open(BASE_DIR / "coco.names", "r") as f: classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Initialize Database
def init_db():
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']}")
        conn.close()

        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                class_name VARCHAR(255),
                confidence FLOAT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()
        print("Database initialized successfully.")
    except mysql.connector.Error as err:
        print(f"Database Error: {err}")

init_db()

# Connect to device default camera (webcam)
print("Attempting to connect to device camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam. Exiting.")
    exit(1)
    
print("Successfully connected to webcam.")

# Set optimal resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_skip = 1
frame_id = 0
colors = np.random.uniform(0, 255, size=(len(classes), 3))
prev_time = time.time()

# Global variable to store current frame counts
current_frame_counts = {}
last_log_time = {}

def log_detection(class_name, confidence):
    global last_log_time
    current_time = time.time()
    
    # Throttle logging: only log same class once every 2 seconds
    if class_name in last_log_time and current_time - last_log_time[class_name] < 2:
        return

    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        sql = "INSERT INTO detection_logs (class_name, confidence) VALUES (%s, %s)"
        val = (class_name, confidence)
        cursor.execute(sql, val)
        conn.commit()
        conn.close()
        last_log_time[class_name] = current_time
    except mysql.connector.Error as err:
        print(f"Logging Error: {err}")

def generate_frames():
    global frame_id, prev_time, current_frame_counts
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % frame_skip != 0:
            continue
            
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)
        
        boxes, confidences, class_ids = [], [], []
        temp_counts = {}

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                score = confidences[i]
                color = colors[class_ids[i]]
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update counts
                temp_counts[label] = temp_counts.get(label, 0) + 1
                
                # Log to DB
                log_detection(label, score)

        current_frame_counts = temp_counts

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time - prev_time > 0 else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # High quality JPEG encoding (95)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify(current_frame_counts)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
