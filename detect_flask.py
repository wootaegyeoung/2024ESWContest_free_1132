from flask import Flask, render_template, Response, jsonify
import cv2
import torch
import os
import time
import glob

app = Flask(__name__)

model_path= r"C:\Users\USER\Documents\KTH_4grade\yolov5_KTH_site\stream\yolo\runs\cafe_mangement2\weights\best.pt"
#model_path=r"C:\Users\USER\Documents\KTH_4grade\yolov5_KTH_site\stream\yolo\runs\cafe_mangement\weights\best.pt"
#model_path=r"C:\Users\USER\Documents\KTH_4grade\yolov5_KTH_site\stream\yolo\yolov5s.pt"
# Load custom model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Time variables
last_snapshot_time = 0  # Last snapshot time initialized to 0 or a past timestamp
snapshot_interval = 10  # Snapshot interval set to 30 minutes (in seconds)

# Alarm variables
alert_status = {'alert': False}
object_size_percentage = 0

snapshots_dir = "static/snapshots"
os.makedirs(snapshots_dir, exist_ok=True)

# Clear existing snapshots on startup
def clear_snapshots(directory):
    files = glob.glob(f'{directory}/*')
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Error deleting file {f}: {e}")

# Clear the snapshots directory before the app runs
clear_snapshots(snapshots_dir)

camera = cv2.VideoCapture(0)

def gen_frames():
    global object_size_percentage, last_snapshot_time
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                results = model(frame)
                total_box_size = 0
                frame_size = frame.shape[0] * frame.shape[1]

                for *xyxy, conf, cls in results.xyxy[0]:
                    class_name = model.names[int(cls)]  # Get the class name
                    #if class_name in ['person','Dirt', 'coffee_cup', 'Mugs']:  # Check if the class is 'Dirt'
                    x1, y1, x2, y2 = map(int, xyxy)
                    box_width = x2 - x1
                    box_height = y2 - y1
                    box_size = box_width * box_height
                    total_box_size += box_size

                    label = f'{class_name} {conf:.2f}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Calculate the percentage of the frame occupied by 'Dirt' objects
                object_size_percentage = (total_box_size / frame_size) * 100 if frame_size > 0 else 0
                closest_percentage = min([100, 80, 50, 30], key=lambda x: abs(x - object_size_percentage))
                color = (0, 255, 0) if closest_percentage <= 80 else (0, 0, 255)

                # Snapshot function (object detection with 30-minute interval)
                if any(model.names[int(cls)] in ['Phone', 'Wallet'] for *xyxy, conf, cls in results.xyxy[0]):
                    current_time = time.time()
                    if current_time - last_snapshot_time >= snapshot_interval:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        current_time_text = time.strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(frame, f'{current_time_text}', (10, frame.shape[0] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.imwrite(f'{snapshots_dir}/{timestamp}.jpg', frame)
                        cv2.putText(frame, f'Snapshot taken at {timestamp}', (10, frame.shape[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        last_snapshot_time = current_time

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                print(f"Error processing frame: {e}")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/alert', methods=['GET'])
def get_alert():
    return jsonify(alert_status)

@app.route('/object_size', methods=['GET'])
def get_object_size():
    return jsonify({'size_percentage': object_size_percentage})

@app.route('/')
def index():
    snapshots = sorted(os.listdir(snapshots_dir), reverse=True)
    snapshot_info = [
        {'filename': filename, 'timestamp': filename.split('.')[0]}
        for filename in snapshots
    ]
    return render_template('index.html', snapshot_info=snapshot_info)

@app.route('/snapshots')
def get_snapshots():
    snapshots = sorted(os.listdir(snapshots_dir), reverse=True)
    snapshot_list = [
        {'path': f'/static/snapshots/{filename}', 'timestamp': filename}
        for filename in snapshots
    ]
    return jsonify(snapshot_list)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
