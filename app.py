from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

cap = None

traffic_data = {
    "total_vehicles": 0,
    "lane_A": {
        "vehicle_count": 0,
        "green_light_duration": 10
    },
    "lane_B": {
        "vehicle_count": 0,
        "green_light_duration": 10
    }
}


def generate_frames():
    global cap, traffic_data

    while True:

        if cap is None:
            continue

        success, frame = cap.read()

        if not success:
            print("Video ended. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model(frame, verbose=False)

        total = 0

        for result in results:
            for box in result.boxes:

                cls = int(box.cls[0])

                # car, motorcycle, bus, truck
                if cls in [2, 3, 5, 7]:
                    total += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)

        traffic_data["total_vehicles"] = total

        lane_a = total // 2
        lane_b = total - lane_a

        traffic_data["lane_A"]["vehicle_count"] = lane_a
        traffic_data["lane_B"]["vehicle_count"] = lane_b

        traffic_data["lane_A"]["green_light_duration"] = max(10, lane_a * 2)
        traffic_data["lane_B"]["green_light_duration"] = max(10, lane_b * 2)

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        frame = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/api/traffic-data")
def traffic():
    return jsonify(traffic_data)


@app.route("/api/start-analysis", methods=["POST"])
def start_analysis():

    global cap

    data = request.get_json()

    source = data.get("source")

    print("Requested source:", source)

    if cap is not None:
        cap.release()

    if source == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture("traffic.mp4")

    if cap.isOpened():
        print("✅ Video opened successfully")
        return jsonify({
            "status": "started"
        })

    else:
        print("❌ Cannot open traffic.mp4")
        return jsonify({
            "status": "error",
            "message": "Cannot open traffic.mp4"
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)