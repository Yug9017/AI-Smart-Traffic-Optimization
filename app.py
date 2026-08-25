from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO
from pathlib import Path
import cv2
import time

app = Flask(__name__)
CORS(app)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "yolo26n.pt"
VIDEO_PATH = BASE_DIR / "traffic.mp4"

# ============================================================
# YOLO26
# ============================================================

print("Loading YOLO26n...")

model = YOLO(str(MODEL_PATH))

print("YOLO26n loaded successfully.")

# ============================================================
# GLOBAL VARIABLES
# ============================================================

cap = None

traffic_data = {
    "total_vehicles": 0,

    "lane_A": {
        "vehicle_count": 0,
        "green_light_duration": 20
    },

    "lane_B": {
        "vehicle_count": 0,
        "green_light_duration": 20
    },

    "traffic_density": "Low",
    "priority_lane": "A"
}


# ============================================================
# VEHICLE CLASSES
# ============================================================

# COCO:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck

VEHICLE_CLASSES = {2, 3, 5, 7}


# ============================================================
# ADAPTIVE SIGNAL LOGIC
# ============================================================

def calculate_signal_timing(count_a, count_b):

    total = count_a + count_b

    if total == 0:
        return 20, 20, "Low", "None"

    # Traffic density
    if total < 10:
        density = "Low"
    elif total < 20:
        density = "Medium"
    else:
        density = "High"

    # Give more green time to the busier direction.
    minimum_green = 15
    maximum_green = 60

    if count_a > count_b:

        difference = count_a - count_b

        green_a = min(
            maximum_green,
            max(minimum_green, 25 + difference * 2)
        )

        green_b = min(
            maximum_green,
            max(minimum_green, 25 - difference)
        )

        priority = "A"

    elif count_b > count_a:

        difference = count_b - count_a

        green_b = min(
            maximum_green,
            max(minimum_green, 25 + difference * 2)
        )

        green_a = min(
            maximum_green,
            max(minimum_green, 25 - difference)
        )

        priority = "B"

    else:

        green_a = 25
        green_b = 25
        priority = "Equal"

    return (
        int(green_a),
        int(green_b),
        density,
        priority
    )


# ============================================================
# VIDEO PROCESSING
# ============================================================

def generate_frames():

    global cap, traffic_data

    while True:

        if cap is None:

            time.sleep(0.1)

            continue

        success, frame = cap.read()

        # Restart video when finished
        if not success:

            print("Video finished. Restarting...")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            continue

        height, width = frame.shape[:2]

        # ====================================================
        # DEFINE TRAFFIC REGIONS
        # ====================================================

        # Your video has a central divider.
        #
        # Left side  = Direction A
        # Right side = Direction B
        #
        # We ignore the central divider.

        center_left = int(width * 0.45)
        center_right = int(width * 0.55)

        lane_a_polygon = [
            (0, height),
            (center_left, height),
            (center_left, int(height * 0.40)),
            (int(width * 0.46), int(height * 0.30)),
            (int(width * 0.40), int(height * 0.30)),
            (0, int(height * 0.45))
        ]

        lane_b_polygon = [
            (center_right, height),
            (width, height),
            (width, int(height * 0.45)),
            (int(width * 0.60), int(height * 0.30)),
            (int(width * 0.54), int(height * 0.30)),
            (center_right, int(height * 0.40))
        ]

        # ====================================================
        # YOLO26 TRACKING
        # ====================================================

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        lane_a_ids = set()
        lane_b_ids = set()

        # ====================================================
        # PROCESS DETECTIONS
        # ====================================================

        for result in results:

            if result.boxes is None:
                continue

            boxes = result.boxes

            for i in range(len(boxes)):

                cls = int(boxes.cls[i])

                if cls not in VEHICLE_CLASSES:
                    continue

                confidence = float(boxes.conf[i])

                if confidence < 0.35:
                    continue

                # Bounding box
                x1, y1, x2, y2 = map(
                    int,
                    boxes.xyxy[i]
                )

                # Vehicle center
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                # Tracking ID
                if boxes.id is not None:

                    track_id = int(boxes.id[i])

                else:

                    track_id = -1

                # ====================================================
                # DETERMINE REGION
                # ====================================================

                inside_a = cv2.pointPolygonTest(
                    __import__("numpy").array(
                        lane_a_polygon,
                        dtype="int32"
                    ),
                    (cx, cy),
                    False
                ) >= 0

                inside_b = cv2.pointPolygonTest(
                    __import__("numpy").array(
                        lane_b_polygon,
                        dtype="int32"
                    ),
                    (cx, cy),
                    False
                ) >= 0

                # ====================================================
                # COUNT VEHICLE
                # ====================================================

                if inside_a:

                    if track_id != -1:
                        lane_a_ids.add(track_id)

                    label = f"A | ID {track_id}"

                elif inside_b:

                    if track_id != -1:
                        lane_b_ids.add(track_id)

                    label = f"B | ID {track_id}"

                else:

                    label = f"ID {track_id}"

                # ====================================================
                # DRAW VEHICLE
                # ====================================================

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2
                )

        # ====================================================
        # COUNTS
        # ====================================================

        count_a = len(lane_a_ids)
        count_b = len(lane_b_ids)

        total = count_a + count_b

        # ====================================================
        # SIGNAL OPTIMIZATION
        # ====================================================

        green_a, green_b, density, priority = calculate_signal_timing(
            count_a,
            count_b
        )

        traffic_data["total_vehicles"] = total

        traffic_data["lane_A"]["vehicle_count"] = count_a
        traffic_data["lane_B"]["vehicle_count"] = count_b

        traffic_data["lane_A"]["green_light_duration"] = green_a
        traffic_data["lane_B"]["green_light_duration"] = green_b

        traffic_data["traffic_density"] = density
        traffic_data["priority_lane"] = priority

        # ====================================================
        # DRAW TRAFFIC REGIONS
        # ====================================================

        cv2.polylines(
            frame,
            [__import__("numpy").array(
                lane_a_polygon,
                dtype="int32"
            )],
            True,
            (255, 0, 0),
            3
        )

        cv2.polylines(
            frame,
            [__import__("numpy").array(
                lane_b_polygon,
                dtype="int32"
            )],
            True,
            (0, 0, 255),
            3
        )

        # Labels
        cv2.putText(
            frame,
            f"LANE A: {count_a}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

        cv2.putText(
            frame,
            f"LANE B: {count_b}",
            (width - 300, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        cv2.putText(
            frame,
            f"Density: {density}",
            (30, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Priority: {priority}",
            (30, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # ====================================================
        # ENCODE FRAME
        # ====================================================

        success, buffer = cv2.imencode(
            ".jpg",
            frame
        )

        if not success:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )


# ============================================================
# VIDEO FEED
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================
# TRAFFIC DATA API
# ============================================================

@app.route("/api/traffic-data")
def get_traffic_data():

    return jsonify(traffic_data)


# ============================================================
# START ANALYSIS
# ============================================================

@app.route("/api/start-analysis", methods=["POST"])
def start_analysis():

    global cap

    data = request.get_json(silent=True) or {}

    source = data.get("source")

    if cap is not None:

        cap.release()
        cap = None

    if source == "webcam":

        cap = cv2.VideoCapture(0)

    else:

        cap = cv2.VideoCapture(
            str(VIDEO_PATH)
        )

    if not cap.isOpened():

        return jsonify({
            "status": "error",
            "message": "Could not open video source"
        }), 500

    return jsonify({
        "status": "started",
        "model": "YOLO26n"
    })


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return jsonify({
        "status": "running",
        "model": "YOLO26n",
        "system": "AI Smart Traffic Optimization"
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("------------------------------------------")
    print("AI SMART TRAFFIC OPTIMIZATION")
    print("YOLO26n + ByteTrack")
    print("------------------------------------------")

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True,
        threaded=True
    )
