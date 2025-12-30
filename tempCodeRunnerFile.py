from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
from pose_model import FitnessTrainer, ExerciseType
import cv2
import numpy as np
import time
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

trainer = FitnessTrainer()

@app.route('/start_camera')
def start_camera():
    """API endpoint to start camera - returns JSON status"""
    init_camera()
    if camera is not None:
        return jsonify({"status": "Camera started successfully"})
    else:
        return jsonify({"status": "Failed to start camera"}), 500

# -------- CAMERA SETUP --------
camera = None
camera_lock = threading.Lock()


def init_camera():
    global camera
    if camera is None:
        for idx in [0, 1, 2]:  # macOS fallback
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                camera = cap
                print(f"Camera opened at index {idx}")
                return
        print("❌ Could not open any camera")


def generate_frames():
    global camera
    init_camera()

    if camera is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "CAMERA NOT AVAILABLE",
            (120, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        ret, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )
        return

    while True:
        with camera_lock:
            success, frame = camera.read()

        if not success:
            break

        try:
            frame = trainer.process_frame(frame)

            if trainer.current_exercise != ExerciseType.NONE:
                current_fb = trainer.exercises[trainer.current_exercise]
                socketio.emit(
                    "stats_update",
                    {
                        "reps": current_fb.counter,
                        "feedback": current_fb.feedback,
                        "rate": f"{current_fb.rep_rate:.1f}",
                    },
                )

            ret, buffer = cv2.imencode(".jpg", frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

        except Exception as e:
            print("Camera processing error:", e)
            break


# -------- ROUTES --------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/stop_camera")
def stop_camera():
    """API endpoint to stop camera - returns JSON status"""
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None
    return jsonify({"status": "Camera stopped"})


@app.route("/reset_counters")
def reset_counters():
    """API endpoint to reset exercise counters"""
    for exercise in trainer.exercises.values():
        exercise.counter = 0
        exercise.rep_rate = 0
        exercise.feedback = ""
    return jsonify({"status": "Counters reset"})


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# -------- SOCKET EVENTS --------
@socketio.on("set_exercise")
def handle_exercise_change(exercise):
    exercise_map = {
        "bicep": ExerciseType.BICEP_CURL,
        "squat": ExerciseType.SQUAT,
        "lateral": ExerciseType.LATERAL_RAISE,
        "none": ExerciseType.NONE,
    }
    trainer.current_exercise = exercise_map.get(exercise, ExerciseType.NONE)
    socketio.emit("exercise_changed", {"exercise": exercise})


# -------- MAIN --------
if __name__ == "__main__":
    socketio.run(
        app,
        host="127.0.0.1",
        port=5000,
        debug=False,       # VERY IMPORTANT
        use_reloader=False  # VERY IMPORTANT
    )