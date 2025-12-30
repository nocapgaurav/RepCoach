# 🏋️ RepCoach – AI Exercise Tracker

RepCoach is an AI-powered exercise tracking web application that uses computer vision
and machine learning to analyze human posture in real time, count repetitions,
and provide feedback for exercises using a webcam.

This project acts as a **virtual AI fitness coach**, helping users perform exercises
with correct form at home.

---

## 🚀 Features

- 📷 Real-time webcam-based pose detection
- 🧠 AI-powered exercise recognition
- 🔢 Automatic repetition counting
- 📊 Live exercise statistics & feedback
- 🌐 Web-based interface
- 🔄 Real-time updates using Socket.IO

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask
- Flask-SocketIO

### AI / ML
- OpenCV
- MediaPipe (Pose Estimation)
- NumPy
- Scikit-learn (Random Forest)

---

## 📂 Project Structure

```
RepCoach/
├── app.py                          # Main Flask application
├── pose_model.py                   # Pose detection & exercise logic
├── pose2.py / pose3.py             # Model training & experiments
├── exercise_feedback_system_fixed.py
│
├── templates/
│   └── index.html                  # Web UI
│
├── static/
│   ├── script.js                   # Frontend logic
│   └── style.css                   # Styling
│
├── exercise_data.csv               # Dataset (optional)
├── .gitignore                      # Ignored files
├── README.md                       # Project documentation
```

> ⚠️ Note: Generated files like `.pkl`, `.pyc`, and `__pycache__` are intentionally ignored.

---

## ▶️ How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/nocapgaurav/RepCoach.git
cd RepCoach
```

---

### 2️⃣ Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate   # macOS / Linux
```

---

### 3️⃣ Install dependencies
```bash
pip install flask flask-socketio opencv-python mediapipe numpy scikit-learn
```

---

### 4️⃣ Run the application
```bash
python app.py
```

---

### 5️⃣ Open in browser
```
http://127.0.0.1:5000
```

Allow **camera access** when prompted.

---

## 🧠 Supported Exercises

- 💪 Bicep Curls
- 🦵 Squats
- 🤸 Lateral Raises

The system tracks:
- Repetitions
- Exercise state
- Live feedback
- Accuracy metrics

---

## 🎯 Use Cases

- Personal fitness tracking
- Home workout assistance
- AI & Computer Vision learning
- Final-year / portfolio project

---

## 👨‍💻 Author

**Gaurav Pandit**  
AI & Software Engineering Enthusiast  

GitHub: https://github.com/nocapgaurav

---

## ⭐ Acknowledgements

- MediaPipe by Google
- OpenCV community
- Flask & Python open-source ecosystem