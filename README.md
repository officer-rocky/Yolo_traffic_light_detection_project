# 🚦 Real-time Traffic Light Detection using YOLOv8 + OpenCV

This project detects and classifies **traffic light states (Red, Yellow, Green)** in real-time from video or webcam feed using **YOLOv8** and **OpenCV**.  

## 📌 Features
- ✅ Detects traffic lights in video frames
- ✅ Classifies light states (RED, YELLOW, GREEN) using **HSV color space**
- ✅ Works on both video files and live webcam
- ✅ Bounding boxes and labels displayed in real-time
- ✅ Handles multiple traffic lights in one frame

## 📂 Project Structure
traffic_light_system/
│-- images/ # Screenshots (optional)
│-- traffic_light.py # Main Python script
│-- requirements.txt # Dependencies
│-- README.md # Documentation
│-- traffic2.mp4 # Sample traffic video (optional)

yaml
Copy code

---

## ⚡ Setup Instructions

### 1️⃣ Clone this repository
```bash
git clone https://github.com/your-username/Yolo_traffic_light_detection_project.git
cd Yolo_traffic_light_detection_project
2️⃣ Create and activate virtual environment
bash
Copy code
python -m venv traffic_env
traffic_env\Scripts\activate   # On Windows
3️⃣ Install dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Run the project
bash
Copy code
python traffic_light.py
🖼️ Screenshots
🔴 Red Light Detection

🟡 Yellow Light Detection

🟢 Green Light Detection



🛠️ Tech Stack
Python

OpenCV

NumPy

YOLOv8 (Ultralytics)

🎯 Evaluation
✔️ Detects traffic lights in real-time

✔️ Correctly classifies RED, YELLOW, GREEN states

✔️ Annotates video feed with bounding boxes + labels

📌 Deliverables
Python script (traffic_light.py)

Real-time annotated output

Screenshots in images/

Report on detection accuracy