#  Sentinel AI — Automated Traffic Accident Detection & Response System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Deep Learning](https://img.shields.io/badge/Model-MobileNet%20%2B%20LSTM-orange.svg)](https://www.tensorflow.org/)
[![LLM](https://img.shields.io/badge/LLM-Gemini%20Flash-purple.svg)](https://deepmind.google/technologies/gemini/)
[![Hugging Face](https://img.shields.io/badge/Model%20Hub-Hugging%20Face-yellow.svg)](https://huggingface.org/)

An end-to-end AI-powered traffic safety pipeline designed to process real-time video streams, detect vehicular accidents, classify ego-vehicle involvement, and generate automated incident reports using Vision-Language Models (VLMs).

---

##  Key Features

- **Multi-Tier AI Pipeline:**
  - **Tier 1 (Crash Detection):** Sequential MobileNet + LSTM model to detect traffic accidents from video frames in real time.
  - **Tier 2 (Ego-Involvement Classification):** Secondary classification layer to determine if the primary captured vehicle (ego-vehicle) is actively involved in the incident.
- **VLM Reasoning Layer:** Powered by **Gemini Flash** to analyze keyframes, extract context, and automatically generate structured accident summaries.
- **Low-Latency Architecture:** Optimized frame extraction (16-frame uniform sampling) for rapid model inference.
- **Automated Weight Downloading:** Seamless integration with **Hugging Face Hub** to fetch model weights automatically on initial launch.
- **RESTful API Service:** Built with Flask for smooth integration into web, mobile, or dashboard applications.

---

##  System Architecture

[ Video Input ]
│
▼
[ Frame Sampling ] ──► (Extract 16 Uniform Frames via OpenCV)
│
▼
[ Tier 1 Inference ] ──► (MobileNet + LSTM Crash Classifier)
│
├──► [ No Crash ] ──► Return Safe Payload
│
└──► [ Crash Detected ]
│
▼
[ Tier 2 Inference ] ──► (Ego-Vehicle Involvement Analysis)
│
▼
[ Gemini Flash API ] ──► (Generate Structured Summary Report)
│
▼
[ JSON API Response + Keyframe Payload ]


---

##  Tech Stack

- **Core & Backend:** Python, Flask, OpenCV, Werkzeug
- **Deep Learning & CV:** TensorFlow, Keras (`tf-keras`), MobileNet, LSTM
- **Generative AI:** Gemini Flash API
- **Model Hosting & Deployment:** Hugging Face Hub, Docker

---

##  Quick Start & Installation

### 1. Prerequisites
Make sure you have **Python 3.10+** and `git` installed.

### 2. Clone the Repository
```bash
git clone [https://github.com/omarrrmohammed/sentinel-ai.git](https://github.com/omarrrmohammed/sentinel-ai.git)
cd sentinel-ai
3. Create a Virtual Environment & Install Dependencies
Bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
4. Set Environment Variables (Optional)
If you are integrating the Gemini VLM module, set your Gemini API key:

Bash
export GEMINI_API_KEY="your-api-key-here"     # Linux/macOS
set GEMINI_API_KEY="your-api-key-here"        # Windows Command Prompt
5. Run the Application
Bash
python main.py
Note: On the first run, the system will automatically download tier1_crash_detector.h5 and tier2_ego_detector.h5 from Hugging Face Hub if they are not present locally.

The app will be running at http://127.0.0.1:5000.

 API Endpoints
POST /analyze
Uploads a video file for crash inference.

Request: multipart/form-data with video file attachment (.mp4, .avi, .mov, .mkv).

Response Example:


{
  "status": "success",
  "crash_detected": true,
  "ego_involved": true,
  "yolo_image_base64": "data:image/jpeg;base64,...",
  "ai_report_json": {
    "case_id": "CAS-8F3BD912",
    "classification": "Crash Detected",
    "liability_determination_ar": "Ego-Involved"
  }
}
GET /health
Returns system operational status./

🔗 Live Demo & Links : https://sentinel-ai-eight-kohl.vercel.app/
Hugging Face Model Repository: Sentinel AI on Hugging Face

Author : Omar Mohamed



Omar Mohamed

Email: ommo9745@gmail.com
