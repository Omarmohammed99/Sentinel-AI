import os
import cv2
import numpy as np
import requests
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

REPO_ID = "omarrmohammed/Sentinel-AI-Core"

# ── 1. Smart Model Downloader & Loader ──────────────────────────────
def get_model_path(filename: str) -> str:
    """Checks if model exists locally; if not, downloads it from Hugging Face Hub."""
    if not os.path.exists(filename):
        print(f" Downloading {filename} from Hugging Face ({REPO_ID})...")
        try:
            return hf_hub_download(repo_id=REPO_ID, filename=filename)
        except Exception as e:
            print(f" Failed to download {filename} from HF: {e}")
            return filename
    return filename

print(" Loading Tier 1 & Tier 2 Models...")
try:
    t1_path = get_model_path("tier1_crash_detector.h5")
    t2_path = get_model_path("tier2_ego_detector.h5")

    tier1_model = load_model(t1_path, compile=False)
    tier2_model = load_model(t2_path, compile=False)
    print(" Tier 1 & Tier 2 Models Loaded Successfully!")
except Exception as e:
    print(f" Failed to load models: {e}")
    tier1_model = None
    tier2_model = None

# ── 2. FastAPI Setup & Schemas ──────────────────────────────────────
app = FastAPI(
    title="Sentinel AI Core API",
    description="API for detecting traffic accidents and ego-involvement.",
    version="1.0.0"
)

class MazenPayload(BaseModel):
    location: dict
    sensorData: dict
    videoUrl: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "location": {
                    "city": "Madinaty",
                    "coordinates": ["31.63604806401607", "30.098426398998807"],
                    "country": "Egypt",
                    "fullAddress": "72, Group 14, Madinaty, 19519, Egypt"
                },
                "sensorData": {
                    "angularVelocity": "364.585921904921",
                    "deltaV": "-13",
                    "peakGForce": "3.20",
                    "speedAtImpact": "113"
                },
                "videoUrl": "https://res.cloudinary.com/demo/video/upload/v1358880749/dog.mp4"
            }
        }
    )

# ── 3. Helper Functions ─────────────────────────────────────────────
def download_video(video_url: str) -> str:
    try:
        response = requests.get(video_url, stream=True, timeout=30)
        response.raise_for_status()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                temp_file.write(chunk)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        print(f"Video Download Error: {e}")
        return ""

def extract_frames(video_path, indices, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
        frames.append(frame)
    cap.release()
    frames_array = np.array(frames, dtype=np.float32) / 255.0
    return np.expand_dims(frames_array, axis=0)

# ── 4. API Endpoints ────────────────────────────────────────────────
@app.get("/")
def read_root():
    return {"message": "Sentinel AI Core API is Running! Go to /docs to test it."}

@app.post("/process-accident")
async def process_accident(payload: MazenPayload):
    mazen_data = payload.model_dump()
    video_url = mazen_data.get("videoUrl")

    if not video_url:
        raise HTTPException(status_code=400, detail="Missing videoUrl in payload")

    print(f" Received Request for Video: {video_url}")

    # 1. Download Video
    temp_vid_path = download_video(video_url)
    if not temp_vid_path:
        raise HTTPException(status_code=400, detail="Failed to download video from the provided URL")

    # 2. Run AI Inference (Tier 1 & Tier 2)
    try:
        if tier1_model is None or tier2_model is None:
            raise HTTPException(status_code=503, detail="AI Models are not loaded on the server")

        cap = cv2.VideoCapture(temp_vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames > 0:
            frame_indices = np.linspace(0, total_frames - 1, 16, dtype=int).tolist()
        else:
            frame_indices = [0] * 16

        frames = extract_frames(temp_vid_path, indices=frame_indices)

        # Predict Tier 1 (Crash Detection)
        t1_pred = tier1_model.predict(frames, verbose=0)[0][0]
        is_crash = bool(t1_pred > 0.5)

        # Predict Tier 2 (Ego Involvement)
        is_involved = False
        t2_pred = 0.0
        if is_crash:
            t2_pred = tier2_model.predict(frames, verbose=0)[0][0]
            is_involved = bool(t2_pred > 0.5)

        # Append AI results to payload response
        mazen_data["ai_analysis"] = {
            "crash_detected": is_crash,
            "crash_confidence": float(t1_pred),
            "ego_involved": is_involved,
            "ego_confidence": float(t2_pred)
        }

        print(f"AI Done! (Crash: {is_crash}, Involved: {is_involved}). Returning response.")
        return mazen_data

    except HTTPException:
        raise
    except Exception as e:
        print(f"AI Processing Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")
    finally:
        # Guarantee temporary video deletion to prevent storage memory leaks
        if os.path.exists(temp_vid_path):
            os.remove(temp_vid_path)
