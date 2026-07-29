import os
import uuid
import logging
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tf_keras.models import load_model
from huggingface_hub import hf_hub_download

# ── Logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("sentinel")

# ── App Config ─────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB upload cap
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET", "sentinel-dev-key")

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# ── Load AI Models ─────────────────────────────────────────────────
log.info("Loading Tier 1 & Tier 2 Models...")

HF_REPO_ID = "omarrmohammed/Sentinel-AI-Core"

def get_model_path(filename):
    """Checks if model exists locally, otherwise downloads it from Hugging Face Hub."""
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    if not os.path.exists(local_path):
        log.info(f"Model {filename} not found locally. Downloading from Hugging Face ({HF_REPO_ID})...")
        try:
            downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
            return downloaded_path
        except Exception as err:
            log.error(f"Failed to download {filename} from Hugging Face: {err}")
            return local_path
    return local_path

try:
    t1_path = get_model_path("tier1_crash_detector.h5")
    t2_path = get_model_path("tier2_ego_detector.h5")
    
    tier1_model = load_model(t1_path, compile=False)
    tier2_model = load_model(t2_path, compile=False)
    log.info("Models Loaded Successfully!")
except Exception as e:
    log.error(f"Failed to load models: {e}")
    tier1_model = None
    tier2_model = None

# ── Helpers ────────────────────────────────────────────────────────
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_frames(video_path, indices, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames_for_model = []
    base64_frames = [] 
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, target_size)
        
        # 1. Prepare image for the model (RGB + Normalization)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_for_model.append(rgb_frame)
        
        # 2. Extract base64 image for UI display
        _, buffer = cv2.imencode('.jpg', frame)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        base64_frames.append(f"data:image/jpeg;base64,{b64_str}")

    cap.release()
    
    frames_array = np.array(frames_for_model, dtype=np.float32) / 255.0
    return np.expand_dims(frames_array, axis=0), base64_frames 

# ── Routes ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if not tier1_model or not tier2_model:
        return jsonify({"error": "AI Models are not loaded. Check server logs."}), 500

    # ── 1. Validate & save the video file ──────────────────────────
    video_file = request.files.get("video")

    if not video_file or video_file.filename == "":
        return jsonify({"error": "No video file provided."}), 400

    if not allowed_file(video_file.filename):
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    ext = video_file.filename.rsplit(".", 1)[1].lower()
    unique_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, secure_filename(unique_name))

    try:
        video_file.save(save_path)
        log.info("Video saved → %s", save_path)
    except OSError as e:
        log.error("Failed to save video: %s", e)
        return jsonify({"error": "Server could not save the uploaded file."}), 500

    # ── 2. Run Local AI Inference ──────────────────────────────────
    try:
        cap = cv2.VideoCapture(save_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 2. Tier 1 Inference (Crash Detection - 16 Frames)
        t1_indices = np.linspace(0, total_frames - 1, 16, dtype=int).tolist()
        t1_input, t1_b64_images = extract_frames(save_path, t1_indices)
        
        t1_pred = tier1_model.predict(t1_input, verbose=0)[0][0]
        is_crash = bool(t1_pred > 0.5) 

        # 3. Tier 2 Inference (Ego-Involvement - 16 Frames)
        if is_crash:
            log.info("Crash Detected! Running Tier 2 (Ego-Involvement)...")
            t2_indices = np.linspace(0, total_frames - 1, 16, dtype=int).tolist()
            t2_input, _ = extract_frames(save_path, t2_indices)
            t2_pred = tier2_model.predict(t2_input, verbose=0)[0][0]
            is_involved = bool(t2_pred > 0.5)
        else:
            log.info("No Crash. Skipping Tier 2.")
            is_involved = False

        keyframe_idx = len(t1_b64_images) // 2
        keyframe_b64 = t1_b64_images[keyframe_idx] if t1_b64_images else ""

    except Exception as e:
        log.error("AI Processing Error: %s", e)
        if os.path.exists(save_path):
            os.remove(save_path)
        return jsonify({"error": f"AI Processing Error: {str(e)}"}), 500

    # Clean up video to save space
    if os.path.exists(save_path):
        os.remove(save_path)

    # ── 3. Build response payload ──────────────────────────────────
    result = {
        "status": "success",
        "crash_detected": is_crash,
        "ego_involved": is_involved,
        "yolo_image_base64": keyframe_b64,
        "ai_report_json": {
            "case_id": f"CAS-{uuid.uuid4().hex[:8].upper()}",
            "classification": "Crash Detected" if is_crash else "Safe",
            "liability_determination_ar": "Ego-Involved" if is_involved else "Not Involved" if is_crash else "N/A"
        }
    }

    log.info("Analysis complete — crash: %s, ego-involved: %s", is_crash, is_involved)
    return jsonify(result)

# ── Health Proxy ───────────────────────────────────────────────────
@app.route("/health")
def health_check():
    status = "online" if tier1_model and tier2_model else "degraded"
    return jsonify({"status": status}), 200 if status == "online" else 503

# ── 413 handler ────────────────────────────────────────────────────
@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({"error": "Video file too large. Maximum size is 500 MB."}), 413

if __name__ == "__main__":
    app.run(debug=False, port=5000, threaded=True)
