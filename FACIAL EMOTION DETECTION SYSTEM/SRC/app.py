from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPool2D, LSTM, Dense, Dropout, BatchNormalization, MaxPooling2D, Activation, Reshape, Input
import numpy as np
import cv2
import os
import platform
import warnings
import threading
from datetime import datetime, timezone
from pathlib import Path

# Filter out the AVCapture deprecation warning
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Initialize FastAPI app with additional settings
app = FastAPI(
    title="Facial Emotion Recognition API",
    description="API for real-time emotion detection from webcam",
    version="1.0.0",
)
# Root and health endpoints
@app.get("/")
async def root():
    return {"message": "Emotion API running", "docs": "/docs", "predict": "/predict", "predict_image": "/predict_image"}

@app.get("/health")
async def health():
    return {"status": "ok"}


# Add CORS middleware with specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Platform-specific settings
SYSTEM = platform.system().lower()
if SYSTEM == 'windows':
    os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
elif SYSTEM == 'darwin':  # macOS
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_stream_provider=0"

def capture_image():
    try:
        # For macOS, specifically use AVFoundation backend
        if SYSTEM == 'darwin':
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            print("Failed to open camera")
            return None
            
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Higher resolution for better face detection
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Wait a moment for camera to initialize
        for _ in range(5):
            cap.read()
            
        # Capture frame
        ret, frame = cap.read()
        
        # Release camera
        cap.release()
        
        if not ret or frame is None:
            print("Failed to capture frame")
            return None
            
        return frame
        
    except Exception as e:
        print(f"Error capturing image: {e}")
        return None

def capture_frames(num_frames=5):
    """Capture multiple frames for prediction smoothing."""
    try:
        if SYSTEM == 'darwin':
            cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Failed to open camera")
            return []

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frames = []
        # Warm up
        for _ in range(5):
            cap.read()

        for _ in range(max(1, num_frames)):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)

        cap.release()
        return frames
    except Exception as e:
        print(f"Error capturing frames: {e}")
        return []

# Define model architecture
def create_model():
    model = Sequential()
    model.add(Input(shape=(48, 48, 1)))
    model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), strides=(1,1), padding='valid'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3,3), strides=(1,1), padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128, (3,3), strides=(1,1), padding='valid'))
    model.add(BatchNormalization(axis=3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Reshape((-1,128)))
    model.add(LSTM(128))
    model.add(Reshape((-1,64)))
    model.add(LSTM(64))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(7, activation='softmax'))
    return model

# Create and compile model
try:
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Load weights
    model.load_weights('fer2013_bilstm_cnn.h5')
    print("Model created and weights loaded successfully")
except Exception as e:
    print(f"Error setting up model: {e}")
    raise

# Emotion labels
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Simple stress score map (0=none, 1=high). Tunable.
STRESS_SCORE = {
    'Angry': 0.9,
    'Disgust': 0.6,
    'Fear': 0.9,
    'Happy': 0.1,
    'Sad': 0.7,
    'Surprise': 0.4,
    'Neutral': 0.2,
}

# In-memory store; replace with DB in production
STRESS_LOG = []  # list of dicts {ts, emotion, confidence, score}
STRESS_LOCK = threading.Lock()
LOG_DIR = Path('data')
LOG_FILE = LOG_DIR / 'stress_events.csv'

def _ensure_log_dir():
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _append_csv(row: dict):
    _ensure_log_dir()
    header = 'ts,emotion,confidence,score_expected,score_primary,calendar_tag'\
        if not LOG_FILE.exists() else None
    line = f"{row.get('ts')},{row.get('emotion')},{row.get('confidence')},{row.get('score_expected')},{row.get('score_primary')},{row.get('calendar_tag','')}\n"
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        if header:
            f.write(header + "\n")
        f.write(line)

def get_camera_index():
    """Get the appropriate camera index based on the platform"""
    if SYSTEM == 'darwin':
        # Try to find built-in camera first on macOS
        for i in range(4):  # Check first 4 indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    return i
        return 0  # Default to 0 if no camera found
    return 0  # Default for other platforms

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces (use scaleFactor and minNeighbors tuned for fewer false positives)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)

    if len(faces) == 0:
        return None

    # Choose the largest face (more likely the main subject)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    (x, y, w, h) = faces[0]

    # Add margin around the face box to capture full expression context
    margin = int(0.15 * max(w, h))
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(gray.shape[1], x + w + margin)
    y1 = min(gray.shape[0], y + h + margin)

    face = gray[y0:y1, x0:x1]

    # Resize to 48x48 pixels
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)

    # Contrast Limited Adaptive Histogram Equalization (helps non-neutral classes)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)

    # Normalize pixel values
    face = face.astype('float32') / 255.0

    # Reshape for model input
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    return face

def get_recommendations(emotions):
    """Return simple recommendations for one or more emotions."""
    rec_map = {
        'Angry': ["Take 5 deep breaths", "Step away briefly", "Relax your jaw and shoulders"],
        'Disgust': ["Identify the trigger", "Refocus attention on a neutral object"],
        'Fear': ["Grounding: name 5 things you see", "Assess actual safety"],
        'Happy': ["Share your joy with someone", "Note what led to this state"],
        'Sad': ["Reach out to a friend", "Go for a short walk"],
        'Surprise': ["Pause and observe", "Adjust expectations if needed"],
        'Neutral': ["Maintain good posture", "Try a gentle smile"]
    }
    seen = set()
    out = []
    for emo in emotions:
        for tip in rec_map.get(emo, []):
            if tip not in seen:
                out.append(tip)
                seen.add(tip)
    return out

def predict_single(image):
    processed = preprocess_image(image)
    if processed is None:
        return None
    preds = model.predict(processed)
    return preds[0]

def get_model_weight_stats():
    """Compute simple statistics of model weights to infer if it's trained."""
    layer_summaries = []
    std_values = []
    zero_like_layers = 0
    total_layers_with_weights = 0
    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue
        total_layers_with_weights += 1
        # Flatten all tensors in this layer
        w_concat = np.concatenate([w.flatten() for w in weights if w.size > 0])
        w_std = float(np.std(w_concat))
        w_mean_abs = float(np.mean(np.abs(w_concat)))
        zero_ratio = float(np.mean(w_concat == 0))
        std_values.append(w_std)
        if w_std < 1e-6 and w_mean_abs < 1e-6:
            zero_like_layers += 1
        layer_summaries.append({
            "layer": layer.name,
            "std": w_std,
            "mean_abs": w_mean_abs,
            "zeros_ratio": zero_ratio
        })

    median_std = float(np.median(std_values)) if std_values else 0.0
    likely_trained = (median_std > 1e-3) and (zero_like_layers == 0)
    return {
        "total_layers_with_weights": total_layers_with_weights,
        "median_weight_std": median_std,
        "zero_like_layers": zero_like_layers,
        "likely_trained": likely_trained,
        "layers": layer_summaries[:10]  # cap to keep response small
    }

@app.get("/model_status")
async def model_status():
    try:
        stats = get_model_weight_stats()
        return {
            "model_input_shape": list(model.input_shape) if hasattr(model, 'input_shape') else None,
            "num_classes": len(EMOTIONS),
            "emotions": EMOTIONS,
            "weights": stats
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/quickcheck")
async def quickcheck(num_frames: int = 15):
    """Capture multiple frames and report prediction distribution to spot bias."""
    try:
        frames = capture_frames(num_frames=max(5, min(30, num_frames)))
        if len(frames) == 0:
            return JSONResponse(content={"error": "Failed to capture from webcam"}, status_code=500)
        counts = {e: 0 for e in EMOTIONS}
        probs_accum = np.zeros(len(EMOTIONS), dtype=np.float32)
        valid = 0
        for f in frames:
            p = predict_single(f)
            if p is None:
                continue
            valid += 1
            probs_accum += p.astype(np.float32)
            counts[EMOTIONS[int(np.argmax(p))]] += 1
        if valid == 0:
            return JSONResponse(content={"error": "No face detected in any frame"}, status_code=400)
        mean_probs = (probs_accum / float(valid)).tolist()
        distribution = {k: v for k, v in counts.items()}
        top_idx = int(np.argmax(mean_probs))
        return {
            "frames_processed": valid,
            "top_emotion": EMOTIONS[top_idx],
            "top_confidence": float(mean_probs[top_idx]),
            "distribution": distribution,
            "mean_probabilities": {emo: float(mean_probs[i]) for i, emo in enumerate(EMOTIONS)}
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/predict")
async def predict_emotion(calendar_tag: str = ""):
    # Capture multiple frames and average predictions for stability
    frames = capture_frames(num_frames=7)

    if len(frames) == 0:
        return JSONResponse(
            content={"error": "Failed to capture image from webcam"},
            status_code=500
        )

    preds_list = []
    for f in frames:
        p = predict_single(f)
        if p is not None:
            preds_list.append(p)

    if len(preds_list) == 0:
        return JSONResponse(
            content={"error": "No face detected in the image"},
            status_code=400
        )

    predictions = np.mean(np.stack(preds_list, axis=0), axis=0)

    # Top emotions
    top_indices = np.argsort(predictions)[::-1]
    primary_idx = int(top_indices[0])
    primary = EMOTIONS[primary_idx]
    primary_conf = float(predictions[primary_idx])

    # If another emotion is close, include it as a secondary mention
    secondary_emotions = []
    threshold_delta = 0.12  # within 12% probability considered a tie/relative mention
    for idx in top_indices[1:3]:
        if float(predictions[idx]) >= primary_conf - threshold_delta:
            secondary_emotions.append(EMOTIONS[int(idx)])

    emotions_detected = [primary] + secondary_emotions
    recs = get_recommendations(emotions_detected)

    # Compute stress scores
    expected_stress = float(np.sum([float(p) * STRESS_SCORE[e] for p, e in zip(predictions, EMOTIONS)]))
    primary_stress = float(STRESS_SCORE.get(primary, 0.5))

    # Log
    event = {
        'ts': datetime.now(timezone.utc).isoformat(),
        'emotion': primary,
        'confidence': primary_conf,
        'score_expected': expected_stress,
        'score_primary': primary_stress,
        'calendar_tag': calendar_tag or ""
    }
    with STRESS_LOCK:
        STRESS_LOG.append(event)
    _append_csv(event)

    return {
        "emotion": primary,
        "confidence": primary_conf,
        "emotions_detected": emotions_detected,  # multiple mentions separated as list
        "recommendations": recs,
        "stress": {
            "expected": expected_stress,
            "primary": primary_stress
        },
        "all_probabilities": {
            emotion: float(prob)
            for emotion, prob in zip(EMOTIONS, predictions)
        }
    }

# New endpoint: predict from uploaded image (frontend snapshot)
@app.post("/predict_image")
async def predict_emotion_from_image(file: UploadFile = File(...), calendar_tag: str = ""):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse(content={"error": "Invalid image"}, status_code=400)
        preds = predict_single(image)
        if preds is None:
            return JSONResponse(content={"error": "No face detected in the image"}, status_code=400)
        top_indices = np.argsort(preds)[::-1]
        primary_idx = int(top_indices[0])
        primary = EMOTIONS[primary_idx]
        primary_conf = float(preds[primary_idx])
        secondary_emotions = []
        threshold_delta = 0.12
        for idx in top_indices[1:3]:
            if float(preds[idx]) >= primary_conf - threshold_delta:
                secondary_emotions.append(EMOTIONS[int(idx)])
        emotions_detected = [primary] + secondary_emotions
        recs = get_recommendations(emotions_detected)
        expected_stress = float(np.sum([float(p) * STRESS_SCORE[e] for p, e in zip(preds, EMOTIONS)]))
        primary_stress = float(STRESS_SCORE.get(primary, 0.5))
        event = {
            'ts': datetime.now(timezone.utc).isoformat(),
            'emotion': primary,
            'confidence': primary_conf,
            'score_expected': expected_stress,
            'score_primary': primary_stress,
            'calendar_tag': calendar_tag or ""
        }
        with STRESS_LOCK:
            STRESS_LOG.append(event)
        _append_csv(event)
        return {
            "emotion": primary,
            "confidence": primary_conf,
            "emotions_detected": emotions_detected,
            "recommendations": recs,
            "stress": {
                "expected": expected_stress,
                "primary": primary_stress
            },
            "all_probabilities": {
                emotion: float(prob)
                for emotion, prob in zip(EMOTIONS, preds)
            }
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

def _parse_iso(ts_str: str) -> datetime:
    try:
        dt = datetime.fromisoformat(ts_str)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None

@app.get("/events")
async def list_events(start: str = "", end: str = "", calendar_tag: str = ""):
    """Return logged events from memory and CSV within an optional date range and tag."""
    with STRESS_LOCK:
        rows = list(STRESS_LOG)
    # Also read CSV if exists to include past runs
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                next(f, None)
                for line in f:
                    parts = line.rstrip("\n").split(',')
                    if len(parts) >= 5:
                        rows.append({
                            'ts': parts[0],
                            'emotion': parts[1],
                            'confidence': float(parts[2]),
                            'score_expected': float(parts[3]),
                            'score_primary': float(parts[4]),
                            'calendar_tag': parts[5] if len(parts) > 5 else ""
                        })
        except Exception:
            pass

    start_dt = _parse_iso(start) if start else None
    end_dt = _parse_iso(end) if end else None

    out = []
    for r in rows:
        ts = _parse_iso(r.get('ts', ''))
        if ts is None:
            continue
        if start_dt and ts < start_dt:
            continue
        if end_dt and ts > end_dt:
            continue
        if calendar_tag and (r.get('calendar_tag', '') != calendar_tag):
            continue
        out.append({
            'ts': ts.isoformat(),
            'emotion': r.get('emotion'),
            'confidence': float(r.get('confidence', 0.0)),
            'score_expected': float(r.get('score_expected', 0.0)),
            'score_primary': float(r.get('score_primary', 0.0)),
            'calendar_tag': r.get('calendar_tag', '')
        })
    out.sort(key=lambda x: x['ts'])
    return {"events": out}

@app.get("/stress_intervals")
async def stress_intervals(threshold: float = 0.6, min_gap_minutes: float = 5.0):
    """Group consecutive high-stress events into intervals and compute durations.
    threshold: expected stress >= threshold is considered stressed
    min_gap_minutes: if the gap between stressed samples exceeds this, start a new interval
    """
    with STRESS_LOCK:
        rows = list(STRESS_LOG)
    events = []
    for r in rows:
        ts = _parse_iso(r.get('ts', ''))
        if ts is None:
            continue
        score = float(r.get('score_expected', 0.0))
        events.append({'ts': ts, 'score': score, 'emotion': r.get('emotion')})
    events.sort(key=lambda x: x['ts'])

    if not events:
        return {"intervals": []}

    intervals = []
    current = None
    max_gap = min_gap_minutes * 60.0
    for ev in events:
        if ev['score'] >= threshold:
            if current is None:
                current = {
                    'start': ev['ts'],
                    'end': ev['ts'],
                    'peak_score': ev['score'],
                    'peak_ts': ev['ts']
                }
            else:
                # Check gap
                if (ev['ts'] - current['end']).total_seconds() > max_gap:
                    # Close previous interval
                    duration_min = (current['end'] - current['start']).total_seconds() / 60.0
                    intervals.append({
                        'start': current['start'].isoformat(),
                        'end': current['end'].isoformat(),
                        'duration_minutes': duration_min,
                        'peak_ts': current['peak_ts'].isoformat(),
                        'peak_score': current['peak_score']
                    })
                    # Start new
                    current = {
                        'start': ev['ts'],
                        'end': ev['ts'],
                        'peak_score': ev['score'],
                        'peak_ts': ev['ts']
                    }
                else:
                    current['end'] = ev['ts']
                    if ev['score'] > current['peak_score']:
                        current['peak_score'] = ev['score']
                        current['peak_ts'] = ev['ts']
        else:
            # Non-stressed sample; may close interval if gap grows when next stressed sample appears
            pass

    if current is not None:
        duration_min = (current['end'] - current['start']).total_seconds() / 60.0
        intervals.append({
            'start': current['start'].isoformat(),
            'end': current['end'].isoformat(),
            'duration_minutes': duration_min,
            'peak_ts': current['peak_ts'].isoformat(),
            'peak_score': current['peak_score']
        })

    return {"intervals": intervals}

def _period_key(ts: datetime, period: str):
    if period == 'day':
        return ts.date().isoformat()
    if period == 'week':
        iso = ts.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"
    if period == 'month':
        return f"{ts.year}-{ts.month:02d}"
    if period == 'year':
        return f"{ts.year}"
    return ts.date().isoformat()

@app.get("/stress_summary")
async def stress_summary(period: str = 'day', days: int = 30, focus_threshold: float = 0.35, peak_threshold: float = 0.75, top_n_peaks: int = 3):
    """Summarize stress over a recent window.
    period: one of day|week|month|year
    days: lookback window (ignored for year-level if too large)
    """
    period = period.lower()
    if period not in {'day', 'week', 'month', 'year'}:
        return JSONResponse(content={"error": "Invalid period. Use day|week|month|year."}, status_code=400)
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - float(days) * 86400.0

    with STRESS_LOCK:
        rows = list(STRESS_LOG)

    agg = {}
    count = {}
    primary_agg = {}
    focus_count = {}
    max_peak = {}
    # Also collect for overall peaks and avg interval
    window_events = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(r['ts'])
        except Exception:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts.timestamp() < cutoff:
            continue
        score_exp = float(r.get('score_expected', 0.0))
        score_pri = float(r.get('score_primary', 0.0))
        key = _period_key(ts, period)
        agg[key] = agg.get(key, 0.0) + score_exp
        primary_agg[key] = primary_agg.get(key, 0.0) + score_pri
        count[key] = count.get(key, 0) + 1
        if score_exp <= focus_threshold:
            focus_count[key] = focus_count.get(key, 0) + 1
        # track max peak per period
        if score_exp >= (max_peak.get(key, {"score": -1}).get("score", -1)):
            max_peak[key] = {"ts": ts.isoformat(), "score": score_exp, "emotion": r.get('emotion')}
        # for overall peaks and interval
        window_events.append({"ts": ts, "score": score_exp, "emotion": r.get('emotion')})

    result = []
    # Estimate average interval to convert samples to minutes
    avg_interval_sec = 0.0
    if len(window_events) >= 2:
        window_events_sorted = sorted(window_events, key=lambda x: x['ts'])
        deltas = []
        for i in range(1, len(window_events_sorted)):
            deltas.append((window_events_sorted[i]['ts'] - window_events_sorted[i-1]['ts']).total_seconds())
        if deltas:
            avg_interval_sec = float(sum(deltas) / len(deltas))

    for k in sorted(agg.keys()):
        c = count[k]
        f = focus_count.get(k, 0)
        focus_minutes = (f * avg_interval_sec / 60.0) if (avg_interval_sec > 0 and f > 0) else 0.0
        result.append({
            'period': k,
            'avg_stress_expected': (agg[k] / c) if c else 0.0,
            'avg_stress_primary': (primary_agg[k] / c) if c else 0.0,
            'samples': c,
            'focus_samples': f,
            'focus_minutes_est': focus_minutes,
            'peak': max_peak.get(k)
        })

    # Overall peaks across window
    peaks = [e for e in window_events if e['score'] >= peak_threshold]
    peaks_sorted = sorted(peaks, key=lambda x: x['score'], reverse=True)[:max(1, min(10, top_n_peaks))]
    peaks_out = [{"ts": e['ts'].isoformat(), "score": e['score'], "emotion": e['emotion']} for e in peaks_sorted]

    return {"period": period, "window_days": days, "focus_threshold": focus_threshold, "peak_threshold": peak_threshold, "summary": result, "top_peaks": peaks_out}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=2000) 