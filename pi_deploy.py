#!/usr/bin/env python3
"""
pi_deploy.py – FRC Pit Safety Monitor (Raspberry Pi)

Runs the safety glasses detection loop optimized for Pi 4/5.
Supports picamera2 (ribbon) and USB webcams with auto-reconnect.
"""

import cv2
import time
import os
import sys
import threading
import logging
import numpy as np

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pit_safety")

# ── Config (pulled from environment, set by launcher.py) ─
CONFIRM_FRAMES  = int(os.environ.get("CONFIRM_FRAMES", 8))
MIN_CONF        = float(os.environ.get("MIN_CONF", 0.70))
CAM_RETRY_DELAY = int(os.environ.get("CAM_RETRY_DELAY", 5))
DATA_SAVE_INT   = float(os.environ.get("DATA_SAVE_INT", 0))
PROCESS_EVERY_N = int(os.environ.get("PROCESS_EVERY_N", 2))
TFLITE_THREADS  = int(os.environ.get("TFLITE_THREADS", 2))
MODEL_PATH      = os.environ.get("MODEL_PATH", "")
CAMERA_INDEX    = os.environ.get("CAMERA_INDEX", "auto")
FULLSCREEN      = os.environ.get("FULLSCREEN", "1") == "1"
DEMO_MODE       = os.environ.get("DEMO_MODE", "1") == "1"

WIN_NAME = "FRC Pit Safety - Monitor"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Check for picamera2 (only available on Pi OS)
HAS_PICAMERA2 = False
try:
    from picamera2 import Picamera2
    HAS_PICAMERA2 = True
except ImportError:
    pass


# ── TFLite Model Loading ────────────────────────────────
def load_interpreter(model_path: str, num_threads: int):
    """Try multiple TFLite backends in priority order."""
    Interpreter = None
    load_delegate = None

    # 1) ai-edge-litert (Google's newer package)
    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate
    except ImportError:
        pass

    # 2) tflite-runtime (classic pip package)
    if Interpreter is None:
        try:
            import importlib
            mod = importlib.import_module("tflite_runtime.interpreter")
            Interpreter = mod.Interpreter
            load_delegate = getattr(mod, "load_delegate", None)
        except (ImportError, ModuleNotFoundError):
            pass

    # 3) Full TensorFlow (heavy, but works)
    if Interpreter is None:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
            load_delegate = tf.lite.experimental.load_delegate
        except (ImportError, Exception):
            logger.error("No TFLite runtime found. Install ai-edge-litert or tflite-runtime.")
            sys.exit(1)

    # Try XNNPACK for acceleration on Pi
    delegates = []
    if load_delegate:
        try:
            delegates = [load_delegate("xnnpack", {"num_threads": num_threads})]
            logger.info(f"XNNPACK delegate loaded ({num_threads} threads)")
        except Exception:
            logger.warning("XNNPACK delegate not available, using default")

    interp = Interpreter(
        model_path=model_path,
        num_threads=num_threads,
        experimental_delegates=delegates or None,
    )
    interp.allocate_tensors()
    return interp


def infer(interp, input_idx, output_idx, is_int8, imgsz, frame_rgb, conf):
    """Run inference and return bounding boxes [(x1,y1,x2,y2), ...]."""
    img = cv2.resize(frame_rgb, (imgsz, imgsz))
    if is_int8:
        tensor = img.astype(np.uint8)[np.newaxis]
    else:
        tensor = (img.astype(np.float32) / 255.0)[np.newaxis]

    interp.set_tensor(input_idx, tensor)
    interp.invoke()

    preds = interp.get_tensor(output_idx)[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    confs = preds[:, 4].astype(np.float32)
    if confs.max() > 1.0:
        confs /= 255.0

    filtered = preds[confs >= conf]
    h, w = frame_rgb.shape[:2]
    sx, sy = w / imgsz, h / imgsz

    boxes = []
    for d in filtered:
        vals = d[:4].astype(np.float32)
        # INT8 models output pixel coords as 0-255, need to rescale
        if vals.max() > 1.0:
            vals /= 255.0
            vals *= imgsz
        cx, cy, bw, bh = vals[0] * sx, vals[1] * sy, vals[2] * sx, vals[3] * sy
        boxes.append((int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)))
    return boxes


# ── Per-Person Violation Tracker ─────────────────────────
class TrackedPerson:
    """Tracks consecutive frames without glasses to confirm a violation."""
    def __init__(self):
        self.streak = 0
        self.is_safe = True

    def update(self, has_glasses):
        if has_glasses:
            self.streak = max(0, self.streak - 1)
            if self.streak == 0:
                self.is_safe = True
        else:
            self.streak += 1
            if self.streak >= CONFIRM_FRAMES:
                self.is_safe = False


# ── Camera Auto-Detection ────────────────────────────────
def find_working_camera():
    """Probe for the first working camera (Pi cam or USB)."""
    if HAS_PICAMERA2:
        try:
            picam = Picamera2()
            picam.close()
            logger.info("Found Pi Camera (libcamera)")
            return -1
        except Exception:
            pass

    for i in range(5):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret and frame is not None:
                    logger.info(f"Found USB camera at index {i}")
                    return i
        except Exception:
            pass

    logger.warning("No working camera found")
    return None


# ── Threaded Camera Buffer (auto-reconnect) ──────────────
class CamBuffer:
    """Continuously grabs frames in a background thread so the main
    loop never blocks on a slow camera read."""

    def __init__(self, idx=0, use_picamera=None):
        self.idx = idx
        self.force_picamera = use_picamera
        self.frame = None
        self.ok = False
        self.alive = True
        self.lock = threading.Lock()
        self.use_picamera2 = False
        self.picam = None
        self.cap = None
        self._init_cam()
        threading.Thread(target=self._loop, daemon=True).start()

    def _init_cam(self):
        use_picam = (self.force_picamera is True) or (self.idx == -1)

        if use_picam and HAS_PICAMERA2:
            try:
                self.picam = Picamera2()
                config = self.picam.create_video_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.picam.configure(config)
                self.picam.start()
                self.use_picamera2 = True
                logger.info("Camera: picamera2 (hardware accelerated)")
                return
            except Exception as e:
                logger.warning(f"picamera2 failed: {e}, falling back to OpenCV")
                self.picam = None

        self._init_opencv()

    def _init_opencv(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.idx)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.use_picamera2 = False
        logger.info(f"Camera: OpenCV VideoCapture (index {self.idx})")

    def _loop(self):
        while self.alive:
            if self.use_picamera2 and self.picam:
                try:
                    frame = self.picam.capture_array()
                    with self.lock:
                        self.ok = True
                        self.frame = frame
                except Exception:
                    time.sleep(CAM_RETRY_DELAY)
                    self._init_cam()
            else:
                if not self.cap or not self.cap.isOpened():
                    time.sleep(CAM_RETRY_DELAY)
                    self._init_opencv()
                    continue
                ok, f = self.cap.read()
                with self.lock:
                    self.ok, self.frame = ok, f
                if not ok:
                    time.sleep(1)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ok, self.frame.copy()

    def is_rgb(self):
        return self.use_picamera2

    def release(self):
        self.alive = False
        if self.picam:
            try:
                self.picam.stop()
            except Exception:
                pass
        if self.cap:
            self.cap.release()


# ── HUD Drawing ─────────────────────────────────────────
COLOR_SAFE   = (80, 220, 0)
COLOR_UNSAFE = (255, 60, 0)
COLOR_HUD    = (15, 15, 15)
WHITE        = (255, 255, 255)


def make_no_camera_slate(width=640, height=480):
    """Placeholder frame shown while waiting for a camera."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(img, "NO CAMERA / RECONNECTING", (40, height // 2 - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (60, 60, 255), 2, cv2.LINE_AA)
    cv2.putText(img, time.strftime("%H:%M:%S"), (40, height // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def apply_fullscreen_once(win_name, state):
    if not FULLSCREEN or state["done"]:
        return
    try:
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception:
        pass
    state["done"] = True


def draw_hud(frame, overall_unsafe, fps, violations, people_count, tag, data_collecting):
    h, w = frame.shape[:2]

    # Status box in top-left
    cv2.rectangle(frame, (10, 10), (220, 95), COLOR_HUD, -1)
    status_txt = "PIT SECURE" if not overall_unsafe else "UNSAFE PIT"
    status_col = COLOR_SAFE if not overall_unsafe else COLOR_UNSAFE
    cv2.putText(frame, status_txt, (22, 36), cv2.FONT_HERSHEY_DUPLEX, 0.6, status_col, 1, cv2.LINE_AA)
    cv2.putText(frame, f"People: {people_count}", (22, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Violations: {violations}", (22, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # FPS + engine tag in bottom-right
    info = f"{int(fps)} FPS | {tag}"
    (iw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.putText(frame, info, (w - iw - 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1, cv2.LINE_AA)

    # Red dot when saving training data
    if data_collecting:
        cv2.circle(frame, (w - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 55, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)


# ── Main Loop ────────────────────────────────────────────
def main():
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    # Auto-detect model if not specified
    model_path = MODEL_PATH
    if not model_path:
        tflite_files = [f for f in os.listdir(".") if f.endswith(".tflite")]
        if not tflite_files:
            logger.error("No .tflite model found in current directory")
            sys.exit(1)
        model_path = tflite_files[0]

    # Pick camera
    if CAMERA_INDEX in ("auto", ""):
        camera_idx = find_working_camera()
        if camera_idx is None:
            camera_idx = 0
            logger.warning("No camera found at startup; will keep retrying")
    else:
        camera_idx = int(CAMERA_INDEX)

    logger.info(f"Model: {model_path}")
    logger.info(f"Config: CONFIRM_FRAMES={CONFIRM_FRAMES}, MIN_CONF={MIN_CONF}, PROCESS_EVERY_N={PROCESS_EVERY_N}")
    logger.info(f"Camera: index {camera_idx} (-1 = Pi Camera)")
    logger.info(f"FULLSCREEN={FULLSCREEN} DEMO_MODE={DEMO_MODE}")

    if DATA_SAVE_INT > 0:
        logger.info(f"Training data collection: every {DATA_SAVE_INT}s -> data_collection/")

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    interp = load_interpreter(model_path, TFLITE_THREADS)

    in_det = interp.get_input_details()[0]
    input_idx = in_det["index"]
    output_idx = interp.get_output_details()[0]["index"]
    is_int8 = in_det["dtype"] == np.uint8
    imgsz = in_det["shape"][1]
    model_tag = f"Pi | {'INT8' if is_int8 else 'FP32'}"

    use_picam = camera_idx == -1
    cam = CamBuffer(idx=max(0, camera_idx), use_picamera=use_picam if use_picam else None)

    tracks = {}
    violations = 0
    overall_unsafe = False
    prev_t = time.perf_counter()
    last_data_t = prev_t
    frame_count = 0
    cached_faces = []
    cached_g_boxes = []

    if DEMO_MODE:
        logger.info("Main loop started (DEMO_MODE: quit key disabled)")
    else:
        logger.info("Main loop started — press 'q' to quit")

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    fs_state = {"done": False}

    while True:
        ok, frame = cam.read()
        if not ok or frame is None:
            cv2.imshow(WIN_NAME, make_no_camera_slate())
            apply_fullscreen_once(WIN_NAME, fs_state)
            key = cv2.waitKey(30) & 0xFF
            if not DEMO_MODE and key == ord("q"):
                break
            continue

        frame_count += 1

        # picamera2 gives RGB, OpenCV gives BGR
        if cam.is_rgb():
            frame_rgb = frame
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        now = time.perf_counter()

        # Save raw images for training data
        if DATA_SAVE_INT > 0 and (now - last_data_t) >= DATA_SAVE_INT:
            os.makedirs("data_collection", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"data_collection/raw_{ts}.jpg", frame_bgr)
            logger.info(f"Saved training image: raw_{ts}.jpg")
            last_data_t = now

        # Only run detection every N frames to save CPU
        if frame_count % PROCESS_EVERY_N == 0:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            cached_faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            cached_g_boxes = infer(interp, input_idx, output_idx, is_int8, imgsz, frame_rgb, MIN_CONF)

        faces = cached_faces
        g_boxes = cached_g_boxes

        # Match glasses detections to faces
        current_frame_unsafe = False
        active_ids = set()

        for i, (fx, fy, fw, fh) in enumerate(faces):
            fid = f"P_{i+1}"
            active_ids.add(fid)
            if fid not in tracks:
                tracks[fid] = TrackedPerson()

            # Check if any glasses bounding box center falls inside this face
            has_glasses = False
            for (gx1, gy1, gx2, gy2) in g_boxes:
                gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                if fx < gcx < fx + fw and fy < gcy < fy + fh:
                    has_glasses = True
                    break

            tracks[fid].update(has_glasses)
            col = COLOR_SAFE if tracks[fid].is_safe else COLOR_UNSAFE
            if not tracks[fid].is_safe:
                current_frame_unsafe = True
                msg = "GLASSES REQUIRED"
                (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
                cv2.rectangle(frame_bgr, (fx, fy - 36), (fx + mw + 10, fy - 22), col, -1)
                cv2.putText(frame_bgr, msg, (fx + 5, fy - 24), cv2.FONT_HERSHEY_DUPLEX, 0.4, WHITE, 1, cv2.LINE_AA)

            cv2.rectangle(frame_bgr, (fx, fy), (fx + fw, fy + fh), col, 2)
            cv2.rectangle(frame_bgr, (fx, fy - 22), (fx + 45, fy), col, -1)
            cv2.putText(frame_bgr, fid, (fx + 5, fy - 7), cv2.FONT_HERSHEY_DUPLEX, 0.45, WHITE, 1, cv2.LINE_AA)

        # New violation = first unsafe frame after a safe period
        if current_frame_unsafe and not overall_unsafe:
            violations += 1
            os.makedirs("violations", exist_ok=True)
            cv2.imwrite(f"violations/pit_danger_{time.strftime('%H%M%S')}.jpg", frame_bgr)
            logger.warning(f"Violation #{violations} detected")

        overall_unsafe = current_frame_unsafe
        tracks = {tid: t for tid, t in tracks.items() if tid in active_ids}

        now = time.perf_counter()
        fps = 1.0 / (now - prev_t + 1e-9)
        prev_t = now

        draw_hud(frame_bgr, overall_unsafe, fps, violations, len(faces), model_tag, DATA_SAVE_INT > 0)
        cv2.imshow(WIN_NAME, frame_bgr)
        apply_fullscreen_once(WIN_NAME, fs_state)

        key = cv2.waitKey(1) & 0xFF
        if not DEMO_MODE and key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
