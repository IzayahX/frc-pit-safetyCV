#!/usr/bin/env python3
"""
pi_deploy.py – FRC Pit Safety Monitor (Raspberry Pi)

Parity-focused deployment:
- Auto-detect model input layout (NHWC/NCHW), dtype, quantization
- Dynamically use the model's input resolution (no hardcoded imgsz)
- Letterbox-by-default (no aspect squish) with correct box unprojection
- Robust camera + inference error handling with clear logging
"""

import cv2
import time
import os
import sys
import threading
import logging
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pit_safety")


def _env_int(name: str, default: int, minimum=None):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning(f"Invalid {name}={raw!r}; using {default}")
        value = default
    if minimum is not None and value < minimum:
        logger.warning(f"{name}={value} is below minimum {minimum}; using {minimum}")
        value = minimum
    return value


def _env_float(name: str, default: float, minimum=None, maximum=None):
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning(f"Invalid {name}={raw!r}; using {default}")
        value = default
    if minimum is not None and value < minimum:
        logger.warning(f"{name}={value} is below minimum {minimum}; using {minimum}")
        value = minimum
    if maximum is not None and value > maximum:
        logger.warning(f"{name}={value} is above maximum {maximum}; using {maximum}")
        value = maximum
    return value


# ── Config (pulled from environment, set by launcher.py) ─
CONFIRM_FRAMES = _env_int("CONFIRM_FRAMES", 8, minimum=1)
MIN_CONF = _env_float("MIN_CONF", 0.70, minimum=0.0, maximum=1.0)
CAM_RETRY_DELAY = _env_int("CAM_RETRY_DELAY", 5, minimum=1)
DATA_SAVE_INT = _env_float("DATA_SAVE_INT", 0, minimum=0.0)
PROCESS_EVERY_N = _env_int("PROCESS_EVERY_N", 2, minimum=1)
TFLITE_THREADS = _env_int("TFLITE_THREADS", 2, minimum=1)
MODEL_PATH = os.environ.get("MODEL_PATH", "")
CAMERA_INDEX = os.environ.get("CAMERA_INDEX", "auto")
FULLSCREEN = os.environ.get("FULLSCREEN", "1") == "1"
DEMO_MODE = os.environ.get("DEMO_MODE", "1") == "1"
RESIZE_MODE = os.environ.get("RESIZE_MODE", "letterbox").strip().lower()
DEBUG_PREPROC = os.environ.get("DEBUG_PREPROC", "0") == "1"

WIN_NAME = "FRC Pit Safety - Monitor"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ── Camera backend availability ──────────────────────────
HAS_PICAMERA2 = False
try:
    from picamera2 import Picamera2

    HAS_PICAMERA2 = True
except ImportError:
    pass


# ── TFLite Model Loading ─────────────────────────────────
def load_interpreter(model_path: str, num_threads: int):
    """Try multiple TFLite backends in priority order."""
    Interpreter = None
    load_delegate = None
    backend = None

    try:
        from ai_edge_litert.interpreter import Interpreter, load_delegate

        backend = "ai-edge-litert"
    except ImportError:
        pass

    if Interpreter is None:
        try:
            import importlib

            mod = importlib.import_module("tflite_runtime.interpreter")
            Interpreter = mod.Interpreter
            load_delegate = getattr(mod, "load_delegate", None)
            backend = "tflite-runtime"
        except (ImportError, ModuleNotFoundError):
            pass

    if Interpreter is None:
        try:
            import tensorflow as tf

            Interpreter = tf.lite.Interpreter
            load_delegate = tf.lite.experimental.load_delegate
            backend = "tensorflow"
        except (ImportError, Exception):
            logger.error("No TFLite runtime found. Install ai-edge-litert or tflite-runtime.")
            sys.exit(1)

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
    logger.info(f"TFLite backend: {backend or 'unknown'}")
    return interp


_logged_model_io = {"done": False}


def _input_spec(input_detail: dict):
    shape = list(input_detail.get("shape", []))
    dtype = input_detail.get("dtype")
    q = input_detail.get("quantization", (0.0, 0))
    q_scale = float(q[0]) if q and q[0] is not None else 0.0
    q_zero = int(q[1]) if q and q[1] is not None else 0

    layout = "unknown"
    h = w = c = None
    if len(shape) == 4 and shape[0] in (1, -1):
        if shape[3] in (1, 3, 4) and shape[1] > 0 and shape[2] > 0:
            layout = "NHWC"
            h, w, c = int(shape[1]), int(shape[2]), int(shape[3])
        elif shape[1] in (1, 3, 4) and shape[2] > 0 and shape[3] > 0:
            layout = "NCHW"
            c, h, w = int(shape[1]), int(shape[2]), int(shape[3])

    return {
        "shape": shape,
        "dtype": dtype,
        "layout": layout,
        "h": h,
        "w": w,
        "c": c,
        "q_scale": q_scale,
        "q_zero": q_zero,
    }


def _letterbox(img_rgb: np.ndarray, new_w: int, new_h: int):
    h0, w0 = img_rgb.shape[:2]
    r = min(new_w / w0, new_h / h0)
    new_unpad_w = int(round(w0 * r))
    new_unpad_h = int(round(h0 * r))
    resized = cv2.resize(img_rgb, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
    dw = new_w - new_unpad_w
    dh = new_h - new_unpad_h
    left = dw // 2
    right = dw - left
    top = dh // 2
    bottom = dh - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, r, left, top


def _preprocess(frame_rgb: np.ndarray, spec: dict):
    if spec["layout"] == "unknown" or spec["h"] is None or spec["w"] is None:
        raise ValueError(f"Unsupported model input shape/layout: {spec['shape']} layout={spec['layout']}")

    in_h, in_w = spec["h"], spec["w"]

    if RESIZE_MODE == "stretch":
        img = cv2.resize(frame_rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        meta = {
            "mode": "stretch",
            "r": (in_w / frame_rgb.shape[1], in_h / frame_rgb.shape[0]),
            "padx": 0,
            "pady": 0,
        }
    else:
        img, r, pad_x, pad_y = _letterbox(frame_rgb, in_w, in_h)
        meta = {"mode": "letterbox", "r": r, "padx": pad_x, "pady": pad_y}

    if spec["layout"] == "NCHW":
        img = np.transpose(img, (2, 0, 1))  # CHW

    dtype = spec["dtype"]
    q_scale = spec["q_scale"]
    q_zero = spec["q_zero"]

    if dtype == np.float32:
        tensor = (img.astype(np.float32) / 255.0)[np.newaxis]
        meta["quant_mode"] = "float01"
    elif dtype in (np.uint8, np.int8):
        # scale ~1 => model likely trained in 0-255 domain; else 0-1 domain
        if q_scale and q_scale > 0.1:
            real = img.astype(np.float32)
            meta["quant_mode"] = "u8255"
        else:
            real = img.astype(np.float32) / 255.0
            meta["quant_mode"] = "u801"

        if not q_scale:
            tensor = img.astype(dtype)[np.newaxis]
            meta["quant_mode"] = "raw"
        else:
            q = np.round(real / q_scale + q_zero)
            if dtype == np.uint8:
                q = np.clip(q, 0, 255).astype(np.uint8)
            else:
                q = np.clip(q, -128, 127).astype(np.int8)
            tensor = q[np.newaxis]
    else:
        raise ValueError(f"Unsupported input dtype: {dtype}")

    return tensor, meta


def _unproject_box(x1, y1, x2, y2, meta: dict, orig_w: int, orig_h: int):
    if meta["mode"] == "stretch":
        sx, sy = meta["r"]
        x1 = x1 / sx
        x2 = x2 / sx
        y1 = y1 / sy
        y2 = y2 / sy
    else:
        r = meta["r"]
        x1 = (x1 - meta["padx"]) / r
        x2 = (x2 - meta["padx"]) / r
        y1 = (y1 - meta["pady"]) / r
        y2 = (y2 - meta["pady"]) / r

    x1 = int(max(0, min(orig_w - 1, x1)))
    x2 = int(max(0, min(orig_w - 1, x2)))
    y1 = int(max(0, min(orig_h - 1, y1)))
    y2 = int(max(0, min(orig_h - 1, y2)))
    return x1, y1, x2, y2


def infer(interp, input_detail: dict, output_detail: dict, frame_rgb: np.ndarray, conf: float):
    """Run inference and return bounding boxes in original-frame coords."""
    spec = _input_spec(input_detail)

    if not _logged_model_io["done"]:
        logger.info(
            f"Model input: shape={spec['shape']} layout={spec['layout']} dtype={spec['dtype']} "
            f"q=({spec['q_scale']},{spec['q_zero']})"
        )
        try:
            out_shape = list(output_detail.get("shape", []))
            logger.info(f"Model output: shape={out_shape} dtype={output_detail.get('dtype')}")
        except Exception:
            pass
        logger.info(f"Preprocess: RESIZE_MODE={RESIZE_MODE}")
        _logged_model_io["done"] = True

    try:
        tensor, meta = _preprocess(frame_rgb, spec)
    except Exception as e:
        logger.error(f"Preprocess failed: {e}")
        return []

    try:
        interp.set_tensor(input_detail["index"], tensor)
        interp.invoke()
        preds = interp.get_tensor(output_detail["index"])[0]
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return []

    if hasattr(preds, "shape") and len(preds.shape) == 2 and preds.shape[0] < preds.shape[1]:
        preds = preds.T

    # Validate output: must be 2-D with at least [cx, cy, w, h, conf] columns
    if not isinstance(preds, np.ndarray) or preds.ndim != 2 or preds.shape[1] < 5:
        logger.warning(
            "Unexpected model output shape: %s; expected (N, >=5). Skipping frame.",
            getattr(preds, "shape", type(preds)),
        )
        return []

    confs = preds[:, 4].astype(np.float32)
    if confs.size and confs.max() > 1.0:
        confs /= 255.0

    filtered = preds[confs >= conf]

    orig_h, orig_w = frame_rgb.shape[:2]
    in_h, in_w = spec["h"], spec["w"]

    boxes = []
    for d in filtered:
        vals = d[:4].astype(np.float32)

        if vals.max() <= 1.5:
            vals[0] *= in_w
            vals[1] *= in_h
            vals[2] *= in_w
            vals[3] *= in_h
        elif vals.max() <= 255.0 and max(in_w, in_h) > 255:
            vals = (vals / 255.0)
            vals[0] *= in_w
            vals[1] *= in_h
            vals[2] *= in_w
            vals[3] *= in_h

        cx, cy, bw, bh = vals[0], vals[1], vals[2], vals[3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        x1, y1, x2, y2 = _unproject_box(x1, y1, x2, y2, meta, orig_w, orig_h)
        boxes.append((x1, y1, x2, y2))

    if DEBUG_PREPROC:
        try:
            os.makedirs("debug_preproc", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(
                f"debug_preproc/frame_{ts}.jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            )
        except Exception:
            pass

    return boxes


# ── Per-Person Violation Tracker ─────────────────────────
class TrackedPerson:
    """Tracks consecutive frames without glasses to confirm a violation."""

    def __init__(self, box):
        self.streak = 0
        self.is_safe = True
        self.last_box = box  # (x1, y1, x2, y2) in frame coords

    def update(self, has_glasses, box):
        self.last_box = box
        if has_glasses:
            self.streak = max(0, self.streak - 1)
            if self.streak == 0:
                self.is_safe = True
        else:
            self.streak += 1
            if self.streak >= CONFIRM_FRAMES:
                self.is_safe = False


# ── IoU / Track Matching Helpers ─────────────────────────
def _iou(a, b):
    """Intersection-over-Union for two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_tracks(tracks, faces, iou_thresh=0.30):
    """
    Spatially match current face detections to existing track IDs by IoU.

    Args:
        tracks:     dict {fid: TrackedPerson}  (each has .last_box)
        faces:      sequence of (x, y, w, h) from detectMultiScale
        iou_thresh: minimum IoU score to accept a match

    Returns:
        matched:    dict {face_index: fid}   — confirmed spatial matches
        new_idxs:   list of face indices with no matching existing track
        stale_fids: set of track IDs not matched to any current detection
    """
    if not tracks or not len(faces):
        return {}, list(range(len(faces))), set(tracks.keys())

    candidates = []
    for fi, (fx, fy, fw, fh) in enumerate(faces):
        face_box = (fx, fy, fx + fw, fy + fh)
        for fid, person in tracks.items():
            score = _iou(face_box, person.last_box)
            if score >= iou_thresh:
                candidates.append((score, fi, fid))

    # Greedy assignment: highest IoU pairs first
    candidates.sort(key=lambda x: -x[0])
    matched = {}
    used_fids = set()
    for score, fi, fid in candidates:
        if fi not in matched and fid not in used_fids:
            matched[fi] = fid
            used_fids.add(fid)

    new_idxs = [fi for fi in range(len(faces)) if fi not in matched]
    stale_fids = set(tracks.keys()) - used_fids
    return matched, new_idxs, stale_fids


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
                logger.warning(f"picamera2 init failed: {e}, falling back to OpenCV")
                self.picam = None

        self._init_opencv()

    def _init_opencv(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            self.cap = cv2.VideoCapture(self.idx)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.use_picamera2 = False
            logger.info(f"Camera: OpenCV VideoCapture (index {self.idx})")
        except Exception as e:
            logger.error(f"Failed to init OpenCV camera (index {self.idx}): {e}")
            self.cap = None

    def _loop(self):
        while self.alive:
            if self.use_picamera2 and self.picam:
                try:
                    frame = self.picam.capture_array()
                    with self.lock:
                        self.ok = True
                        self.frame = frame
                except Exception as e:
                    logger.warning(f"picamera2 capture failed: {e}")
                    time.sleep(CAM_RETRY_DELAY)
                    self._init_cam()
            else:
                if not self.cap:
                    time.sleep(CAM_RETRY_DELAY)
                    self._init_opencv()
                    continue
                if not self.cap.isOpened():
                    time.sleep(CAM_RETRY_DELAY)
                    self._init_opencv()
                    continue
                try:
                    ok, f = self.cap.read()
                except Exception as e:
                    logger.warning(f"OpenCV read failed: {e}")
                    ok, f = False, None
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
COLOR_SAFE = (80, 220, 0)
COLOR_UNSAFE = (255, 60, 0)
COLOR_HUD = (15, 15, 15)
WHITE = (255, 255, 255)


def make_no_camera_slate(width=640, height=480):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        img,
        "NO CAMERA / RECONNECTING",
        (40, height // 2 - 20),
        cv2.FONT_HERSHEY_DUPLEX,
        0.9,
        (60, 60, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        time.strftime("%H:%M:%S"),
        (40, height // 2 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
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

    cv2.rectangle(frame, (10, 10), (220, 95), COLOR_HUD, -1)
    status_txt = "PIT SECURE" if not overall_unsafe else "UNSAFE PIT"
    status_col = COLOR_SAFE if not overall_unsafe else COLOR_UNSAFE
    cv2.putText(
        frame, status_txt, (22, 36), cv2.FONT_HERSHEY_DUPLEX, 0.6, status_col, 1, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        f"People: {people_count}",
        (22, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Violations: {violations}",
        (22, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    info = f"{int(fps)} FPS | {tag}"
    (iw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.putText(
        frame,
        info,
        (w - iw - 10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (110, 110, 110),
        1,
        cv2.LINE_AA,
    )

    if data_collecting:
        cv2.circle(frame, (w - 20, 20), 8, (0, 0, 255), -1)
        cv2.putText(
            frame, "REC", (w - 55, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA
        )


def main():
    if not os.environ.get("DISPLAY"):
        os.environ["DISPLAY"] = ":0"

    model_path = MODEL_PATH
    if not model_path:
        try:
            tflite_files = [f for f in os.listdir(".") if f.endswith(".tflite")]
        except Exception as e:
            logger.error(f"Failed to list model directory: {e}")
            sys.exit(1)
        if not tflite_files:
            logger.error("No .tflite model found in current directory")
            sys.exit(1)
        model_path = tflite_files[0]

    if CAMERA_INDEX in ("auto", ""):
        camera_idx = find_working_camera()
        if camera_idx is None:
            camera_idx = 0
            logger.warning("No camera found at startup; will keep retrying")
    else:
        try:
            camera_idx = int(CAMERA_INDEX)
        except (TypeError, ValueError):
            logger.warning(f"Invalid CAMERA_INDEX={CAMERA_INDEX!r}; auto-detecting")
            camera_idx = find_working_camera()
            if camera_idx is None:
                camera_idx = 0

    logger.info(f"Model: {model_path}")
    logger.info(
        f"Config: CONFIRM_FRAMES={CONFIRM_FRAMES}, MIN_CONF={MIN_CONF}, PROCESS_EVERY_N={PROCESS_EVERY_N}, RESIZE_MODE={RESIZE_MODE}"
    )
    logger.info(f"Camera: index {camera_idx} (-1 = Pi Camera)")
    logger.info(f"FULLSCREEN={FULLSCREEN} DEMO_MODE={DEMO_MODE}")

    if DATA_SAVE_INT > 0:
        logger.info(f"Training data collection: every {DATA_SAVE_INT}s -> data_collection/")

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        logger.error(f"Failed to load face cascade: {FACE_CASCADE_PATH}")
        sys.exit(1)

    interp = load_interpreter(model_path, TFLITE_THREADS)
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    model_tag = f"Pi | {str(in_det.get('dtype'))}"

    use_picam = camera_idx == -1
    cam = CamBuffer(idx=max(0, camera_idx), use_picamera=use_picam if use_picam else None)

    tracks = {}
    next_id = 0
    violations = 0
    overall_unsafe = False
    prev_t = time.perf_counter()
    last_data_t = prev_t
    frame_count = 0
    cached_faces = []
    cached_g_boxes = []
    cached_matched = {}  # face_index -> fid; kept in sync with cached_faces

    if DEMO_MODE:
        logger.info("Main loop started (DEMO_MODE: quit key disabled)")
    else:
        logger.info("Main loop started — press 'q' to quit")

    try:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    except Exception as e:
        logger.error(f"Failed to create display window: {e}")
        cam.release()
        sys.exit(1)
    fs_state = {"done": False}

    try:
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

            if cam.is_rgb():
                frame_rgb = frame
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            now = time.perf_counter()
            if DATA_SAVE_INT > 0 and (now - last_data_t) >= DATA_SAVE_INT:
                last_data_t = now
                try:
                    os.makedirs("data_collection", exist_ok=True)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    if cv2.imwrite(f"data_collection/raw_{ts}.jpg", frame_bgr):
                        logger.info(f"Saved training image: raw_{ts}.jpg")
                    else:
                        logger.warning("Failed to write training image (disk full or permissions issue?)")
                except OSError as e:
                    logger.warning(f"Could not save training image: {e}")

            if frame_count % PROCESS_EVERY_N == 0:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                new_faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
                new_g_boxes = infer(interp, in_det, out_det, frame_rgb, MIN_CONF)

                # --- Identity tracking: match detections to existing tracks by IoU ---
                matched, new_idxs, stale_fids = match_tracks(tracks, new_faces)

                # Drop tracks whose face has disappeared
                for fid in stale_fids:
                    del tracks[fid]

                # Allocate new stable IDs for unmatched detections
                for fi in new_idxs:
                    next_id += 1
                    fid = f"P{next_id}"
                    matched[fi] = fid
                    fx, fy, fw, fh = new_faces[fi]
                    tracks[fid] = TrackedPerson((fx, fy, fx + fw, fy + fh))

                # Update each track's state (streak + last_box) exactly once per inference
                for fi, (fx, fy, fw, fh) in enumerate(new_faces):
                    fid = matched[fi]
                    face_box = (fx, fy, fx + fw, fy + fh)
                    has_glasses = False
                    for (gx1, gy1, gx2, gy2) in new_g_boxes:
                        gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                        if fx < gcx < fx + fw and fy < gcy < fy + fh:
                            has_glasses = True
                            break
                    tracks[fid].update(has_glasses, face_box)

                cached_faces = new_faces
                cached_g_boxes = new_g_boxes
                cached_matched = matched

            faces = cached_faces
            g_boxes = cached_g_boxes

            current_frame_unsafe = False
            for fi, (fx, fy, fw, fh) in enumerate(faces):
                fid = cached_matched.get(fi)
                # Guard against cached_matched being stale (e.g. first few frames)
                if fid is None or fid not in tracks:
                    continue
                col = COLOR_SAFE if tracks[fid].is_safe else COLOR_UNSAFE
                if not tracks[fid].is_safe:
                    current_frame_unsafe = True
                    msg = "GLASSES REQUIRED"
                    (mw, _mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
                    cv2.rectangle(frame_bgr, (fx, fy - 36), (fx + mw + 10, fy - 22), col, -1)
                    cv2.putText(
                        frame_bgr,
                        msg,
                        (fx + 5, fy - 24),
                        cv2.FONT_HERSHEY_DUPLEX,
                        0.4,
                        WHITE,
                        1,
                        cv2.LINE_AA,
                    )

                cv2.rectangle(frame_bgr, (fx, fy), (fx + fw, fy + fh), col, 2)
                cv2.rectangle(frame_bgr, (fx, fy - 22), (fx + 45, fy), col, -1)
                cv2.putText(
                    frame_bgr, fid, (fx + 5, fy - 7), cv2.FONT_HERSHEY_DUPLEX, 0.45, WHITE, 1, cv2.LINE_AA
                )

            if current_frame_unsafe and not overall_unsafe:
                violations += 1
                logger.warning(f"Violation #{violations} detected")
                try:
                    os.makedirs("violations", exist_ok=True)
                    if not cv2.imwrite(
                        f"violations/pit_danger_{time.strftime('%H%M%S')}.jpg", frame_bgr
                    ):
                        logger.warning("Failed to write violation snapshot (disk full or permissions issue?)")
                except OSError as e:
                    logger.warning(f"Could not save violation snapshot: {e}")

            overall_unsafe = current_frame_unsafe

            now = time.perf_counter()
            fps = 1.0 / (now - prev_t + 1e-9)
            prev_t = now

            draw_hud(frame_bgr, overall_unsafe, fps, violations, len(faces), model_tag, DATA_SAVE_INT > 0)
            cv2.imshow(WIN_NAME, frame_bgr)
            apply_fullscreen_once(WIN_NAME, fs_state)

            key = cv2.waitKey(1) & 0xFF
            if not DEMO_MODE and key == ord("q"):
                break
    except Exception as e:
        logger.exception(f"Fatal runtime error in main loop: {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
