"""
webcam_test.py – FRC Pit Safety Monitor (Laptop Simulator)

Lets you test detection logic on your laptop before deploying to the Pi.
Supports both ONNX (fast, GPU-capable) and TFLite INT8 (matches Pi behavior).
"""

import cv2
import time
import os
import sys
import threading
import multiprocessing
import numpy as np

# ── Settings ─────────────────────────────────────────────
CONFIRM_FRAMES = 10
MIN_CONF = 0.65
DATA_SAVE_INT = 15.0  # seconds between training image saves (0 to disable)
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


# ── Engine Loaders ───────────────────────────────────────
def load_onnx(model_path):
    import onnxruntime as rt
    opts = rt.SessionOptions()
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = multiprocessing.cpu_count()
    available = rt.get_available_providers()
    providers = [p for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"] if p in available]
    if providers:
        session = rt.InferenceSession(model_path, sess_options=opts, providers=providers)
    else:
        session = rt.InferenceSession(model_path, sess_options=opts)
    in_n = session.get_inputs()[0].name
    out_n = session.get_outputs()[0].name
    imgsz = session.get_inputs()[0].shape[2] if isinstance(session.get_inputs()[0].shape[2], int) else 416
    provider_name = providers[0] if providers else "DefaultProvider"
    return session, in_n, out_n, imgsz, f"ONNX ({provider_name})"


def load_tflite(model_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            print("[ERR] TFLite runtime not found. Run: pip install ai-edge-litert")
            sys.exit(1)

    num_threads = min(4, multiprocessing.cpu_count())
    interp = Interpreter(model_path=model_path, num_threads=num_threads)
    interp.allocate_tensors()

    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    imgsz = in_det["shape"][1]
    is_int8 = in_det["dtype"] == np.uint8
    return interp, in_det["index"], out_det["index"], imgsz, f"TFLite {'INT8' if is_int8 else 'FP32'}"


# ── Inference Wrappers ───────────────────────────────────
def infer_onnx(session, in_n, out_n, frame_rgb, imgsz, conf):
    img = cv2.resize(frame_rgb, (imgsz, imgsz)).astype(np.float32) / 255.0
    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))[np.newaxis]
    preds = session.run([out_n], {in_n: img})[0][0].T
    filtered = preds[preds[:, 4] >= conf]
    h, w = frame_rgb.shape[:2]
    sx, sy = w / imgsz, h / imgsz
    return [(int((d[0]-d[2]/2)*sx), int((d[1]-d[3]/2)*sy),
             int((d[0]+d[2]/2)*sx), int((d[1]+d[3]/2)*sy)) for d in filtered]


def infer_tflite(interp, in_idx, out_idx, frame_rgb, imgsz, conf):
    in_det = interp.get_input_details()[0]
    is_int8 = in_det["dtype"] == np.uint8
    img = cv2.resize(frame_rgb, (imgsz, imgsz))
    tensor = img.astype(np.uint8 if is_int8 else np.float32)[np.newaxis]
    if not is_int8:
        tensor /= 255.0
    interp.set_tensor(in_idx, tensor)
    interp.invoke()
    preds = interp.get_tensor(out_idx)[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T
    confs = preds[:, 4]
    if is_int8:
        confs = confs.astype(np.float32) / 255.0
    filtered = preds[confs >= conf]
    h, w = frame_rgb.shape[:2]
    sx, sy = w / imgsz, h / imgsz
    return [(int((d[0]-d[2]/2)*sx), int((d[1]-d[3]/2)*sy),
             int((d[0]+d[2]/2)*sx), int((d[1]+d[3]/2)*sy)) for d in filtered]


# ── Per-Person Tracker ───────────────────────────────────
class TrackedPerson:
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


# ── Threaded Camera Buffer ───────────────────────────────
class CamBuffer:
    """Reads frames in a background thread so the main loop doesn't stall."""
    def __init__(self, idx):
        self.idx = idx
        self.cap = None
        self.frame = None
        self.ok = False
        self.alive = True
        self.lock = threading.Lock()
        self._init_cam()
        threading.Thread(target=self._loop, daemon=True).start()

    def _init_cam(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _loop(self):
        while self.alive:
            if not self.cap or not self.cap.isOpened():
                time.sleep(2)
                self._init_cam()
                continue
            try:
                ok, f = self.cap.read()
            except Exception as e:
                print(f"[WARN] Camera read failed: {e}", file=sys.stderr)
                ok, f = False, None
            with self.lock:
                self.ok, self.frame = ok, f
            if not ok:
                time.sleep(0.5)

    def read(self):
        with self.lock:
            return self.ok, (self.frame.copy() if self.frame is not None else None)

    def release(self):
        self.alive = False
        if self.cap:
            self.cap.release()


# ── HUD ──────────────────────────────────────────────────
COLOR_SAFE = (80, 220, 0)
COLOR_UNSAFE = (255, 60, 0)
COLOR_HUD = (15, 15, 15)
WHITE = (255, 255, 255)


def draw_hud(frame, overall_unsafe, fps, violations, people_count, tag):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (10, 10), (220, 95), COLOR_HUD, -1)
    stat_txt = "PIT SECURE" if not overall_unsafe else "UNSAFE PIT"
    stat_col = COLOR_SAFE if not overall_unsafe else COLOR_UNSAFE
    cv2.putText(frame, stat_txt, (22, 36), 1, 1.2, stat_col, 1)
    cv2.putText(frame, f"People: {people_count}", (22, 58), 1, 0.9, (180, 180, 180), 1)
    cv2.putText(frame, f"Violations: {violations}", (22, 80), 1, 0.9, (180, 180, 180), 1)
    info = f"{int(fps)} FPS | {tag}"
    (iw, _), _ = cv2.getTextSize(info, 1, 0.8, 1)
    cv2.putText(frame, info, (w - iw - 10, h - 10), 1, 0.8, (110, 110, 110), 1)


# ── Main Loop ────────────────────────────────────────────
def main():
    print("\n[Simulator] Select Inference Engine:")
    print("1. ONNX (best.onnx) — optimized for laptop")
    print("2. TFLite (best_int8.tflite) — simulates Pi behavior")
    try:
        choice = input("Choice [1/2] (default 1): ").strip()
    except EOFError:
        choice = ""

    try:
        if choice == "2":
            engine, in_idx, out_idx, imgsz, tag = load_tflite("best_int8.tflite")
            engine_type = "TFLITE"
        else:
            engine, in_idx, out_idx, imgsz, tag = load_onnx("best.onnx")
            engine_type = "ONNX"
    except Exception as e:
        print(f"[ERR] Failed to load inference engine: {e}", file=sys.stderr)
        return

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print(f"[ERR] Failed to load face cascade: {FACE_CASCADE_PATH}", file=sys.stderr)
        return
    cam = CamBuffer(0)
    tracks = {}
    next_id = 0
    violations = 0
    overall_unsafe = False
    prev_t = time.perf_counter()
    last_data_t = prev_t

    try:
        while True:
            ok, frame = cam.read()
            if not ok or frame is None:
                time.sleep(0.005)  # avoid CPU spin while camera isn't ready
                continue

            # Periodically save raw frames for training data
            now = time.perf_counter()
            if DATA_SAVE_INT > 0 and (now - last_data_t) >= DATA_SAVE_INT:
                last_data_t = now
                try:
                    os.makedirs("data_collection", exist_ok=True)
                    if not cv2.imwrite(
                        f"data_collection/sim_raw_{time.strftime('%H%M%S')}.jpg", frame
                    ):
                        print("[WARN] Failed to write training image (disk full or permissions issue?)", file=sys.stderr)
                except OSError as e:
                    print(f"[WARN] Could not save training image: {e}", file=sys.stderr)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if engine_type == "ONNX":
                g_boxes = infer_onnx(engine, in_idx, out_idx, frame_rgb, imgsz, MIN_CONF)
            else:
                g_boxes = infer_tflite(engine, in_idx, out_idx, frame_rgb, imgsz, MIN_CONF)

            cur_unsafe = False

            # --- Identity tracking: match detections to existing tracks by IoU ---
            matched, new_idxs, stale_fids = match_tracks(tracks, faces)

            # Drop tracks whose face has disappeared
            for fid in stale_fids:
                del tracks[fid]

            # Allocate new stable IDs for unmatched detections
            for fi in new_idxs:
                next_id += 1
                fid = f"P{next_id}"
                matched[fi] = fid
                fx, fy, fw, fh = faces[fi]
                tracks[fid] = TrackedPerson((fx, fy, fx + fw, fy + fh))

            for fi, (fx, fy, fw, fh) in enumerate(faces):
                fid = matched[fi]
                face_box = (fx, fy, fx + fw, fy + fh)
                has_glasses = False
                for (gx1, gy1, gx2, gy2) in g_boxes:
                    gcx, gcy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
                    if fx < gcx < fx + fw and fy < gcy < fy + fh:
                        has_glasses = True
                        break
                tracks[fid].update(has_glasses, face_box)
                col = COLOR_SAFE if tracks[fid].is_safe else COLOR_UNSAFE
                if not tracks[fid].is_safe:
                    cur_unsafe = True
                    msg = "GLASSES REQUIRED"
                    (mw, mh), _ = cv2.getTextSize(msg, 1, 0.9, 1)
                    cv2.rectangle(frame, (fx, fy - 35), (fx + mw + 10, fy - 15), col, -1)
                    cv2.putText(frame, msg, (fx + 5, fy - 17), 1, 0.9, WHITE, 1)
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), col, 2)
                cv2.rectangle(frame, (fx, fy - 22), (fx + 45, fy), col, -1)
                cv2.putText(frame, fid, (fx + 5, fy - 7), 1, 1, WHITE, 1)

            if cur_unsafe and not overall_unsafe:
                violations += 1
            overall_unsafe = cur_unsafe
            fps = 1.0 / (time.perf_counter() - prev_t + 1e-9)
            prev_t = time.perf_counter()
            draw_hud(frame, overall_unsafe, fps, violations, len(faces), tag)
            cv2.imshow("FRC Pit SIMULATOR", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as e:
        print(f"[ERR] Runtime failure: {e}", file=sys.stderr)
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
