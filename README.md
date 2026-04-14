# FRC Pit Safety Monitor

Real-time safety glasses detection for FIRST Robotics Competition pit areas. Uses a custom YOLOv8 model to detect whether people in the pit are wearing safety glasses and alerts when violations are found.

Built for Raspberry Pi 4/5 with a camera module or USB webcam. Also runs on a laptop for testing.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## How It Works

1. **Face detection** — OpenCV Haar cascades find faces in each frame
2. **Glasses detection** — A YOLOv8 model (TFLite INT8) checks for safety glasses
3. **Matching** — If a glasses detection overlaps a face, that person is marked safe
4. **Confirmation** — A violation only triggers after several consecutive frames without glasses (reduces false alarms)

## Project Structure

```
├── launcher.py          # Tkinter GUI — camera detection, settings, auto-start
├── pi_deploy.py         # Main detection loop (optimized for Pi)
├── webcam_test.py       # Laptop simulator (ONNX or TFLite)
├── config.env           # Runtime settings (editable)
├── install.sh           # One-command Pi setup script
├── pit_safety.service   # Optional systemd service for auto-start
├── best_int8.tflite     # YOLOv8 model (INT8 quantized for Pi)
├── best.onnx            # YOLOv8 model (ONNX for laptop testing)
└── data_collection/     # Raw training images (auto-generated)
```

## Quick Start

### Raspberry Pi

Copy the project to a USB drive, plug it into the Pi, then:

```bash
cd /media/pi/YOUR_USB/
chmod +x install.sh
./install.sh
```

The installer handles everything — system packages, Python venv, auto-start on boot.

After install, the monitor lives at `~/pit_safety/` and launches automatically when the Pi boots to desktop.

### Laptop (for testing)

```bash
pip install opencv-python numpy onnxruntime
python webcam_test.py
```

Pick ONNX for speed or TFLite to simulate Pi behavior.

## Configuration

Edit `config.env` or use the launcher GUI:

| Setting | Default | Description |
|---|---|---|
| `CONFIRM_FRAMES` | `8` | Consecutive frames without glasses before triggering |
| `MIN_CONF` | `0.70` | Detection confidence threshold |
| `PROCESS_EVERY_N` | `2` | Skip frames to save CPU (1 = process all) |
| `CAMERA_INDEX` | `auto` | Camera to use (`auto`, `-1` for Pi cam, `0`+ for USB) |
| `DATA_SAVE_INT` | `0` | Seconds between training image captures (0 = off) |
| `FULLSCREEN` | `1` | Run in fullscreen kiosk mode |
| `DEMO_MODE` | `1` | Touchless mode — no quit button, auto-starts on camera detect |

## Features

- **Auto camera detection** — finds Pi Camera or USB webcams automatically
- **Auto-reconnect** — unplugging the camera doesn't crash it, just waits for reconnection
- **Kiosk/demo mode** — fullscreen, no keyboard needed, starts on boot
- **Training data collection** — optionally saves raw frames for retraining your model
- **Violation snapshots** — saves a JPEG when a violation is first detected
- **Laptop simulator** — test your model and logic without a Pi

## Model

The included model is a YOLOv8n trained on safety glasses detection. Two formats:

- `best_int8.tflite` — INT8 quantized, runs on Pi at ~5-10 FPS
- `best.onnx` — full precision, runs on laptop with ONNX Runtime

To train your own model, collect images using the `DATA_SAVE_INT` feature, label them, and retrain with [Ultralytics YOLOv8](https://docs.ultralytics.com/).

## Requirements

### Raspberry Pi
- Raspberry Pi 4 or 5 (Pi 3 works but slow)
- Pi Camera Module or USB webcam
- Raspberry Pi OS with desktop (Bookworm recommended)
- Monitor (HDMI) for the safety display

### Laptop
- Python 3.9+
- Webcam
- See `requirements.txt`

## License

MIT License — see [LICENSE](LICENSE) for details.
