# Edge AI for Real-Time Wildfire Detection

A real-time wildfire detection system that combines Edge AI inference on a Raspberry Pi with a YOLOv8-based detection model. The system uses a client–server architecture: a Raspberry Pi captures and sends images to a Mac-hosted Flask server, which runs YOLO inference and returns detection results with latency measurements.

---

## Repository Structure

```
Edge-AI-for-Real-Time-Wildfire-Detection/
│
├── Fire/                          # Training images — fire samples
├── Non Fire1/                     # Training images — non-fire samples
├── predict-2/                     # Prediction outputs / results
│
├── fire_nano_trained_int8.tflite  # Quantised INT8 TFLite model for edge deployment
├── server.py                      # Flask inference server (runs on Mac/PC with GPU)
├── client.py                      # Edge client script (runs on Raspberry Pi)
├── received.jpg                   # Last image received by the server
├── test_fire_video.mp4            # Sample test video for validation
└── README.md                      # Project documentation
```

---

## How It Works

The system follows a **split-inference** client–server model:

1. **Client (Raspberry Pi)** captures a fire-scene image and sends it via HTTP POST to the server.
2. **Server (Mac/PC)** receives the image, runs YOLOv8 nano inference, and logs detection count and inference latency.
3. End-to-end round-trip latency is measured and averaged across multiple runs.

Additionally, a quantised **INT8 TFLite model** (`fire_nano_trained_int8.tflite`) is included for fully on-device edge inference without needing a server.

---

## Setup & Installation

### Prerequisites

- Python 3.8+
- A Raspberry Pi (client) and a Mac/Linux machine (server), connected on the same network
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8

### Install Dependencies

**On the server (Mac/PC):**

```bash
pip install flask ultralytics
```

**On the client (Raspberry Pi):**

```bash
pip install requests
```

---

## Usage

### 1. Start the Server

On your Mac/PC, update the model path in `server.py` if needed, then run:

```bash
python server.py
```

The server starts on port `6000` and exposes a `/predict` endpoint.

### 2. Run the Client

On the Raspberry Pi, update the server IP address (`MAC_IP`) in `client.py` to match your Mac's local IP, then run:

```bash
python client.py
```

The client sends the image 21 times, prints per-run latency, and outputs the average end-to-end latency.

---

## API Reference

### `POST /predict`

Accepts a multipart form upload with a single image file.

| Field   | Type | Description              |
|---------|------|--------------------------|
| `image` | file | JPEG/PNG image to analyse |

**Response:**

```json
{ "status": "done" }
```

Server logs inference time (ms) and detection count to stdout.

---

## Model Details

| Model | Format | Optimisation | Use Case |
|-------|--------|--------------|----------|
| `fire_nano_trained.pt` | PyTorch | None | Server-side inference (YOLOv8n) |
| `fire_nano_trained_int8.tflite` | TFLite | INT8 quantisation | On-device edge inference |

The YOLOv8 nano model was trained on the included `Fire` and `Non Fire1` image datasets. The INT8 quantised TFLite version is optimised for deployment on resource-constrained devices such as the Raspberry Pi.

---

## Dataset

The training dataset is organised into two classes:

- **Fire/** — images containing visible fire or smoke
- **Non Fire1/** — images of outdoor scenes without fire

---

## Latency Benchmarking

`client.py` measures both **inference latency** (server-side, printed by `server.py`) and **end-to-end round-trip latency** (client-side, including network transfer). Results from 21 runs are averaged and printed on completion.

---

## Tech Stack

- **Python** — core language
- **YOLOv8 (Ultralytics)** — object detection model
- **TensorFlow Lite** — quantised edge model format
- **Flask** — lightweight HTTP inference server
- **Raspberry Pi** — edge client device

---

## Future Work

- Deploy the TFLite model fully on-device (no server required)
- Add real-time video stream inference via OpenCV
- Integrate SMS/email alerting on fire detection
- Explore further model compression (pruning, lower bit-width quantisation)

---

## License

This project is open source. Feel free to use, modify, and build upon it.
