# NeuralVision AI — YOLOv4 Object Detection Web App

A real-time object detection system built with **YOLOv4**, **OpenCV DNN** and
**Flask**. A webcam feed is annotated with bounding boxes and class labels, streamed
to the browser as MJPEG, and every detection is logged to a **MySQL** database.

![stack](https://img.shields.io/badge/Python-3.10%2B-blue)
![stack](https://img.shields.io/badge/Flask-3.x-black)
![stack](https://img.shields.io/badge/YOLOv4-OpenCV%20DNN-green)

## Features

- **Live detection feed** — YOLOv4 (COCO, 80 classes) run through OpenCV's DNN module
  at 1280x720, confidence threshold 0.5, NMS IoU 0.4.
- **MJPEG streaming** — frames served over `/video_feed`, no WebSocket needed.
- **Detection dashboard** — "NeuralVision AI" UI showing the live feed, model info and
  a live-updating list of currently detected classes (polled from `/stats` every 500 ms).
- **MySQL logging** — each detection is written to a `detection_logs` table, throttled to
  one row per class per 2 seconds. The database and table are created automatically on
  first run.
- **FPS overlay** rendered directly onto the video stream.

## Project structure

```
object/
├── object.py             # Flask app: YOLO inference, video streaming, DB logging
├── config.example.py     # template for MySQL credentials
├── download_weights.py   # fetches yolov4.weights (~246 MB, not stored in git)
├── requirements.txt
├── coco.names            # 80 COCO class labels
├── yolov4.cfg            # Darknet model configuration
├── templates/
│   └── index.html        # dashboard
└── static/css/style.css
```

## Setup

### 1. Requirements

- Python 3.10+
- A MySQL server (XAMPP/WAMP's MySQL is fine)
- A webcam

```bash
pip install -r requirements.txt
```

### 2. Get the model weights

`yolov4.weights` is ~246 MB and is **not** committed to this repository. Fetch it once:

```bash
python download_weights.py
```

Or download it manually from the
[AlexeyAB/darknet yolov4 release](https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4.weights)
and place it next to `object.py`.

`yolov4.cfg` and `coco.names` are already in the repo.

### 3. Configure the database

```bash
cp config.example.py config.py     # Windows: copy config.example.py config.py
```

Then edit `config.py` with your MySQL credentials:

```python
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'your-password',
    'database': 'neuralvision'
}
```

`config.py` is git-ignored, so your credentials stay local. Start your MySQL server
before launching the app — the `neuralvision` database and the `detection_logs` table
are created automatically.

### 4. Run

```bash
python object.py
```

Then open **http://localhost:5000**.

> On some systems the app must run from its own folder so that the relative model paths
> resolve — the code now resolves paths relative to `object.py`, so either works.

## How it works

| Step | What happens |
| --- | --- |
| 1 | OpenCV opens the default webcam at 1280x720. |
| 2 | Each frame becomes a 416x416 blob (`swapRB=True`) and is fed to the Darknet network. |
| 3 | Detections above 0.5 confidence are collected and passed through `cv2.dnn.NMSBoxes`. |
| 4 | Boxes, labels and confidences are drawn; FPS is overlaid top-left. |
| 5 | The frame is JPEG-encoded (quality 95) and yielded as one MJPEG part. |
| 6 | Surviving detections are throttled and inserted into `detection_logs`. |
| 7 | The browser polls `/stats` and lists the current per-class counts. |

## API

| Route | Method | Description |
| --- | --- | --- |
| `/` | GET | Detection dashboard |
| `/video_feed` | GET | MJPEG stream of annotated frames |
| `/stats` | GET | JSON map of `class -> count` for the current frame |

## Notes and limitations

- Inference runs on **CPU** — expect roughly 2–8 FPS depending on hardware. Lower the
  resolution or increase `frame_skip` in `object.py` to trade accuracy for speed.
- The database connection is opened per logged detection rather than pooled.
- `debug=True` in `app.run()` is for development only; use a WSGI server such as
  Waitress or Gunicorn in production.
- The Flask dev server binds to `0.0.0.0:5000`, so the feed is reachable from other
  machines on the same network — the webcam feed is not authenticated.

## Roadmap

- [ ] Swap the per-detection DB connect for a connection pool
- [ ] Add a session history page backed by `detection_logs`
- [ ] Optional GPU inference via `cv2.dnn` CUDA backend
- [ ] Replace the dev server with Waitress

## Acknowledgements

- [YOLOv4](https://github.com/AlexeyAB/darknet) by Alexey Bochkovskiy
- COCO class labels, `yolov4.cfg` and `yolov4.weights` from the Darknet project
