# TODO List for Deploying Object Detection as Web App and Improving Clarity

- [x] Create requirements.txt with Flask and opencv-python dependencies
- [x] Modify object.py to integrate Flask for video streaming, adjust frame_skip to 1, set resolution to 1280x720, increase confidence to 0.5
- [x] Install dependencies from requirements.txt
- [x] Run the Flask app and test video feed in browser
- [x] Move MySQL credentials out of source into config.py (git-ignored), add config.example.py
- [x] Stop tracking yolov4.weights (246 MB) and add download_weights.py
- [x] Resolve model/data paths relative to object.py so the app can run from any cwd
- [x] Write README with setup, API and limitations
- [ ] Replace the dev server with Waitress/Gunicorn for deployment
- [ ] Add authentication before exposing the feed beyond localhost
