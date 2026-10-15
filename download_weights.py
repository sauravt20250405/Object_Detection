"""Download the YOLOv4 weights file required by object.py.

The yolov4.weights file (~246 MB) is too large for a plain git push, so it is
not stored in this repository. Run this script once before starting the app:

    python download_weights.py
"""

import os
import sys
import urllib.request

WEIGHTS_URL = (
    "https://github.com/AlexeyAB/darknet/releases/download/yolov4/"
    "yolov4.weights"
)
WEIGHTS_FILE = "yolov4.weights"
MIN_SIZE = 240 * 1024 * 1024  # sanity check: the real file is ~246 MB


def already_present() -> bool:
    return os.path.isfile(WEIGHTS_FILE) and os.path.getsize(WEIGHTS_FILE) > MIN_SIZE


def report(count: int, block: int, total: int) -> None:
    if total <= 0:
        return
    done = min(count * block / total * 100, 100.0)
    sys.stdout.write(f"\rDownloading yolov4.weights ... {done:5.1f}%")
    sys.stdout.flush()


def main() -> int:
    if already_present():
        print(f"{WEIGHTS_FILE} already present, nothing to do.")
        return 0

    print(f"Fetching {WEIGHTS_URL}")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_FILE, reporthook=report)
    except Exception as exc:  # noqa: BLE001 - surface any download failure to the user
        print(f"\nDownload failed: {exc}")
        print(
            "Download the file manually from\n"
            f"  {WEIGHTS_URL}\n"
            f"and place it in this folder as {WEIGHTS_FILE}."
        )
        return 1

    print()
    if not already_present():
        print("Downloaded file looks incomplete - please retry.")
        return 1
    print(f"Saved {WEIGHTS_FILE} ({os.path.getsize(WEIGHTS_FILE) / 1024 / 1024:.1f} MB).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
