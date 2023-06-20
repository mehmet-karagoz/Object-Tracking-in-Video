# Object Tracking in Video
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![OpenCV 4.5](https://img.shields.io/badge/OpenCV-4.5-blue)](https://opencv.org/releases/)
[![YOLOv4](https://img.shields.io/badge/YOLOv4--blue)](https://github.com/AlexeyAB/darknet)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

>This repository contains scripts for object tracking in video using OpenCV's built-in trackers and the YOLOv4 object detection model.

## Files

- `coco.names`: This file contains the names of the 80 object classes that the YOLOv4 model can detect.
- `main.py`: This script uses OpenCV's built-in KCF tracker to track an object in a video.
- `requirements.txt`: This file lists all the Python dependencies that you need to run the scripts.
- `second.py`: This script uses the YOLOv4 object detection model to detect objects in a video and then uses OpenCV's built-in CSRT tracker to track one of the detected objects.
- `test.mp4`: This is the video in which you want to track objects.
- `yolov4.cfg`: This is the configuration file for the YOLOv4 model.

## Usage

1. Install the required Python dependencies:

```bash
pip install -r requirements.txt
```
2. Run the main.py script to track an object using the KCF tracker:
```bash
python main.py
```
3. Run the second.py script to detect objects using the YOLOv4 model and track one of the detected objects using the CSRT tracker:
```bash
python second.py
```
**Note**:
> When you run the scripts, a window will open showing the first frame of the video. You need to select the object that you want to track by drawing a bounding box around it. After you have selected the object, press ENTER to start tracking.

## Examples
