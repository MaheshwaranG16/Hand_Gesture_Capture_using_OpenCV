# Hand_Gesture_Capture_using_OpenCV
 
## Overview

This project captures hand gesture images using OpenCV and MediaPipe and saves them in a specified folder. It allows both automatic and manual image saving while displaying detected hand landmarks in real-time.

## Features

- Automatic Image Capture (Saves a fixed number of images per session)

- Hand Landmark Detection using MediaPipe

- Manual Image Capture (Press 's' to save extra images)

- Restart Capture Session (Press 'r' to reset image counter)

- Quit Anytime (Press 'q' to exit)

## Installation

Ensure you have Python and the required libraries installed:
```bash
pip install opencv-python mediapipe numpy
```
## Usage

Run the script to start capturing hand gestures:
```bash
python Hand_Gesture_Capture.py
```

## Controls

| Key  | Action |
| ------------- | ------------- |
| s | Manually save an image  |
| r | Restart automatic image saving  |
| q | Quit the program  |


## Example Output

When running the script, you will see a window displaying detected hand landmarks. Images will be saved automatically, and you can manually capture extra images as needed.
