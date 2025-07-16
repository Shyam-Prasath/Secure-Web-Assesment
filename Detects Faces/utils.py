import cv2
import os
import numpy as np
from datetime import datetime

# Load face cascade globally so it's not loaded repeatedly
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(gray_frame):
    faces = face_cascade.detectMultiScale(
        gray_frame, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(60, 60)
    )
    return faces

def analyze_motion(fgmask):
    motion_level = np.sum(fgmask) / 255
    return motion_level

def log_cheating(event_text):
    if not os.path.exists("logs"):
        os.makedirs("logs")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("logs/cheating_log.txt", "a") as f:
        f.write(f"[{timestamp}] {event_text}\n")

def save_screenshot(frame, event_text):
    if not os.path.exists("screenshots"):
        os.makedirs("screenshots")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/{timestamp}_{event_text.replace(' ', '_')}.jpg"
    cv2.imwrite(filename, frame)
