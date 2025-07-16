import cv2
import numpy as np
from utils import detect_faces, analyze_motion, log_cheating, save_screenshot

def main():
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    print("[INFO] Starting exam monitoring... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Face detection
        faces = detect_faces(gray)

        # Motion detection
        fgmask = fgbg.apply(gray)
        motion_level = analyze_motion(fgmask)

        # Behavior analysis
        cheating_flags = []
        if len(faces) == 0:
            cheating_flags.append("No face detected")
        if len(faces) > 1:
            cheating_flags.append("Multiple faces detected")
        if motion_level > 50000:
            cheating_flags.append("Excessive motion detected")

        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show alerts on screen
        y0 = 30
        for flag in cheating_flags:
            cv2.putText(frame, flag, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y0 += 30

        # Log and save screenshots
        for flag in cheating_flags:
            log_cheating(flag)
            save_screenshot(frame, flag)

        cv2.imshow("Exam Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
