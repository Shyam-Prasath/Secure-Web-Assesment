import cv2
from ultralytics import YOLO
import os
from datetime import datetime

# === Setup Directories ===
os.makedirs("logs", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)
log_file_path = os.path.join("logs", "detections.txt")

# === Load YOLOv8n Model ===
print("[INFO] Loading YOLOv8 model...")
model_path = os.path.join("models", "yolov8n.pt")
yolo_model = YOLO(model_path if os.path.exists(model_path) else "yolov8n.pt")

# === Start Webcam ===
print("[INFO] Starting video capture...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

print("[INFO] Press 'q' to quit.")

# === Run Detection Loop ===
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    frame_count += 1
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # YOLO Object Detection
    results = yolo_model.predict(source=frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls]

            color = (255, 0, 0)
            alert_labels = ["cell phone", "laptop", "book"]

            if label.lower() in alert_labels:
                color = (0, 0, 255)
                # === Log Detection ===
                log_entry = f"[{timestamp}] ALERT: {label} detected at ({x1},{y1}) with {conf:.2f} confidence.\n"
                with open(log_file_path, "a") as log_file:
                    log_file.write(log_entry)

                # === Save Screenshot ===
                screenshot_name = f"screenshots/{label}_{frame_count}.jpg"
                cv2.imwrite(screenshot_name, frame)
                print(log_entry.strip())

                cv2.putText(frame, f'ALERT: {label}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # === Draw Bounding Box ===
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Secure Assessment Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()