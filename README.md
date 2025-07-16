# Secure Web Assessment System

## 📌 Abstract

This project presents a **dual-layered intelligent monitoring system** aimed at enhancing the **security and integrity of online assessments** through real-time **object detection** and **behavioural analysis**.

### 🔍 Component 1: Object Detection (YOLOv8)

- Utilizes the **YOLOv8 deep learning model** to detect unauthorized objects like:
  - 📱 Mobile Phones  
  - 💻 Laptops  
  - 📚 Books
- Features:
  - Live webcam session monitoring
  - Visual highlights and bounding boxes on detected objects
  - Logging of detection events with timestamps
  - Automatic screenshot capture for audit trails

### 🎯 Component 2: Behavioural Monitoring (Traditional CV)

- Uses **traditional computer vision techniques** for:
  - Face detection
  - Motion analysis
- Flags suspicious behavior such as:
  - 🚫 No face detected
  - 👥 Multiple faces detected
  - 🔄 Excessive or unusual movement
- Triggers:
  - Activity logging
  - Screenshot capture

### ✅ Why It Matters

In an era where **remote assessments** and **online exams** are the norm, this project provides a robust, AI-powered proctoring solution. By combining **deep learning precision** with **rule-based visual analytics**, it ensures:
- Higher exam integrity
- Real-time violation detection
- Scalable and automated proctoring workflows
