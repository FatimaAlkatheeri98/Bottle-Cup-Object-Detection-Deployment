# Bottle & Cup Object Detection Deployment

## Project Overview

This project deploys a YOLOv8 object detection model that identifies **bottles** and **cups** in uploaded images. The application was developed using **Flask**, containerized with **Docker**, and deployed online using **Render**.

Users can upload an image through a web interface and receive an annotated image with detected objects and confidence scores.

---
## Features

- Detects bottles and cups using YOLOv8
- Upload images through a web interface
- Displays annotated prediction results
- Shows detected object names and confidence scores
- Deployed as an online web application

---

## Technologies Used

- Python
- YOLOv8
- Flask
- Docker
- Render
- OpenCV
- PyTorch

---

## Project Structure

```text
BottleCupDetector/
│
├── app.py
├── best.pt
├── Dockerfile
├── requirements.txt
├── templates/
    └── index.html

```

---

## How to Run Locally

1. Clone the repository.

```bash
git clone <repository-url>
```

2. Install the required packages.

```bash
pip install -r requirements.txt
```

3. Start the application.

```bash
python app.py
```

4. Open your browser and visit:

```
http://localhost:10000
```

---

## Live Deployment

Render:

https://bottle-cup-object-detection-deployment.onrender.com

---

## Demo

Upload an image containing a bottle or cup, then click **Detect Objects** to view the prediction and confidence scores.



---

## Dataset

- Source: Roboflow
- Classes:
  - Bottle
  - Cup
- Annotation Format: YOLOv8

---

## Author

Fatima Alkatheeri

Machine Learning Specialization – ZAKA AI 


