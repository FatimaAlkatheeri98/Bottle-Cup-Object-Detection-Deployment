import os
import uuid
from pathlib import Path

import cv2
import torch
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"
MODEL_PATH = BASE_DIR / "best.pt"

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

model = YOLO(str(MODEL_PATH))


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    uploaded_image = None
    detections = []
    error = None

    if request.method == "POST":

        if "image" not in request.files:
            error = "Please upload an image."
            return render_template(
                "index.html",
                error=error,
                result_image=result_image,
                uploaded_image=uploaded_image,
                detections=detections
            )

        file = request.files["image"]

        if file.filename == "":
            error = "No image was selected."
            return render_template(
                "index.html",
                error=error,
                result_image=result_image,
                uploaded_image=uploaded_image,
                detections=detections
            )

        if not allowed_file(file.filename):
            error = "Only JPG, JPEG, and PNG images are allowed."
            return render_template(
                "index.html",
                error=error,
                result_image=result_image,
                uploaded_image=uploaded_image,
                detections=detections
            )

        original_name = secure_filename(file.filename)
        extension = original_name.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex}.{extension}"

        input_path = UPLOAD_FOLDER / unique_name
        output_path = RESULT_FOLDER / unique_name

        file.save(input_path)

        with torch.inference_mode():
            prediction_results = model.predict(
                source=str(input_path),
                conf=0.25,
                imgsz=160,
                device="cpu",
                max_det=10,
                save=False,
                verbose=False
            )

        prediction = prediction_results[0]
        annotated_image = prediction.plot()

        cv2.imwrite(str(output_path), annotated_image)

        if prediction.boxes is not None:
            for class_id, confidence in zip(
                prediction.boxes.cls.tolist(),
                prediction.boxes.conf.tolist()
            ):
                detections.append(
                    {
                        "class_name": model.names[int(class_id)],
                        "confidence": round(float(confidence) * 100, 2)
                    }
                )

        uploaded_image = url_for(
            "static",
            filename=f"uploads/{unique_name}"
        )

        result_image = url_for(
            "static",
            filename=f"results/{unique_name}"
        )

    return render_template(
        "index.html",
        result_image=result_image,
        uploaded_image=uploaded_image,
        detections=detections,
        error=error
    )


@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        result_image=None,
        uploaded_image=None,
        detections=[],
        error="The uploaded image is larger than 10 MB."
    ), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
    
