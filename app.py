from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import base64
import io
import os
import requests

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

API_KEY = os.environ.get("GOOGLE_VISION_API")

available_transforms = {
    "original": "Imagen original",
    "grayscale": "Escala de grises"
}

def apply_transform(image, transform):
    if transform == "original":
        return image

    if transform == "grayscale":
        return image.convert("L").convert("RGB")

    raise ValueError("Transformacion no soportada")

def encode_image(image, quality=80):
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=quality, optimize=True)
    encoded_result = base64.b64encode(output.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_result}"

@app.route("/")
def home():
    return {"message": "Google Vision API funcionando"}

@app.route("/transform", methods=["POST"])
def transform_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    transform = request.form.get("transform", "grayscale")

    if transform not in available_transforms:
        return jsonify({
            "error": "Unsupported transform",
            "availableTransforms": available_transforms
        }), 400

    image = Image.open(request.files["image"].stream).convert("RGB")
    image.thumbnail((900, 900))
    image = apply_transform(image, transform)

    return jsonify({
        "image": encode_image(image, quality=85),
        "transform": transform,
        "transformLabel": available_transforms[transform]
    })

@app.route("/predict", methods=["POST"])
def predict():
    if not API_KEY:
        print("NO EXISTE API KEY")
        return jsonify({"error": "Missing GOOGLE_VISION_API"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    transform = request.form.get("transform", "original")

    if transform not in available_transforms:
        return jsonify({
            "error": "Unsupported transform",
            "availableTransforms": available_transforms
        }), 400

    image = Image.open(file.stream).convert("RGB")
    image.thumbnail((700, 700))
    image = apply_transform(image, transform)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    url = f"https://vision.googleapis.com/v1/images:annotate?key={API_KEY}"

    payload = {
        "requests": [
            {
                "image": {
                    "content": encoded_image
                },
                "features": [
                    {
                        "type": "OBJECT_LOCALIZATION",
                        "maxResults": 20
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as error:
        print(error)
        return jsonify({"error": "Google Vision request failed"}), 502

    result = response.json()
    print(result)

    if "error" in result:
        return jsonify(result), 500

    objects = result["responses"][0].get("localizedObjectAnnotations", [])

    detections = []
    draw = ImageDraw.Draw(image)

    width, height = image.size

    for obj in objects:
        class_name = obj["name"]
        confidence = round(obj["score"] * 100, 2)

        vertices = obj["boundingPoly"]["normalizedVertices"]

        x_values = [vertex.get("x", 0) * width for vertex in vertices]
        y_values = [vertex.get("y", 0) * height for vertex in vertices]

        x1, x2 = min(x_values), max(x_values)
        y1, y2 = min(y_values), max(y_values)

        detections.append({
            "object": class_name,
            "confidence": confidence
        })

        label = f"{class_name} {confidence}%"
        label_width = max(180, min(len(label) * 9 + 20, 340))

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="#0ea5e9",
            width=5
        )

        draw.rectangle(
            [(x1, max(y1 - 30, 0)), (x1 + label_width, y1)],
            fill="#0ea5e9"
        )

        draw.text(
            (x1 + 5, max(y1 - 25, 0)),
            label,
            fill="white"
        )

    return jsonify({
        "detections": detections,
        "image": encode_image(image, quality=80),
        "transform": transform,
        "transformLabel": available_transforms[transform]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
