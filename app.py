from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw
import requests
import io
import base64
import os

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

API_KEY = os.environ.get("GOOGLE_VISION_API_KEY")

translations = {
    "Person": "El próximo Ingeniero en Sistemas",
    "Car": "Carro",
    "Vehicle": "Vehículo",
    "Mobile phone": "Celular",
    "Laptop": "Compu",
    "Computer keyboard": "Teclado",
    "Computer mouse": "Mouse",
    "Television": "Tele",
    "Bottle": "Botella",
    "Drinkware": "Vaso",
    "Chair": "Silla",
    "Couch": "Sofá",
    "Bed": "Cama",
    "Dog": "Perro",
    "Cat": "Gato",
    "Bird": "Pájaro",
    "Book": "Libro",
    "Backpack": "Mochila",
    "Ball": "Balón"
}

@app.route("/")
def home():
    return {"message": "Google Vision API funcionando"}

@app.route("/predict", methods=["POST"])
def predict():
    if not API_KEY:
        return jsonify({"error": "Missing GOOGLE_VISION_API_KEY"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    image = Image.open(file.stream).convert("RGB")
    image.thumbnail((900, 900))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
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

    response = requests.post(url, json=payload)
    result = response.json()

    if "error" in result:
        return jsonify(result), 500

    objects = result["responses"][0].get("localizedObjectAnnotations", [])

    detections = []
    draw = ImageDraw.Draw(image)

    width, height = image.size

    for obj in objects:
        english_name = obj["name"]
        class_name = translations.get(english_name, english_name)
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

        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=5
        )

        draw.rectangle(
            [(x1, max(y1 - 30, 0)), (x1 + 260, y1)],
            fill="red"
        )

        draw.text(
            (x1 + 5, max(y1 - 25, 0)),
            label,
            fill="white"
        )

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=80, optimize=True)

    encoded_result = base64.b64encode(output.getvalue()).decode("utf-8")

    return jsonify({
        "detections": detections,
        "image": f"data:image/jpeg;base64,{encoded_result}"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)