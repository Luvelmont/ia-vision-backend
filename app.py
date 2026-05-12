from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw
import os
import io
import base64

app = Flask(__name__)
CORS(app)

model = YOLO("yolov8n.pt")

translations = {
    "person": "El proximo Ingeniero en Sistemas",
    "car": "Esto probablemente no es un carro",
    "truck": "Camión tal vez",
    "bus": "Autobús",
    "motorcycle": "Una motito",
    "bicycle": "La bicla",
    "cell phone": "Cel",
    "laptop": "Compu",
    "keyboard": "Teclado",
    "mouse": "Mouse",
    "tv": "Tele",
    "bottle": "Botella",
    "cup": "Vaso",
    "chair": "Silla",
    "couch": "Sofá",
    "bed": "Cama",
    "dog": "Perro",
    "cat": "Gato",
    "bird": "Pájaro",
    "book": "Libro",
    "backpack": "Mochila",
    "clock": "Reloj",
    "banana": "Banana",
    "apple": "Manzana",
    "pizza": "Pizza",
    "sports ball": "Balon, ojala y sea de fucho"
}

@app.route('/')
def home():
    return {
        "message": "IA Vision Lab funcionando"
    }

@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']

    image = Image.open(file.stream).convert("RGB")
    image.thumbnail((640, 640))

    results = model(image)

    detections = []

    draw = ImageDraw.Draw(image)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            english_name = model.names[class_id]
            class_name = translations.get(english_name, english_name)

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "object": class_name,
                "confidence": round(confidence * 100, 2)
            })

            label = f"{class_name} {round(confidence * 100, 1)}%"

            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline="red",
                width=5
            )

            draw.rectangle(
                [(x1, y1 - 30), (x1 + 200, y1)],
                fill="red"
            )

            draw.text(
                (x1 + 5, y1 - 25),
                label,
                fill="white"
            )

    # buffer = io.BytesIO()
    # image.save(buffer, format="JPEG")
    # encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return jsonify({
        "detections": detections
        # "image": f"data:image/jpeg;base64,{encoded_image}"
    })

if __name__ == '__main__':
    app.run(debug=True)