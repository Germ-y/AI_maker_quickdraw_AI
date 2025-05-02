from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import re
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("quickdraw_5class_model.keras")
with open("categories.txt", "r") as f:
    class_names = f.read().splitlines()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = re.sub('^data:image/.+;base64,', '', data['image'])
    decoded = base64.b64decode(image_data)
    nparr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    resized = cv2.resize(img, (28, 28))
    _, thresh = cv2.threshold(resized, 180, 255, cv2.THRESH_BINARY_INV)
    normalized = thresh / 255.0
    input_img = normalized.reshape(1, 28, 28, 1)

    if np.sum(input_img) < 10:
        return "No Drawing"

    prediction = model.predict(input_img, verbose=0)
    confidence = np.max(prediction)

    if confidence < 0.7:
        return "🫨 잘 모르겠어요.ㅠ"
    else:
        return f"✅ {class_names[np.argmax(prediction)]} ({confidence * 100:.1f}%) 이게 맞나요?"


if __name__ == '__main__':
    app.run(debug=True)
