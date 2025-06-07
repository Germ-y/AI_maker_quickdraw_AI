from flask import Flask, render_template, request
import cv2
import numpy as np
import base64
import re
from tensorflow.keras.models import load_model

name="bicycle"

name_dict={name:name,
           'necklace':"necklace"}
app = Flask(__name__)

model = load_model(f"quickdraw_class_model_{name}.keras")
with open(f"categories_{name}.txt", "r") as f:
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

    if name_dict[class_names[np.argmax(prediction)]]=="necklace":
        return f"🫨 {name_dict[class_names[np.argmax(prediction)]]} ({confidence * 100:.1f})점"
    else:
        return f"✅ {name_dict[class_names[np.argmax(prediction)]]} ({confidence * 100:.1f})점"


if __name__ == '__main__':
    app.run(debug=True)
