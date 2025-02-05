from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
model = tf.keras.models.load_model("blood_group_model.h5")  # Load trained model
categories = ['A+', 'B+', 'O+', 'AB+', 'A-', 'B-', 'O-', 'AB-']  # Blood group labels
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    img = cv2.resize(img, (128, 128))  # Resize to match model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img = preprocess_image(filepath)
            prediction = model.predict(img)
            predicted_label = categories[np.argmax(prediction)]
            return render_template('index.html', prediction=predicted_label, image=filepath)
    return render_template('index.html', prediction=None, image=None)

if __name__ == '__main__':
    app.run(debug=True)
