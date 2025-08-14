from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = tf.keras.models.load_model('cat_dog_classifier_model.h5')

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_SIZE = (150, 150)

def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
    
        file = request.files['file']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            prediction = model.predict(img_array)[0][0]

            label = 'Dog' if prediction > 0.5 else 'Cat'
            confidence = prediction if prediction > 0.5 else 1 - prediction

            return render_template('index.html', label=label, confidence=confidence, image_url=filepath)

    return render_template('index.html', label=None)

if __name__ == '__main__':
    app.run(debug=True)
