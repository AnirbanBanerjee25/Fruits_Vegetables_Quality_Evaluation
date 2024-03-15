# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('Quality_Evaluation_Model.h5')

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(100, 100))  # Adjust target size as per your model's input dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return 'No selected file'

    # Check if the file has an allowed extension
    if file:
        # Save the uploaded file to the uploads folder
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the uploaded image
        img_array = preprocess_image(filepath)

        # Make predictions using the model
        prediction = model.predict(img_array)

        # Convert the prediction probabilities to class labels
        predicted_class = np.argmax(prediction)
        class_labels = ['rottentomato', 'freshtomato', 'rottenpotato', 'freshpotato', 'rottenoranges', 'freshoranges', 
                'rottenokra', 'freshokra', 'rottencucumber', 'freshcucumber', 'rottencapsicum', 'freshcapsicum', 
                'rottenbittergroud', 'freshbittergroud', 'rottenbanana', 'freshbanana', 'rottenapples', 'freshapples']
        predicted_label = class_labels[predicted_class]

        # Return the prediction
        return f'File uploaded successfully. Prediction: {predicted_label}'

    else:
        return 'Invalid file'

if __name__ == '__main__':
    app.run(debug=True)
