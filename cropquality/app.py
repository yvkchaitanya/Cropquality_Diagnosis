import os
from flask import Flask, render_template, request, redirect, url_for, flash
from PIL import Image
import numpy as np
import tensorflow as tf

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = 'supersecretkey'

# Categories and Quality Labels
categories = ['Mirchi', 'Mango', 'Lemon', 'Papaya', 'Potato', 'Tomato', 'Banana']
quality_labels = ['90', '70', '60']
IMG_SIZE = 128

# Load the model
model_path = "C:/Users/yvkch/OneDrive/Documents/Projects/cropquality/updated_model.h5"
#model = tf.keras.models.load_model(model_path, custom_objects={'Custom>Adam': tf.keras.optimizers.Adam})  # Adjust custom_objects if needed
model = tf.keras.models.load_model(model_path, compile=False)

# Helper function: Allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Preprocess the uploaded image
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Make predictions
def predict_image(image_path):
    try:
        image_array = preprocess_image(image_path)
        if image_array is None:
            return None, None

        predictions = model.predict(image_array)
        crop_pred = predictions[0]
        quality_pred = predictions[1]

        crop_index = np.argmax(crop_pred)
        quality_index = np.argmax(quality_pred)

        predicted_crop = categories[crop_index]
        predicted_quality = quality_labels[quality_index]

        return predicted_crop, predicted_quality
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# Route: Home Page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the file is part of the request
        if 'file' not in request.files:
            flash('No file part in the request')
            return redirect(request.url)

        file = request.files['file']
        # Check if a file is selected
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)

        # Validate file type
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Predict crop and quality
            predicted_crop, predicted_quality = predict_image(file_path)

            if predicted_crop is None or predicted_quality is None:
                flash('Error during prediction. Please try again.')
                return redirect(request.url)

            # Image URL for display
            image_url = url_for('static', filename='uploads/' + filename)

            # Remove the file after prediction to clean up
            os.remove(file_path)

            return render_template('index.html', crop=predicted_crop, quality=predicted_quality, image_url=image_url)

        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)

    # Render empty form
    return render_template('index.html', crop=None, quality=None, image_url=None)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
