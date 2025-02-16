import os
import openai
import logging
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np
import pandas as pd
from keras.models import load_model
import CharacterSegmentation as cs
from PIL import Image, ImageOps
import keras.backend as K
from flask import render_template

# Configure Logging
logging.basicConfig(level=logging.INFO)

# Define custom F1 Score
def f1_score(y_true, y_pred):
    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'fine_tuned_alphanum_model_binary_ex_88.h5')
SEGMENTED_OUTPUT_DIR = os.path.join(BASE_DIR, 'segmented/')
EMNIST_PATH = os.path.join(BASE_DIR, 'data/emnist/emnist/')
mapping_processed = os.path.join(EMNIST_PATH, 'processed-mapping.csv')

# OpenAI API Key (Load from Environment Variable)
openai.api_key = 'Hidden Code'
# Global Variables
model = None
code2char = {}

@app.before_request
def load_resources():
    global model, code2char
    logging.info("Loading model and label mappings...")
    
    # Load the trained model
    model = load_model(MODEL_PATH, custom_objects={'f1_score': f1_score})

    # Load the label mappings
    df = pd.read_csv(mapping_processed)
    code2char = {row['id']: row['char'] for _, row in df.iterrows()}

# Function to preprocess segmented images
def preprocess_image(filepath):
    img = Image.open(filepath).resize((28, 28)).convert('L')
    img = ImageOps.invert(img)
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    return img_array

# Function to solve equation using OpenAI API (ChatGPT)
def solve_equation(equation):
    prompt = f"Solve this equation: {equation}"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"Error solving equation: {str(e)}")
        return f"Error solving equation: {str(e)}"

# Route to serve MathAnalyser.html from the Website folder
@app.route("/math-analyser", methods=["GET"])
def serve_html():
    # Set the path to the correct 'Website' folder
    website_dir = '../../src/templates'
    
    # Check if the MathAnalyser.html exists in the specified directory
    html_path = os.path.join(website_dir, 'MathAnalyser.html')
    print(f"Looking for MathAnalyser.html at: {html_path}")

    if os.path.exists(html_path):
        return send_from_directory(website_dir, 'MathAnalyser.html')
    else:
        logging.error(f"File MathAnalyser.html not found at: {html_path}")
        return jsonify({"error": "File not found!"}), 404

@app.route("/math-analyser", methods=["POST"])
def predict_image():
    # Your existing POST request logic goes here (no change needed)
    try:
        # Ensure the segmented directory exists
        if not os.path.exists(SEGMENTED_OUTPUT_DIR):
            os.makedirs(SEGMENTED_OUTPUT_DIR)

        # Clear the segmented folder
        for file in os.listdir(SEGMENTED_OUTPUT_DIR):
            file_path = os.path.join(SEGMENTED_OUTPUT_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save uploaded file
        file = request.files['file']
        if 'drawing' in file.filename:  # Check if the file is the drawing image (based on filename or any other logic)
            # Save the drawing image
            drawing_image_path = os.path.join(SEGMENTED_OUTPUT_DIR, 'drawing_image.png')
            file.save(drawing_image_path)
            
            # Perform character segmentation on the drawing image
            logging.info("Starting character segmentation for drawing image...")
            cs.image_segmentation(drawing_image_path)
        
        else:
            # Save the uploaded image (non-drawing)
            input_image_path = os.path.join(SEGMENTED_OUTPUT_DIR, 'input_image.png')
            file.save(input_image_path)

            # Perform character segmentation on the uploaded image
            logging.info("Starting character segmentation for input image...")
            cs.image_segmentation(input_image_path)

        # Process each segmented image
        segmented_files = sorted([os.path.join(SEGMENTED_OUTPUT_DIR, f) for f in os.listdir(SEGMENTED_OUTPUT_DIR) if f.endswith('.jpg')])
        if not segmented_files:
            return jsonify({"error": "No segmented images found!"}), 400

        predicted_string = ""
        for seg_file in segmented_files:
            X_data = preprocess_image(seg_file)
            pred = model.predict(X_data)
            pred_label = np.argmax(pred)
            predicted_string += code2char.get(pred_label, "?")

        # Now use the predicted equation with ChatGPT to solve it
        solved_equation = solve_equation(predicted_string)

        # Return the prediction and the solved equation
        result = {
            "predicted_equation": predicted_string,
            "solved_equation": solved_equation
        }

        return jsonify(result), 200

    except Exception as e:
        logging.error(f"Error in /math-analyser: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    

if __name__ == "__main__":
    logging.info("Starting Flask Backend...")
    app.run(debug=True)