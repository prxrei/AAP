##################################################################################### IMPORTS #####################################################################################

from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageOps
import cv2
import numpy as np
from keras.models import load_model
from keras import layers, Model
import keras.backend as K
import keras.utils as image
import tensorflow as tf

import joblib
import pickle
import pandas as pd
import os
from dotenv import load_dotenv
import subprocess
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from Insan import CharacterSegmentation as cs
import openai
import io
import logging
from multiprocessing import Process
import webbrowser

##################################################################################### FILE ROUTING #####################################################################################

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/math-analyser')
def math_analyser():
    return render_template('MathAnalyser.html')

@app.route('/Attendance-taking')
def Attendance_taking():
    return render_template('AttendanceTaking.html')

@app.route('/AddNewIdentity')
def AddNewIdentity():
    return render_template('CaptureNewIdentity.html')

@app.route('/science-classifier')
def science_classifier():
    return render_template('science_classifier.html')

@app.route('/')
def nav():
    return render_template('Nav.html')

load_dotenv()

def run_app():
    app.run(debug=True, port=5000)  # Run Flask Frontend on port 5000

##################################################################################### ALESS PART #####################################################################################

l2_regularisation = 1e-3


def create_embedding_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = tf.image.rgb_to_grayscale(inputs)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_regularisation)
                      )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_regularisation)
                      )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_regularisation)
                      )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(96, (5, 5), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_regularisation)
                      )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (5, 5), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_regularisation)
                      )(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Lambda(lambda x: tf.math.l2_normalize(
        x, axis=-1), name="embedding_normalization")(x)

    return Model(inputs, x, name="EmbeddingModel")


def triplet_loss(margin=0.25):
    def loss(y_pred):
        anchor, positive, negative = tf.split(
            y_pred, num_or_size_splits=3, axis=1)
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss


embedding_model = create_embedding_model((112, 92, 3))
embedding_model.load_weights("../src/Aless/embedding_model.h5")

with open("../src/Aless/embeddings.pkl", "rb") as f:
    train_data = pickle.load(f)

t_embeddings = train_data["embeddings"]
t_labels = train_data["labels"]

input_dim = (92, 112)


def load_images(file, id=None):
    """Processes an uploaded image file (JPEG, PNG, AVIF, WebP, PGM) and resizes it."""
    data = []

    image = np.array(file)

    try:
        image_resized = cv2.resize(image, input_dim)
        image_expanded = np.expand_dims(image_resized, axis=0)
    except Exception as e:
        # Catch any exception and print the error message
        print(f"Error opening image: {e}")
        return None

    if id is not None:
        # Store image and label
        data.append(image_expanded)
        return np.array(data)
    else:
        data.append(image_resized)
        return np.array(data)


def recognize_face(test_embedding, stored_embeddings, stored_labels, threshold):
    distances = [np.linalg.norm(test_embedding - emb)
                 for emb in stored_embeddings]

    min_distance = min(distances)
    best_match_idx = np.argmin(distances)

    if min_distance < threshold:
        person = stored_labels[best_match_idx]
    else:
        person = "Person not recognised"

    return person, min_distance


def add_embedding(embedding_file, image, ID, embedding_model=embedding_model):
    with open(embedding_file, "rb") as f:
        data = pickle.load(f)

    embeddings_list = data["embeddings"]
    labels_list = data["labels"]

    processed_image = load_images(image, ID)
    new_embedding = embedding_model.predict(processed_image)
    new_identity = ID

    embeddings = np.append(embeddings_list, new_embedding, axis=0)
    labels = np.append(labels_list, new_identity)

    updated_data = {"embeddings": embeddings, "labels": labels}

    with open(embedding_file, "wb") as f:
        pickle.dump(updated_data, f)

    with open(embedding_file, "rb") as f:
        data = pickle.load(f)

    embeddings_list = list(data["embeddings"])
    labels_list = list(data["labels"])

    return embeddings, labels


def read_pgm(file):
    """Reads a PGM file and converts it to a NumPy array."""
    with file.stream as f:
        header = f.readline().decode().strip()
        if header != "P5":
            raise ValueError("Only binary PGM (P5) format is supported.")

        # Read width, height, and max gray value
        width, height = map(int, f.readline().decode().split())

        # Read pixel data
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape((height, width))

        # Convert NumPy array to PIL Image
        return Image.fromarray(data, mode="L")  # "L" mode for grayscale images


@app.route("/FaceRecognition", methods=["POST"])
def PredictFace():
    file = request.files.get('file')

    filename = secure_filename(file.filename)

    if filename.lower().endswith(".pgm"):
        img = read_pgm(file)
    else:
        img = Image.open(io.BytesIO(file.read())).convert(
            "RGB")

    if img is None:
        raise ValueError("Failed to decode the image.")

    proc_img = load_images(img)

    new_embedding = embedding_model.predict(proc_img)

    person, min_distance = recognize_face(
        new_embedding, t_embeddings, t_labels, 0.45)

    min_distance = f"{min_distance:.3f}"

    return jsonify({
        "result": [
            {"Predicted person": person, "Distance": min_distance}
        ]
    })


@app.route("/AddIdentity", methods=["POST"])
def AddIdentity():
    try:
        id = request.form.get('id')
        if not id:
            return jsonify({"success": False, "message": "ID is required"}), 400

        files = [request.files[key]
                 for key in request.files if request.files[key].filename != '']

        if not files:
            return jsonify({"success": False, "message": "No files provided"}), 400

        for file in files:
            filename = secure_filename(file.filename)

            # Determine file type
            if filename.lower().endswith(".pgm"):
                # Read PGM file and convert it to a PIL image
                image = read_pgm(file)
            else:
                image = Image.open(io.BytesIO(file.read())).convert(
                    "RGB")  # Convert other images to RGB
            add_embedding("../src/Aless/embeddings.pkl", image, id)  # Pass the image object

        return jsonify({"success": True, "message": "New identity added successfully"}), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    

##################################################################################### RYAN PART #####################################################################################


LogRegmodel = joblib.load("../src/Ryan/LogRegModel.pkl")


features = [
    "Attendance", "Age",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular Units 1st sem (grade)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (without evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular Units 2nd sem (grade)"
]


@app.route("/Predictresults", methods=["GET", "POST"])
def Predictresults():
    if request.method == "GET":
        return render_template("Predictresults.html")
    else:

        input_data = request.form

        data = {
            feature: int(input_data.get(feature, 0)) for feature in features
        }

        prediction = LogRegmodel.predict([list(data.values())])

        prediction_result = "Failing" if prediction[0] == 0 else "Passing"

        return redirect(url_for("result", result=prediction_result))


@app.route("/Prediction", methods=["GET"])
def result():

    result = request.args.get("result")

    return render_template("Prediction.html", result=result)


##################################################################################### INSAN PART #####################################################################################

logging.basicConfig(level=logging.INFO)

# Define custom F1 Score
def f1_score(y_true, y_pred):
    precision = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon())
    recall = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) / (K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon())
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Insan', 'fine_tuned_alphanum_model_binary_ex_88.h5')
SEGMENTED_OUTPUT_DIR = os.path.join(BASE_DIR, 'Insan', 'segmented/')
EMNIST_PATH = os.path.join(BASE_DIR, 'Insan', 'data/emnist/emnist/')
mapping_processed = os.path.join(EMNIST_PATH, 'processed-mapping.csv')

# OpenAI API Key (Load from Environment Variable)
openai.api_key = os.getenv("OPENAI_API_KEY")
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


##################################################################################### JOHAN PART #####################################################################################

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Initialize OpenAI client (New API)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Load pre-trained Science Classifier model
SCmodel = tf.keras.models.load_model('./Johan/trainedmodel/science_nlp_rnn_model.h5')

# Load Tokenizer
with open("./Johan/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load LabelEncoder
with open("./Johan/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ✅ Corrected OpenAI function using latest API
def generate_openai_explanation(topic, model="gpt-3.5-turbo", max_tokens=1000):
    """ Uses OpenAI API to generate a structured explanation. """
    
    prompt = f"Explain {topic} in a detailed, structured, and easy-to-understand manner."
    
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful AI that explains science topics in detail."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()

@app.route('/')
def index():
    return render_template('science_classifier.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_query = data.get('query', '')
        print(f"Received query: {user_query}")

        if not user_query:
            raise ValueError("No query provided")

        print("Predicting category...")
        tokens = tokenizer.encode(user_query, truncation=True, max_length=128, add_special_tokens=True)
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences([tokens], maxlen=128, padding='post')

        prediction = SCmodel.predict(padded_sequence)
        predicted_label = np.argmax(prediction)

        topic_category = label_encoder.inverse_transform([predicted_label])[0]
        print(f"Predicted category: {topic_category}")

        print("Generating OpenAI elaboration...")
        elaboration = generate_openai_explanation(topic_category, model="gpt-4", max_tokens=1000)
        print(f"Elaboration: {elaboration}")

        return jsonify({
            'topic': topic_category,
            'response': elaboration
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create two processes: one for backend and one for frontend) 
    app_run= Process(target=run_app)

    # Start both processes
    app_run.start()

    # Open the frontend in the web browser after both processes are started
    webbrowser.open('http://127.0.0.1:5000/')  # Frontend URL

    # Wait for both processes to finish
    app_run.join()
