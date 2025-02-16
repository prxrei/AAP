from flask import Flask, jsonify, request

import tensorflow as tf
import keras.utils as image

import numpy as np
from flask_cors import CORS

import pickle
import cv2
import io
from PIL import Image
import os
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from keras import layers, Model



app = Flask(__name__)
CORS(app)

l2_regularisation = 1e-3

def create_embedding_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    x = tf.image.rgb_to_grayscale(inputs)

    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same'
                      ,kernel_regularizer=tf.keras.regularizers.l2(l2_regularisation)
                     )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same'
                      ,kernel_regularizer=tf.keras.regularizers.l2(l2_regularisation)
                     )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same'
                      ,kernel_regularizer=tf.keras.regularizers.l2(l2_regularisation)
                     )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(96, (5, 5), activation='relu', padding='same'
                      ,kernel_regularizer=tf.keras.regularizers.l2(l2_regularisation)
                     )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (5, 5), activation='relu', padding='same'
                      ,kernel_regularizer=tf.keras.regularizers.l2(l2_regularisation)
                     )(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), name="embedding_normalization")(x)

    return Model(inputs, x, name="EmbeddingModel")

def triplet_loss(margin=0.25):
    def loss(y_pred):
        anchor, positive, negative = tf.split(y_pred, num_or_size_splits=3, axis=1)
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        return tf.reduce_mean(tf.maximum(pos_dist - neg_dist + margin, 0.0))
    return loss



embedding_model = create_embedding_model((112, 92, 3))
embedding_model.load_weights("embedding_model.h5")
    
with open("embeddings.pkl", "rb") as f:
    train_data = pickle.load(f)

t_embeddings = train_data["embeddings"]
t_labels = train_data["labels"]

input_dim = (92, 112)

def load_images(file, id=None):
    """Processes an uploaded image file (JPEG, PNG, AVIF, WebP, PGM) and resizes it."""
    data = []
    labels = []

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
        labels.append(id)
        return np.array(data), np.array(labels)
    else:
        data.append(image_resized)
        return np.array(data)




def recognize_face(test_embedding, stored_embeddings, stored_labels, threshold):
    distances = [np.linalg.norm(test_embedding - emb) for emb in stored_embeddings]

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
    new_embedding = embedding_model.predict(processed_image[0])
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

    person, min_distance= recognize_face(new_embedding, t_embeddings, t_labels, 0.45)

    print(person, min_distance)
            
    min_distance = f"{min_distance:.3f}"
    
    return jsonify({
        "result": [
            {"Predicted person": person, "Distance": min_distance}
        ]
    })



@app.route("/AddNewIdentity", methods=["POST"])
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
            print("Processing file:", file.filename)

            filename = secure_filename(file.filename)

            # Determine file type
            if filename.lower().endswith(".pgm"):
                # Read PGM file and convert it to a PIL image
                image = read_pgm(file)
            else:
                image = Image.open(io.BytesIO(file.read())).convert(
                    "RGB")  # Convert other images to RGB

            add_embedding("embeddings.pkl", image, id)  # Pass the image object
            print("Success: Added embedding")

        return jsonify({"success": True, "message": "New identity added successfully"}), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=False)
    
    