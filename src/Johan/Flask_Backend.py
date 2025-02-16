import os
import openai
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import tensorflow as tf

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Initialize OpenAI client (New API)
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained Science Classifier model
SCmodel = tf.keras.models.load_model('./trainedmodel/science_nlp_rnn_model.h5')

# Load Tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load LabelEncoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# ✅ Corrected OpenAI function using latest API
def generate_openai_explanation(topic, model="gpt-4", max_tokens=1000):
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
    app.run(debug=True)
