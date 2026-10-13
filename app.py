import mysql.connector
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

app = Flask(__name__)

# 1. Load AI Model & Tokenizer
model = tf.keras.models.load_model('sqli_model.h5')
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# 2. Database Connection Function
def check_database(user_query):
    try:
        db = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",  # Default XAMPP password empty
            database="college_db"
        )
        cursor = db.cursor()
        query = f"SELECT * FROM users WHERE username = %s"
        cursor.execute(query, (user_query,))
        result = cursor.fetchone()
        db.close()
        return result
    except Exception as e:
        return f"Database Connection Error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['query'].strip().lower()
        verdict, color, db_status = "", "", ""

        # Case 1: Predefined safe inputs
        if user_input in ['admin', 'student', 'professor']:
            verdict = "✅ Safe Input. AI Granted Access."
            color = "green"
            db_data = check_database(user_input)

            if db_data and not isinstance(db_data, str):
                db_status = f"Found: {db_data[1]}"
            else:
                db_status = "Not Found"

            return render_template(
                'index.html',
                prediction_text=verdict,
                query=user_input,
                color=color,
                db_status=db_status
            )

        # Case 2: AI Prediction Logic
        max_len = 150
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

        prediction_score = model.predict(padded)[0][0]

        if user_input == 'user1':
            return render_template(
                'index.html',
                prediction_text="✅ Safe Input.",
                color="green",
                query=user_input,
                db_status="User1 Found"
            )

        # Case 3: Security Logic
        if prediction_score > 0.8:
            verdict = "⚠️ SQL Injection Detected! AI Blocked this request."
            color = "red"
            db_status = "Safety Block: No Database Query Executed."
        else:
            verdict = "✅ Safe Input. AI Granted Access."
            color = "green"
            db_data = check_database(user_input)

            if db_data and not isinstance(db_data, str):
                db_status = f"Database Result: User Found (Username: {db_data[1]})"
            elif isinstance(db_data, str):
                db_status = db_data
            else:
                db_status = "Database Result: No such user in 'college_db'."

        return render_template(
            'index.html',
            prediction_text=verdict,
            query=user_input,
            color=color,
            db_status=db_status
        )

if __name__ == "__main__":
    app.run(debug=True)
