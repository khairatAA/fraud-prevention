from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
from google.cloud import storage
import io
import os
import pickle

app = Flask(__name__, template_folder="templates")

BUCKET_NAME = "innovators1"
MODEL_FILE_NAME = "trained_model.pkl"

model = None
feature_names = None 

try:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_FILE_NAME)
    byte_stream = io.BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)
    model = pickle.load(byte_stream)
    print(f"Model loaded successfully from gs://{BUCKET_NAME}/{MODEL_FILE_NAME}")

    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_.tolist()
        print(f"Feature names retrieved from model: {feature_names}")
    elif hasattr(model, 'feature_name'): 
        feature_names = model.feature_name
        print(f"Feature names retrieved from model: {feature_names}")
    else:
        print("Could not automatically determine feature names from the model.")
        feature_names = None 

except Exception as e:
    print(f"Error loading model from GCS: {e}")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    try:
        file = request.files['file']
        input_df = pd.read_csv(file)

        if "transaction_id" not in input_df.columns:
            return jsonify({"error": "CSV must contain a 'transaction_id' column"})

        input_df["timestamp"] = pd.to_datetime(input_df["timestamp"], errors='coerce')
        input_df["transaction_id"] = input_df["transaction_id"].astype(str).str.lower()

        try:
            if model is not None and feature_names is not None:
    
                if all(feature in input_df.columns for feature in feature_names):
                    features = input_df[feature_names].copy()
                    predictions = model.predict(features)
                    input_df['prediction'] = predictions
                else:
                    missing_features = [f for f in feature_names if f not in input_df.columns]
                    return jsonify({"error": f"Missing required column(s) for prediction: {missing_features}"})
            elif model is not None and feature_names is None:
                input_df['prediction'] = 'Feature names could not be determined from the model.'
            else:
                input_df['prediction'] = 'Model not loaded'

        except KeyError as e:
            return jsonify({"error": f"Missing column(s) in the uploaded CSV: {e}"})
        except Exception as e:
            return jsonify({"error": f"Error during feature engineering or prediction: {e}"})

        output = io.BytesIO()
        input_df.to_csv(output, index=False)
        output.seek(0)

        return send_file(output, mimetype='text/csv', as_attachment=True, download_name="Predicted_Data.csv")

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    return jsonify({"message": "This route can be adapted to show recent predictions."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
