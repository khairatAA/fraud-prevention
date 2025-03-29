from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
# from google.cloud import storage
from datetime import timedelta
import io

app = Flask(__name__, template_folder="templates")

# Google Cloud Storage settings
# BUCKET_NAME = "innovators1"
FILE_NAME = "testtt.csv"

# Function to read CSV from GCS
def read_file(file_name):
    # client = storage.Client()
    # bucket = client.bucket(bucket_name)
    # blob = bucket.blob(file_name)
    # content = blob.download_as_text()
    df = pd.read_csv("testtt.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# Read and preprocess data
df = read_file(FILE_NAME)

df.dropna(inplace=True)

df.drop(columns=["timestamp_3", "timestamp_4"], axis=1, inplace=True, errors="ignore")

# Final predicted DataFrame

# df = df.rename(columns={"model_prediction": "prediction"})
predicted_df = df[["timestamp", "transaction_id", "calling_msisdn", "prediction"]]


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
        
        # input_df = input_df.rename(columns={"model_prediction": "prediction"})

        input_df["timestamp"] = pd.to_datetime(input_df["timestamp"])

        input_df["transaction_id"] = input_df["transaction_id"].astype(str).str.lower()
        predicted_df["transaction_id"] = predicted_df["transaction_id"].astype(str).str.lower()

        merged_df = input_df.merge(predicted_df, on=["transaction_id"], how="left").drop_duplicates(subset=["transaction_id"])

        # Convert DataFrame to CSV for download
        output = io.BytesIO()
        merged_df.to_csv(output, index=False)
        output.seek(0)

        # Store the CSV file in GCS
        # client = storage.Client()
        # bucket = client.bucket(BUCKET_NAME)
        # blob = bucket.blob("Predicted_Cells.csv")  # Storing in "innovators1" bucket
        # blob.upload_from_file(output, content_type="text/csv")
        # output.seek(0)  # Reset buffer position for file download

        return send_file(output, mimetype='text/csv', as_attachment=True, download_name="Predicted_Cells.csv")

    except Exception as e:
        return jsonify({"error": str(e)})
    

@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    """Returns the first 10 predictions for display."""
    try:
        predictions = predicted_df.sort_values(by="timestamp", ascending=False)
        predictions = predictions[predictions["prediction"]==1].head(4)
        predictions["prediction"] = predictions["prediction"].replace(1, "Fraudulent")

        if predictions.empty:
            return jsonify({"error": "No predictions available"})

        return jsonify({"predictions": predictions.to_dict(orient="records")})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
