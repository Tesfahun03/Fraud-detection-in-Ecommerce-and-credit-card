from flask import Flask, request, jsonify
import joblib
import pandas as pd
import shap
import numpy as np
import os
import sys

# Add the path to the sys.path
sys.path.append(os.path.abspath('..'))
app = Flask(__name__)

# Load the trained model
model = joblib.load(
    'C:/Users/Temp/Desktop/KAI-Projects/Fraud-detection-in-Ecommerce-and-credit-card/fraud_api/models/RF.pkl')


@app.route("/")
def home():
    return "Fraud Detection Model API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json["features"]
        prediction = model.predict([input_data])[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.get_json()
        df = pd.DataFrame(data, index=[0])

        explainer = shap.Explainer(model, df)
        shap_values = explainer(df)

        feature_importance = np.abs(shap_values.values).tolist()
        return jsonify({"shap_values": feature_importance})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


data = pd.read_csv("C:/Users/Temp/Desktop/KAI-Projects/Fraud-detection-in-Ecommerce-and-credit-card/data/cleaned_data.csv",
                   parse_dates=["purchase_time", "signup_time"])


@app.route("/summary", methods=["GET"])
def get_summary():
    total_transactions = len(data)
    fraud_cases = data["class"].sum()
    fraud_percentage = round((fraud_cases / total_transactions) * 100, 2)

    summary = {
        "total_transactions": total_transactions,
        "fraud_cases": int(fraud_cases),
        "fraud_percentage": fraud_percentage
    }
    return jsonify(summary)


@app.route("/fraud_trends", methods=["GET"])
def fraud_trends():
    trends = data.groupby(data["purchase_time"].dt.date)[
        "class"].sum().reset_index()
    trends.columns = ["date", "fraud_cases"]
    return jsonify(trends.to_dict(orient="records"))


@app.route("/fraud_by_device_browser", methods=["GET"])
def fraud_by_device_browser():
    device_fraud = data.groupby("device_id")[
        "class"].sum().nlargest(10).to_dict()
    browser_fraud = data.groupby("browser")["class"].sum().to_dict()

    return jsonify({
        "fraud_by_device": device_fraud,
        "fraud_by_browser": browser_fraud
    })


if __name__ == "__main__":
    app.run(debug=True)
