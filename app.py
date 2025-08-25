import mlflow
import mlflow.sklearn
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model (adjust run_id if needed)
model_uri = "mlruns/950757331535566448/4f8a6084ae114463aca3a7a6c1ce2f7c/artifacts/model"
model = mlflow.sklearn.load_model(model_uri)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Ensure input is a dataframe
        df = pd.DataFrame([data])

        # Prediction
        prediction = model.predict(df)[0]

        # Probability (for binary classification, take probability of class 1)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0][1]
        else:
            proba = None

        return jsonify({
            "prediction": int(prediction),
            "probability": round(float(proba), 4) if proba is not None else "not available"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
