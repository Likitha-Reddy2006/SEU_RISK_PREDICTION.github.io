import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
cols = joblib.load("selected_features.pkl")
label = joblib.load("label.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/result", methods=["POST"])
def result():

    data = [
        float(request.form["radiation"]),
        float(request.form["temperature"]),
        float(request.form["voltage"]),
        float(request.form["device"]),
        float(request.form["memory"]),
        float(request.form["error"]),
        float(request.form["faults"])
    ]

    data = np.array([data])

    # Scale data
    data = scaler.transform(data)

    # Select only the features used by the model
    data = data[:, cols]

    # Predict
    prediction = model.predict(data)[0]

    # Print values in terminal (for debugging)
    print("Prediction:", prediction)
    print("Label Dictionary:", label)

    # Handle both integer and string keys
    if prediction in label:
        risk = label[prediction]
    else:
        risk = label[str(prediction)]

    return render_template("result.html", prediction=risk)


if __name__ == "__main__":
    app.run(debug=True)