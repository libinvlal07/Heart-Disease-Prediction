from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(
    open("models/heart_model1.pkl", "rb")
)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]

    final_features = [np.array(features)]

    prediction = model.predict(final_features)

    output = prediction[0]

    if output == 1:
        result = "Person Has Heart Disease"

    else:
        result = "Person Does Not Have Heart Disease"

    return render_template(
        "index.html",
        prediction_text=result
    )


if __name__ == "__main__":
    app.run(debug=True)