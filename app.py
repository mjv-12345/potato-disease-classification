from flask import Flask, render_template, request
import os
from src.predict import predict

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_image():

    file = request.files["image"]

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )
    file.save(image_path)
    disease, confidence = predict(image_path)
    return render_template(
        "index.html",
        disease=disease,
        confidence=f"{confidence:.2%}"
    )
if __name__ == "__main__":
    app.run(debug=True)