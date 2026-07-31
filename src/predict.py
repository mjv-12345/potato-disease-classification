import tensorflow as tf 
from tensorflow import keras
import numpy as np 

#Step 2 Constants
MODEL_PATH = "models/best_model.keras"

IMAGE_SIZE =(256,256)
CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]
#Step 3 Load the Trained Model
model = keras.models.load_model(MODEL_PATH)

#Step 4 Load Image
def load_image(image_path):
    image = keras.utils.load_img(
        image_path,
        target_size = IMAGE_SIZE
    )
    image = keras.utils.img_to_array(image)
    image = np.expand_dims(image, axis =0)
    return image
#Step 5 Predict Disease
def predict(image_path):
    image = load_image(image_path)
    prediction = model.predict(image)
    predict_class = np.argmax(prediction)
    confidence = np.max(prediction)
    disease = CLASS_NAMES[predict_class]
    return disease, confidence
def main():
    image_path = input("Enter image path:")
    disease, confidence = predict(image_path)
    print(f"\nPrediction:{disease}")
    print(f"Confidence:{confidence:.2%}")
if __name__=="__main__":
    main()