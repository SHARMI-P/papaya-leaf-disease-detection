import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# -----------------------------
# 1. Load the trained model
# -----------------------------
model = tf.keras.models.load_model("papaya_leaf_model.h5")

print("Model loaded successfully!")

# -----------------------------
# 2. Class names (IMPORTANT)
# Must be same order as training
# -----------------------------
class_names = [
    'Bacterial_Blight',
    'Carica_Insect_Hole',
    'Curled_Yellow_Spot',
    'Yellow_Necrotic_Spots_Holes',
    'healthy_leaf',
    'pathogen_symptoms'
]

# -----------------------------
# 3. Function to predict image
# -----------------------------
def predict_image(img_path):
    
    img = image.load_img(img_path, target_size=(224, 224))  # use same size as training
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    print("\nPrediction:", class_names[predicted_class])
    print("Confidence: {:.2f}%".format(confidence))


# -----------------------------
# 4. Take image path from user
# -----------------------------
img_path = input("Enter full image path: ")

if os.path.exists(img_path):
    predict_image(img_path)
else:
    print("Image path not found!")
