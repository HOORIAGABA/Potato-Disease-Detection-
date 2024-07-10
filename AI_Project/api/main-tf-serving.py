from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
import os

app = FastAPI()

# Allowing CORS for development purposes
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
# model_path = r"C:\Users\dell\OneDrive\Desktop\AI_Project-new\AI_Project\models\model_1.h5"
# model = tf.keras.models.load_model(model_path)

# Set the base directory to the root of your project
base_dir = os.path.dirname(os.path.dirname(__file__))

# Construct the path to the model file relative to the base directory
model_path = os.path.join(base_dir, 'models', 'model_1.h5')

# Print the model path for debugging
print(f"Model path: {model_path}")

# Check if the model file exists
if not os.path.exists(model_path):
    # Print contents of the 'models' directory for debugging
    models_dir = os.path.join(base_dir, 'AI_Project', 'models')
    if os.path.exists(models_dir):
        print("Contents of the 'models' directory:")
        print(os.listdir(models_dir))
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the model
MODEL = tf.keras.models.load_model(model_path)


# Define the class names
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_array = tf.image.resize(image, (256, 256))  # Resize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    
    predicted_class = CLASS_NAMES[predicted_class_index]
    confidence = np.max(predictions[0])

    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
