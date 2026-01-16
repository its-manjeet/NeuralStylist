from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from io import BytesIO
from PIL import Image

app = FastAPI()

# 1. LOAD THE BRAIN
print("🧠 Loading Neural Stylist Brain...")
model = keras.models.load_model("fashion_brain.h5")

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 2. IMAGE PROCESSING
def process_image(image_data):
    image = Image.open(BytesIO(image_data))
    image = image.convert('L') 
    image = image.resize((28, 28))
    img_array = np.array(image)
    img_inverted = cv2.bitwise_not(img_array)
    img_final = img_inverted / 255.0
    return img_final.reshape(1, 28, 28)

# 3. API ROUTE
@app.post("/predict")
async def predict_fashion(file: UploadFile = File(...)):
    image_data = await file.read()
    processed_img = process_image(image_data)
    
    prediction = model.predict(processed_img)
    predicted_index = np.argmax(prediction)
    item_name = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # STYLING LOGIC
    stylist_comment = "Classic item."
    vibe_check = "Neutral"

    if item_name == 'Sneaker':
        stylist_comment = "DETECTED: High-Value Footwear. Pair with oversized cargos."
        vibe_check = "Hip Hop / Streetwear"
    elif item_name == 'T-shirt/top':
        stylist_comment = "DETECTED: Upper Body Basic. Layer with Jacket."
        vibe_check = "Casual / Drill"
    elif item_name == 'Bag':
        stylist_comment = "DETECTED: Accessory. Match color with footwear."
        vibe_check = "Utility"

    return {
        "detected_item": item_name,
        "confidence_score": f"{confidence:.2f}%",
        "vibe": vibe_check,
        "neural_stylist_advice": stylist_comment
    }