import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
import cv2

labels = os.listdir("data/train") #['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

model_path = "/model/modelo_mobilenetv2.h5"

model = tf.keras.models.load_model("model/modelo_mobilenetv2.h5")

cap = cv2.VideoCapture(0)  # 0 webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame 
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #  RGB
    image = cv2.resize(image, (224, 224))  #  224x224
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Normalize
    image = np.expand_dims(image, axis=0)  # batch

    # Prediction
    pred = model.predict(image)
    predicted_class_index = np.argmax(pred)
    predicted_class = labels[predicted_class_index]

    # Show prediction
    cv2.putText(frame, f"Predicción: {predicted_class}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Screen frame
    frame_resized = cv2.resize(frame, (1080, 720))
    cv2.imshow("Clasificación en vivo", frame_resized)


    # q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
