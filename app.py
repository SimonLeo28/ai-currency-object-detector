import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import pyttsx3

st.title("👁‍🗨 AI Currency & Object Reader for Visually Impaired")

engine = pyttsx3.init()
model = MobileNetV2(weights='imagenet')

def detect_object(img_array):
    img = cv2.resize(img_array, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    label = decode_predictions(preds, top=1)[0][0][1]
    return label

uploaded = st.camera_input("Take a photo")

if uploaded is not None:
    bytes_data = uploaded.getvalue()
    npimg = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    label = detect_object(frame)
    st.success(f"Detected: {label}")
    engine.say(f"This is a {label}")
    engine.runAndWait()
