# File: app.py
import streamlit as st
import cv2
import numpy as np
# pyttsx3 is removed as it cannot work on the server
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os
import glob
from PIL import Image
import base64

# 1. PAGE CONFIG AND STYLING
st.set_page_config(
    page_title="Vision AI Detector",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a professional look ---
css_string = """
/* General App Styling */
[data-testid="stAppViewContainer"] {
    background-color: #1a1a1a;
    background-image: none;
}
[data-testid="stHeader"] {
    background-color: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: #262730;
}
h1, h2, h3 {
    color: #FAFAFA;
}
.st-emotion-cache-16txtl3 {
    font-family: 'Helvetica', sans-serif;
    color: #D3D3D3;
}
/* Button Styling */
.stButton>button {
    color: #FFFFFF;
    background-color: #4F8BF9;
    border: none;
    border-radius: 12px;
    padding: 12px 30px;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background-color: #3A6DC2;
    transform: scale(1.05);
    box-shadow: 0px 4px 20px rgba(79, 139, 249, 0.3);
}
/* Container Styling */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #262730;
    border-radius: 12px;
    padding: 20px;
}
"""
st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)


# 2. MODEL AND FEATURE LOADING (WRAPPED IN CACHE)
@st.cache_resource
def load_resources():
    print("Loading resources...")
    # Load Fallback Model
    model_fallback = MobileNetV2(weights='imagenet')

    # Initialize ORB Detector
    orb = cv2.ORB_create(nfeatures=2000)

    # --- ENHANCEMENT: Map contains separate 'display' and 'speak' text ---
    folder_to_label_map = {
        "ten_new": {"display": "10 Rupee (New)", "speak": "10 Rupees"},
        "ten_old": {"display": "10 Rupee (Old)", "speak": "10 Rupees"},
        "twenty_new": {"display": "20 Rupee (New)", "speak": "20 Rupees"},
        "twenty_old": {"display": "20 Rupee (Old)", "speak": "20 Rupees"},
        "fifty_new": {"display": "50 Rupee (New)", "speak": "50 Rupees"},
        "fifty_old": {"display": "50 Rupee (Old)", "speak": "50 Rupees"},
        "hundred_new": {"display": "100 Rupee (New)", "speak": "100 Rupees"},
        "hundred_old": {"display": "100 Rupee (Old)", "speak": "100 Rupees"},
        "two_hundred": {"display": "200 Rupee", "speak": "200 Rupees"},
        "five_hundred": {"display": "500 Rupee", "speak": "500 Rupees"},
        "two_thousand": {"display": "2000 Rupee", "speak": "2000 Rupees"}
    }
    currency_features = {}
    for folder_name in folder_to_label_map.keys():
        currency_features[folder_name] = []
        image_paths = glob.glob(os.path.join(folder_name, '*.jpg'))
        image_paths.extend(glob.glob(os.path.join(folder_name, '*.jpeg')))
        image_paths.extend(glob.glob(os.path.join(folder_name, '*.png')))
        for path in image_paths:
            template = cv2.imread(path, 0)
            if template is not None:
                kp, des = orb.detectAndCompute(template, None)
                if des is not None:
                    currency_features[folder_name].append({"kp": kp, "des": des, "img": template})

    print("Resources loaded successfully.")
    return model_fallback, orb, currency_features, folder_to_label_map


# 3. DETECTION AND HELPER FUNCTIONS

# --- NEW: BROWSER-BASED TEXT-TO-SPEECH FUNCTION ---
def text_to_speech_browser(text):
    # Use HTML and JavaScript to trigger browser's speech synthesis
    # A unique ID is used to ensure the script reruns on new text
    text_encoded = base64.b64encode(text.encode()).decode()
    html_string = f"""
        <script>
            var u = new SpeechSynthesisUtterance();
            u.text = atob("{text_encoded}");
            u.lang = 'en-US';
            speechSynthesis.speak(u);
        </script>
    """
    st.components.v1.html(html_string, height=0, width=0)


def detect_currency_with_features(frame, orb, currency_features):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    if des_frame is None: return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    best_match_denom_folder = None
    max_good_matches = 0
    best_match_info = None

    for denom_folder, features_list in currency_features.items():
        for features in features_list:
            if features["des"] is None or len(features["des"]) < 2: continue
            matches = bf.knnMatch(features["des"], des_frame, k=2)
            good_matches = []
            if matches and len(matches) > 1 and len(matches[0]) == 2:
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match_denom_folder = denom_folder
                best_match_info = {
                    "kp_template": features["kp"], "kp_frame": kp_frame,
                    "good_matches": good_matches, "template_img": features["img"]
                }

    debug_image = None
    if best_match_info and max_good_matches > 5:
        debug_image = cv2.drawMatches(best_match_info["template_img"], best_match_info["kp_template"],
                                      gray_frame, best_match_info["kp_frame"],
                                      best_match_info["good_matches"], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if max_good_matches > 15:
        return best_match_denom_folder, debug_image
    return None, debug_image


def detect_object(frame, model_fallback):
    img = cv2.resize(frame, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model_fallback.predict(x, verbose=0)
    label = decode_predictions(preds, top=1)[0][0][1]
    return label.replace("_", " ")


# 4. STREAMLIT APP UI AND LOGIC
# --- SIDEBAR ---
with st.sidebar:
    st.title("About Dhristi")
    st.info(
        "This application uses computer vision to identify Indian currency notes "
        "and other common objects in real-time. \n\n"
        "**Technology Used:**\n"
        "- **Currency Detection:** OpenCV ORB Feature Matching\n"
        "- **Object Detection:** TensorFlow & Keras (MobileNetV2)\n"
        "- **GUI:** Streamlit"
    )
    st.write("---")
    st.warning("This app is for educational purposes. Always verify currency manually.")

# --- MAIN PAGE ---
st.title("👁️ Dhristi: Vision AI Detector")
st.write("A live camera feed will open below. Position your object and click **'Take photo'** to scan it.")
st.divider()

# Load all our models and data once
model_fallback, orb, currency_features, folder_to_label_map = load_resources()

# --- Use st.camera_input for a live feed ---
img_file_buffer = st.camera_input("Click **'Take photo'** to capture an image from your webcam",
                                  label_visibility="collapsed")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with st.spinner('Analyzing Image...'):
        result_folder, debug_frame = detect_currency_with_features(frame, orb, currency_features)

        final_text_result = ""
        text_to_speak = ""

        if result_folder:
            label_info = folder_to_label_map.get(result_folder,
                                                 {"display": "Unknown Currency", "speak": "Unknown Currency"})
            final_text_result = f"I think this is a {label_info['display']} note."
            text_to_speak = label_info['speak']
        else:
            obj_label = detect_object(frame, model_fallback)
            final_text_result = f"This looks like a {obj_label}."
            text_to_speak = final_text_result

        st.divider()
        st.subheader("Analysis Results")

        with st.container(border=True):
            st.success(f"**{final_text_result}**")

        # --- AUTOMATIC BROWSER TEXT-TO-SPEECH ---
        text_to_speech_browser(text_to_speak)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Image Scanned", use_column_width=True)
        with col2:
            if debug_frame is not None:
                st.image(debug_frame, caption="Feature Matching Debug View", use_column_width=True)
            else:
                st.write("No distinct features were matched for currency detection.")
