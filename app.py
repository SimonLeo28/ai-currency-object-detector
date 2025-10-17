# # File: app.py
# import streamlit as st
# import cv2
# import numpy as np
# import pyttsx3
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# import os
# import glob
# from PIL import Image
#
# # ==============================================================================
# # 1. PAGE CONFIG AND STYLING
# # ==============================================================================
# st.set_page_config(
#     page_title="Vision AI Detector",
#     page_icon="👁️",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
# # --- Custom CSS for a professional look ---
# css_string = """
# /* General App Styling */
# [data-testid="stAppViewContainer"] {
#     background-color: #1a1a1a;
#     background-image: none;
# }
# [data-testid="stHeader"] {
#     background-color: rgba(0,0,0,0);
# }
# [data-testid="stSidebar"] {
#     background-color: #262730;
# }
# h1, h2, h3 {
#     color: #FAFAFA;
# }
# .st-emotion-cache-16txtl3 {
#     font-family: 'Helvetica', sans-serif;
#     color: #D3D3D3;
# }
# /* Button Styling */
# .stButton>button {
#     color: #FFFFFF;
#     background-color: #4F8BF9;
#     border: none;
#     border-radius: 12px;
#     padding: 12px 30px;
#     font-size: 16px;
#     font-weight: bold;
#     transition: all 0.3s ease;
# }
# .stButton>button:hover {
#     background-color: #3A6DC2;
#     transform: scale(1.05);
#     box-shadow: 0px 4px 20px rgba(79, 139, 249, 0.3);
# }
# /* Container Styling */
# [data-testid="stVerticalBlockBorderWrapper"] {
#     background-color: #262730;
#     border-radius: 12px;
#     padding: 20px;
# }
# """
# st.markdown(f"<style>{css_string}</style>", unsafe_allow_html=True)
#
#
# # ==============================================================================
# # 2. MODEL AND FEATURE LOADING (WRAPPED IN CACHE)
# # ==============================================================================
# @st.cache_resource
# def load_resources():
#     print("Loading resources...")
#     # Initialize TTS Engine
#     engine = pyttsx3.init()
#
#     # Load Fallback Model
#     model_fallback = MobileNetV2(weights='imagenet')
#
#     # Initialize ORB Detector
#     orb = cv2.ORB_create(nfeatures=2000)
#
#     # Load Currency Features
#     folder_to_label_map = {
#         "ten_new": "10 Rupee (New)", "ten_old": "10 Rupee (Old)",
#         "twenty_new": "20 Rupee (New)", "twenty_old": "20 Rupee (Old)",
#         "fifty_new": "50 Rupee (New)", "fifty_old": "50 Rupee (Old)",
#         "hundred_new": "100 Rupee (New)", "hundred_old": "100 Rupee (Old)",
#         "two_hundred": "200 Rupee", "five_hundred": "500 Rupee",
#         "two_thousand": "2000 Rupee"
#     }
#     currency_features = {}
#     for folder_name in folder_to_label_map.keys():
#         currency_features[folder_name] = []
#         image_paths = glob.glob(os.path.join(folder_name, '*.jpg'))
#         image_paths.extend(glob.glob(os.path.join(folder_name, '*.jpeg')))
#         image_paths.extend(glob.glob(os.path.join(folder_name, '*.png')))
#         for path in image_paths:
#             template = cv2.imread(path, 0)
#             if template is not None:
#                 kp, des = orb.detectAndCompute(template, None)
#                 if des is not None:
#                     currency_features[folder_name].append({"kp": kp, "des": des, "img": template})
#
#     print("Resources loaded successfully.")
#     return engine, model_fallback, orb, currency_features, folder_to_label_map
#
#
# # ==============================================================================
# # 3. DETECTION FUNCTIONS (Unchanged from your script)
# # ==============================================================================
# def detect_currency_with_features(frame, orb, currency_features):
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
#
#     if des_frame is None: return None, None
#
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING)
#     best_match_denom_folder = None
#     max_good_matches = 0
#     best_match_info = None
#
#     for denom_folder, features_list in currency_features.items():
#         for features in features_list:
#             if features["des"] is None or len(features["des"]) < 2: continue
#             matches = bf.knnMatch(features["des"], des_frame, k=2)
#             good_matches = []
#             if matches and len(matches) > 1 and len(matches[0]) == 2:
#                 for m, n in matches:
#                     if m.distance < 0.75 * n.distance:
#                         good_matches.append(m)
#             if len(good_matches) > max_good_matches:
#                 max_good_matches = len(good_matches)
#                 best_match_denom_folder = denom_folder
#                 best_match_info = {
#                     "kp_template": features["kp"], "kp_frame": kp_frame,
#                     "good_matches": good_matches, "template_img": features["img"]
#                 }
#
#     debug_image = None
#     if best_match_info and max_good_matches > 5:
#         debug_image = cv2.drawMatches(best_match_info["template_img"], best_match_info["kp_template"],
#                                       gray_frame, best_match_info["kp_frame"],
#                                       best_match_info["good_matches"], None,
#                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#
#     if max_good_matches > 15:
#         return best_match_denom_folder, debug_image
#
#     return None, debug_image
#
#
# def detect_object(frame, model_fallback):
#     img = cv2.resize(frame, (224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     preds = model_fallback.predict(x, verbose=0)
#     label = decode_predictions(preds, top=1)[0][0][1]
#     return label.replace("_", " ")
#
#
# # ==============================================================================
# # 4. STREAMLIT APP UI AND LOGIC
# # ==============================================================================
# # --- SIDEBAR ---
# with st.sidebar:
#     st.title("About the Vision AI Detector")
#     st.info(
#         "This application uses computer vision to identify Indian currency notes "
#         "and other common objects in real-time. \n\n"
#         "**Technology Used:**\n"
#         "- **Currency Detection:** OpenCV ORB Feature Matching\n"
#         "- **Object Detection:** TensorFlow & Keras (MobileNetV2)\n"
#         "- **GUI:** Streamlit\n"
#         "- **Text-to-Speech:** pyttsx3"
#     )
#     st.write("---")
#     st.warning("This app is for educational purposes. Always verify currency manually.")
#
# # --- MAIN PAGE ---
# st.title("👁️ Vision AI Detector")
# st.write("A live camera feed will open below. Position your object and click **'Take photo'** to scan it.")
# st.divider()
#
# # Load all our models and data once
# engine, model_fallback, orb, currency_features, folder_to_label_map = load_resources()
#
# # --- Use st.camera_input for a live feed ---
# img_file_buffer = st.camera_input("Click **'Take photo'** to capture an image from your webcam",
#                                   label_visibility="collapsed")
#
# if img_file_buffer is not None:
#     # Convert the image buffer to a CV2 image
#     bytes_data = img_file_buffer.getvalue()
#     frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
#
#     with st.spinner('Analyzing Image...'):
#         result_folder, debug_frame = detect_currency_with_features(frame, orb, currency_features)
#
#         final_text_result = ""
#
#         if result_folder:
#             friendly_label = folder_to_label_map.get(result_folder, "Unknown Currency")
#             final_text_result = f"I think this is a {friendly_label} note."
#         else:
#             obj_label = detect_object(frame, model_fallback)
#             final_text_result = f"This looks like a {obj_label}."
#
#         st.divider()
#         st.subheader("Analysis Results")
#
#         # Display the result in a container
#         with st.container(border=True):
#             st.success(f"**{final_text_result}**")
#
#         # --- AUTOMATIC TEXT-TO-SPEECH ---
#         # The result is spoken automatically here
#         try:
#             engine.say(final_text_result)
#             engine.runAndWait()
#         except Exception as e:
#             st.error(f"Text-to-speech failed: {e}")
#
#         # Display the scanned image and debug view
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Image Scanned", use_column_width=True)
#         with col2:
#             if debug_frame is not None:
#                 st.image(debug_frame, caption="Feature Matching Debug View", use_column_width=True)
#             else:
#                 st.write("No distinct features were matched for currency detection.")

# File: app.py
import streamlit as st
import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os
import glob
from PIL import Image

# ==============================================================================
# 1. PAGE CONFIG AND STYLING
# ==============================================================================
st.set_page_config(
    page_title="Dhristi",
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


# ==============================================================================
# 2. MODEL AND FEATURE LOADING (WRAPPED IN CACHE)
# ==============================================================================
@st.cache_resource
def load_resources():
    print("Loading resources...")
    # Initialize TTS Engine
    engine = pyttsx3.init()

    # Load Fallback Model
    model_fallback = MobileNetV2(weights='imagenet')

    # Initialize ORB Detector
    orb = cv2.ORB_create(nfeatures=2000)

    # Load Currency Features
    folder_to_label_map = {
        "ten_new": "10 Rupee (New)", "ten_old": "10 Rupee (Old)",
        "twenty_new": "20 Rupee (New)", "twenty_old": "20 Rupee (Old)",
        "fifty_new": "50 Rupee (New)", "fifty_old": "50 Rupee (Old)",
        "hundred_new": "100 Rupee (New)", "hundred_old": "100 Rupee (Old)",
        "two_hundred": "200 Rupee", "five_hundred": "500 Rupee",
        "two_thousand": "2000 Rupee"
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
    return engine, model_fallback, orb, currency_features, folder_to_label_map


# ==============================================================================
# 3. DETECTION FUNCTIONS (Unchanged from your script)
# ==============================================================================
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


# ==============================================================================
# 4. STREAMLIT APP UI AND LOGIC
# ==============================================================================
# --- SIDEBAR ---
with st.sidebar:
    st.title("About the Vision AI Detector")
    st.info(
        "This application uses computer vision to identify Indian currency notes "
        "and other common objects in real-time. \n\n"
        "**Technology Used:**\n"
        "- **Currency Detection:** OpenCV ORB Feature Matching\n"
        "- **Object Detection:** TensorFlow & Keras (MobileNetV2)\n"
        "- **GUI:** Streamlit\n"
        "- **Text-to-Speech:** pyttsx3"
    )
    st.write("---")
    st.warning("This app is for educational purposes. Always verify currency manually.")

# --- MAIN PAGE ---
st.title("👁️ Vision AI Detector")
st.write("A live camera feed will open below. Position your object and click **'Take photo'** to scan it.")
st.divider()

# Load all our models and data once
engine, model_fallback, orb, currency_features, folder_to_label_map = load_resources()

# --- Use st.camera_input for a live feed ---
img_file_buffer = st.camera_input("Click **'Take photo'** to capture an image from your webcam",
                                  label_visibility="collapsed")

if img_file_buffer is not None:
    # Convert the image buffer to a CV2 image
    bytes_data = img_file_buffer.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    with st.spinner('Analyzing Image...'):
        result_folder, debug_frame = detect_currency_with_features(frame, orb, currency_features)

        final_text_result = ""

        if result_folder:
            friendly_label = folder_to_label_map.get(result_folder, "Unknown Currency")
            final_text_result = f"I think this is a {friendly_label} note."
        else:
            obj_label = detect_object(frame, model_fallback)
            final_text_result = f"This looks like a {obj_label}."

        st.divider()
        st.subheader("Analysis Results")

        # --- 1. AUTOMATIC TEXT-TO-SPEECH (Runs ONCE automatically) ---
        try:
            engine.say(final_text_result)
            engine.runAndWait()
        except Exception as e:
            st.error(f"Initial text-to-speech failed: {e}")

        # --- 2. DISPLAY RESULT CONTAINER WITH REPEAT BUTTON ---
        with st.container(border=True):
            # Use columns for a clean layout: 85% for text, 15% for the button
            res_col, btn_col = st.columns([0.85, 0.15])

            with res_col:
                st.success(f"**{final_text_result}**")

            with btn_col:
                # This button allows the user to hear the result again
                if st.button("🔊", help="Click to hear the result again"):
                    try:
                        engine.say(final_text_result)
                        engine.runAndWait()
                    except Exception as e:
                        st.error(f"Text-to-speech replay failed: {e}")

        # Display the scanned image and debug view
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Image Scanned", use_column_width=True)
        with col2:
            if debug_frame is not None:
                st.image(debug_frame, caption="Feature Matching Debug View", use_column_width=True)
            else:
                st.write("No distinct features were matched for currency detection.")