# # file: test_key.py
# from roboflow import Roboflow
#
# try:
#     # --- PASTE YOUR NEWEST PRIVATE API KEY HERE ---
#     api_key = "igL9xw1UsSEdLdOweWiP"
#
#     if api_key == "igL9xw1UsSEdLdOweWiP" or len(api_key) < 20:
#         print("🛑 ERROR: Please paste your actual Roboflow Private API Key into the script.")
#     else:
#         print("Connecting to Roboflow with your API key...")
#         rf = Roboflow(api_key=api_key)
#
#         # This command will fail if the key is invalid
#         workspace = rf.workspace()
#
#         print("\n✅ SUCCESS! Your API key is valid.")
#         print(f"Your default workspace is: {workspace.name}")
#
# except Exception as e:
#     print("\n❌ FAILURE: The API key is invalid or there's a connection issue.")
#     print("-----------------------------------------------------------------")
#     print("Error details:")
#     print(e)
#     print("-----------------------------------------------------------------")
#     print("\nNext Steps:")
#     print("1. Go to your Roboflow Workspace Settings -> Roboflow API.")
#     print("2. Click 'Revoke and Re-generate' to create a brand new key.")
#     print("3. Copy the NEW key and paste it into this script.")



#Roboflow
# import cv2
# import pyttsx3
# from roboflow import Roboflow
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
# import numpy as np
# import os # Import the os module
#
# # 1. INITIALIZE TOOLS
# # ==============================================================================
# engine = pyttsx3.init()
# model_fallback = MobileNetV2(weights='imagenet')
#
# # Initialize Roboflow Model
# rf = Roboflow(api_key="1mBNSuTipo7Q8STJHHFK") # PASTE YOUR PRIVATE API KEY
#
# # ## THE FIX IS ON THIS LINE ##
# # Load the project directly using its full path ID
# project = rf.project("aicurrencydetector-c8y6d/indian-currency-recognition-cplpj")
# model_currency = project.version(1).model
#
#
# # 2. DETECTION FUNCTIONS
# # ==============================================================================
# def detect_currency_with_model(frame):
#     # Define a temporary file path
#     temp_file = "temp_frame.jpg"
#     cv2.imwrite(temp_file, frame)
#
#     # Infer on the image
#     prediction = model_currency.predict(temp_file, confidence=50, overlap=50)
#     results = prediction.json()['predictions']
#
#     # Clean up the temporary file
#     if os.path.exists(temp_file):
#         os.remove(temp_file)
#
#     if results:
#         top_prediction = results[0]
#         denomination = top_prediction['class']
#         return f"{denomination} rupee note"
#     return None
#
# def detect_object(frame):
#     img = cv2.resize(frame, (224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     preds = model_fallback.predict(x, verbose=0)
#     label = decode_predictions(preds, top=1)[0][0][1]
#     return label.replace("_", " ")
#
# # 3. HELPER AND MAIN LOOP
# # ==============================================================================
# def speak(text):
#     print(f"Speaking: {text}")
#     engine.say(text)
#     engine.runAndWait()
#
# cap = cv2.VideoCapture(0)
# print("\nCamera is active. Press 's' to scan for an object or 'q' to quit.")
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     cv2.imshow("Object Detector", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     if key == ord('s'):
#         print("\nScanning...")
#         result = detect_currency_with_model(frame)
#
#         if result:
#             speak(f"I see a {result}")
#         else:
#             print("No currency detected, trying general object detection...")
#             obj = detect_object(frame)
#             speak(f"This looks like a {obj}")
#
#     elif key == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


#first one
# import cv2
# import numpy as np
# import pyttsx3
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
# from tensorflow.keras.preprocessing import image
#
# # 1. INITIALIZE TOOLS
# # ==============================================================================
# # Initialize text-to-speech engine
# engine = pyttsx3.init()
#
# # Load pretrained MobileNetV2 model for general object detection
# model = MobileNetV2(weights='imagenet')
#
# # Initialize the ORB feature detector
# orb = cv2.ORB_create(nfeatures=1000)
#
# # 2. PREPARE CURRENCY TEMPLATES
# # ==============================================================================
# # BUG FIX: Use a list of paths for each key to allow multiple templates per denomination.
# currency_templates = {
#     "10": ["currency/10.jpg", "currency/10.jpeg"],
#     "20": ["currency/20.jpg", "currency/20.jpeg"],
#     "50": ["currency/50.jpg", "currency/50.jpeg"],
#     "100": ["currency/100.jpg", "currency/100.jpeg"],
#     "200": ["currency/200.jpg", "currency/200.jpeg"],
#     "500": ["currency/500.jpg", "currency/500.jpeg"]
# }
#
# # Pre-load templates and compute their features.
# # The features for each denomination will be stored in a list.
# currency_features = {}
# for denom, paths in currency_templates.items():
#     currency_features[denom] = []  # Initialize an empty list for this denomination
#     for path in paths:
#         template = cv2.imread(path, 0)
#         if template is None:
#             print(f"Warning: Could not load template image at {path}. Skipping.")
#             continue
#         # Compute keypoints and descriptors
#         kp, des = orb.detectAndCompute(template, None)
#         # IMPROVEMENT: Only add templates that have features
#         if des is not None:
#             currency_features[denom].append({"kp": kp, "des": des})
#
# print("Currency features loaded successfully.")
# for denom, features_list in currency_features.items():
#     print(f"- Loaded {len(features_list)} templates for {denom} Rupee note.")
#
#
# # 3. DETECTION FUNCTIONS
# # ==============================================================================
#
# def detect_currency_with_features(frame):
#     """
#     Detects currency using ORB feature matching. It now checks against all available
#     templates for each denomination.
#     """
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
#
#     if des_frame is None:
#         return None
#
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     best_match_denom = None
#     max_good_matches = 0
#
#     # Loop through each denomination (e.g., "10", "20", ...)
#     for denom, features_list in currency_features.items():
#         # For each denomination, loop through its different template features
#         for features in features_list:
#             if features["des"] is None or len(features["des"]) < 2:
#                 continue
#
#             matches = bf.match(features["des"], des_frame)
#             good_matches = [m for m in matches if m.distance < 70]
#
#             if len(good_matches) > max_good_matches:
#                 max_good_matches = len(good_matches)
#                 best_match_denom = denom
#
#     if max_good_matches > 25:  # Increased threshold slightly for better confidence
#         return f"{best_match_denom} rupee note"
#
#     return None
#
#
# def detect_object(frame):
#     """
#     Fallback function to detect a general object using MobileNetV2.
#     """
#     img = cv2.resize(frame, (224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     preds = model.predict(x, verbose=0)
#     label = decode_predictions(preds, top=1)[0][0][1]
#     return label.replace("_", " ")
#
#
# # 4. HELPER AND MAIN LOOP
# # ==============================================================================
#
# def speak(text):
#     """
#     Converts text to speech.
#     """
#     print(f"Speaking: {text}")
#     engine.say(text)
#     engine.runAndWait()
#
#
# cap = cv2.VideoCapture(0)
# print("\nCamera is active. Press 's' to scan for an object or 'q' to quit.")
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Error: Could not read frame from camera.")
#         break
#
#     cv2.imshow("Object Detector - Press 's' to scan, 'q' to quit", frame)
#     key = cv2.waitKey(1) & 0xFF
#
#     if key == ord('s'):
#         print("\nScanning...")
#         result = detect_currency_with_features(frame)
#         if result:
#             speak(f"I think this is a {result}")
#         else:
#             print("No currency detected, trying general object detection...")
#             obj = detect_object(frame)
#             speak(f"This looks like a {obj}")
#     elif key == ord('q'):
#         print("Quitting...")
#         break
#
# cap.release()
# cv2.destroyAllWindows()


