import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import os
import glob

# 1. INITIALIZE TOOLS
# ==============================================================================
engine = pyttsx3.init()
model_fallback = MobileNetV2(weights='imagenet')  # Renamed to avoid confusion
orb = cv2.ORB_create(nfeatures=2000)

# 2. DYNAMICALLY LOAD CURRENCY TEMPLATES
# ==============================================================================
print("Loading currency template images...")

# This dictionary maps folder names to the desired spoken output
folder_to_label_map = {
    "ten_new": "10 Rupee (New)",
    "ten_old": "10 Rupee (Old)",
    "twenty_new": "20 Rupee (New)",
    "twenty_old": "20 Rupee (Old)",
    "fifty_new": "50 Rupee (New)",
    "fifty_old": "50 Rupee (Old)",
    "hundred_new": "100 Rupee (New)",
    "hundred_old": "100 Rupee (Old)",
    "two_hundred": "200 Rupee",
    "five_hundred": "500 Rupee",
    "two_thousand": "2000 Rupee"
}

currency_features = {}
# Loop through the folders we expect to find
for folder_name in folder_to_label_map.keys():
    # Initialize a list for this denomination's features
    currency_features[folder_name] = []

    # Find all image files in the corresponding folder
    image_paths = glob.glob(os.path.join(folder_name, '*.jpg'))
    image_paths.extend(glob.glob(os.path.join(folder_name, '*.jpeg')))
    image_paths.extend(glob.glob(os.path.join(folder_name, '*.png')))

    if not image_paths:
        print(f"Warning: No images found in folder '{folder_name}'.")
        continue

    # Process each image in the folder
    for path in image_paths:
        template = cv2.imread(path, 0)
        if template is None:
            print(f"Warning: Could not load template image at {path}. Skipping.")
            continue

        kp, des = orb.detectAndCompute(template, None)
        if des is not None:
            currency_features[folder_name].append({"kp": kp, "des": des, "img": template})

print("Currency features loaded successfully.")


# 3. DETECTION FUNCTIONS
# ==============================================================================
def detect_currency_with_features(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    if des_frame is None:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    best_match_denom_folder = None
    max_good_matches = 0
    best_match_info = None

    for denom_folder, features_list in currency_features.items():
        for features in features_list:
            if features["des"] is None or len(features["des"]) < 2:
                continue

            matches = bf.knnMatch(features["des"], des_frame, k=2)
            good_matches = []
            if matches and len(matches[0]) == 2:
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) > max_good_matches:
                max_good_matches = len(good_matches)
                best_match_denom_folder = denom_folder
                best_match_info = {
                    "kp_template": features["kp"],
                    "kp_frame": kp_frame,
                    "good_matches": good_matches,
                    "template_img": features["img"]
                }

    debug_image = None
    if best_match_info and max_good_matches > 5:
        debug_image = cv2.drawMatches(best_match_info["template_img"], best_match_info["kp_template"],
                                      gray_frame, best_match_info["kp_frame"],
                                      best_match_info["good_matches"], None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if max_good_matches > 15:
        # Convert the folder name (e.g., "fifty_new") to a clean label
        friendly_label = folder_to_label_map.get(best_match_denom_folder, "Unknown Currency")
        return f"{friendly_label} note", debug_image

    return None, debug_image


def detect_object(frame):
    img = cv2.resize(frame, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model_fallback.predict(x, verbose=0)
    label = decode_predictions(preds, top=1)[0][0][1]
    return label.replace("_", " ")


# 4. HELPER AND MAIN LOOP
# ==============================================================================
def speak(text):
    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()


cap = cv2.VideoCapture(0)
print("\nCamera is active. Press 's' to scan for an object or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Object Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        print("\nScanning...")
        result, debug_frame = detect_currency_with_features(frame)

        if debug_frame is not None:
            cv2.imshow("Debug View - Feature Matches", debug_frame)

        if result:
            speak(f"I think this is a {result}")
        else:
            print("No currency detected, trying general object detection...")
            obj = detect_object(frame)
            speak(f"This looks like a {obj}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()