import cv2
import numpy as np
import pyttsx3
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# 1. INITIALIZE TOOLS
# ==============================================================================
# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load pretrained MobileNetV2 model for general object detection
model = MobileNetV2(weights='imagenet')

# Initialize the ORB feature detector
orb = cv2.ORB_create(nfeatures=1000)

# 2. PREPARE CURRENCY TEMPLATES
# ==============================================================================
# Paths to your template images
currency_templates = {
    "10": "currency/10.jpg",
    "20": "currency/20.jpg",
    "50": "currency/50.jpg",
    "100": "currency/100.jpg",
    "200": "currency/200.jpg",
    "500": "currency/500.jpg"
}

# Pre-load templates and compute their features (keypoints and descriptors)
# This is done once at the start for efficiency
currency_features = {}
for denom, path in currency_templates.items():
    template = cv2.imread(path, 0)
    # Check if the image was loaded correctly
    if template is None:
        print(f"Warning: Could not load template image at {path}. Skipping.")
        continue
    kp, des = orb.detectAndCompute(template, None)
    currency_features[denom] = {"kp": kp, "des": des}
print("Currency features loaded successfully.")


# 3. DETECTION FUNCTIONS
# ==============================================================================

def detect_currency_with_features(frame):
    """
    Detects currency using ORB feature matching, which is robust to rotation and scale.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

    # If no features are detected in the camera frame, we can't match anything
    if des_frame is None:
        return None

    # Use a Brute-Force Matcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    best_match_denom = None
    max_good_matches = 0

    # Loop through each of our pre-computed currency features
    for denom, features in currency_features.items():
        if features["des"] is None or len(features["des"]) < 2:
            continue

        # Match the descriptors from the template with the descriptors from the camera frame
        matches = bf.match(features["des"], des_frame)

        # Consider a match "good" if its distance is low (lower is better)
        good_matches = [m for m in matches if m.distance < 70]  # You can tune this distance threshold

        # If this currency has more good matches than any we've seen before,
        # it becomes our new best candidate.
        if len(good_matches) > max_good_matches:
            max_good_matches = len(good_matches)
            best_match_denom = denom

    # Require a minimum number of good matches to be confident about the result
    # This prevents false positives from random matches.
    if max_good_matches > 20:  # You can tune this confidence threshold
        return f"{best_match_denom} rupee note"

    return None


def detect_object(frame):
    """
    Fallback function to detect a general object using MobileNetV2.
    """
    img = cv2.resize(frame, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x, verbose=0)  # verbose=0 hides the prediction progress bar
    label = decode_predictions(preds, top=1)[0][0][1]
    return label.replace("_", " ")  # Replace underscores with spaces for better speech


# 4. HELPER AND MAIN LOOP
# ==============================================================================

def speak(text):
    """
    Converts text to speech.
    """
    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()


# Open the default webcam
cap = cv2.VideoCapture(0)

print("\nCamera is active. Press 's' to scan for an object or 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Display the camera feed in a window
    cv2.imshow("Object Detector - Press 's' to scan, 'q' to quit", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 's' key is pressed, run the detection logic
    if key == ord('s'):
        print("\nScanning...")
        # First, try to detect a currency note using our robust feature matcher
        result = detect_currency_with_features(frame)

        # If a currency was found, announce it
        if result:
            speak(f"I think this is a {result}")
        # Otherwise, fall back to the general object detector
        else:
            print("No currency detected, trying general object detection...")
            obj = detect_object(frame)
            speak(f"This looks like a {obj}")

    # If the 'q' key is pressed, break the loop and exit
    elif key == ord('q'):
        print("Quitting...")
        break

# Clean up and close everything
cap.release()
cv2.destroyAllWindows()