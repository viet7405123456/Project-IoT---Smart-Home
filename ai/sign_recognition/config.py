# Set to True to use the laptop's built-in/USB webcam instead of the ESP32-CAM
# stream. Useful for testing gesture recognition before the camera hardware
# is ready. Switch back to False once ESP32-CAM is flashed and on the network.
USE_WEBCAM = True
WEBCAM_INDEX = 0

# IP/URL of the ESP32-CAM video stream (see Serial Monitor output after flashing).
# Only used when USE_WEBCAM is False.
ESP32_CAM_STREAM_URL = "http://192.168.1.50:81/stream"

# Set to True to simulate the relay controller (prints/overlays state instead
# of sending real HTTP requests). Use this until the ESP32 controller board
# is wired up, then switch to False.
USE_MOCK_CONTROLLER = True

# Base URL of the ESP32 controlling relays (light/fan/door).
# Only used when USE_MOCK_CONTROLLER is False.
ESP32_CONTROL_BASE_URL = "http://192.168.1.51"

# How many consecutive frames a gesture must be held before a command is sent.
# Prevents accidental triggers from a hand passing through the frame.
GESTURE_STABLE_FRAMES = 8

# Minimum seconds between two commands sent to the same device, to avoid
# relay chatter if the gesture flickers right at the boundary.
COMMAND_COOLDOWN_SECONDS = 1.5

# Finger count -> command name. Edit this to redefine the sign vocabulary.
GESTURE_TO_COMMAND = {
    1: "light_on",
    2: "light_off",
    3: "fan_on",
    4: "fan_off",
    5: "door_unlock",
    0: "door_lock",
}

# Which recognizer to use:
#   "finger_count" -> rule-based, no extra dependencies, fast (default).
#   "asl_alphabet"  -> HuggingFace SigLIP model classifying ASL letters.
#                      Needs `torch` + `transformers` (see requirements-asl.txt).
RECOGNITION_MODE = "asl_alphabet"

# --- ASL alphabet mode settings (only used when RECOGNITION_MODE == "asl_alphabet") ---

ASL_MODEL_NAME = "prithivMLmods/Alphabet-Sign-Language-Detection"

# Run the (slow) SigLIP classification every Nth frame to keep the video feed
# smooth; MediaPipe hand tracking + the on-screen skeleton still update every frame.
ASL_FRAME_SKIP = 5

# Ignore predictions below this confidence to reduce false triggers from
# domain shift (webcam background vs. the model's training images).
ASL_CONFIDENCE_THRESHOLD = 0.6

# Extra margin (fraction of hand bounding box size) added when cropping the
# hand region before feeding it to the classifier.
ASL_CROP_MARGIN_RATIO = 0.3

# How many consecutive confident classifications of the same letter are
# required before sending a command (classifications happen every
# ASL_FRAME_SKIP frames, not every frame, so this can be smaller than
# GESTURE_STABLE_FRAMES).
ASL_STABLE_CLASSIFICATIONS = 3

# ASL letter -> command name.
LETTER_TO_COMMAND = {
    "L": "light_on",   # Light
    "D": "light_off",  # Dark
    "F": "fan_on",     # Fan
    "S": "fan_off",    # Stop
    "O": "door_unlock",  # Open
    "C": "door_lock",    # Close
}
