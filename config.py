# Set to True to use the laptop's built-in/USB webcam instead of the ESP32-CAM
# stream. Useful for testing before the camera hardware is ready.
USE_WEBCAM = True
WEBCAM_INDEX = 0

# IP/URL of the ESP32-CAM video stream. Only used when USE_WEBCAM is False.
ESP32_CAM_STREAM_URL = "http://192.168.1.50:81/stream"

# Set to True to simulate the relay controller (prints/overlays state instead
# of sending real HTTP requests). Use this until the ESP32 controller board
# is wired up, then switch to False.
USE_MOCK_CONTROLLER = True

# Base URL of the ESP32 controlling relays (light/fan/door).
# Only used when USE_MOCK_CONTROLLER is False.
ESP32_CONTROL_BASE_URL = "http://192.168.1.51"

# Path to the trained model weights (see models/, trained in
# ../training/asl_landmark/ on Kaggle). Stored as plain numpy arrays (.npz),
# not a pickled scikit-learn estimator — see landmark_classifier.py for why.
LANDMARK_MODEL_PATH = "models/asl_landmark_mlp_weights.npz"

# Path to the MediaPipe HandLandmarker model bundle. MUST be the same API
# family used during training (mediapipe.tasks.vision.HandLandmarker, not
# the legacy mp.solutions.hands) — the two produce slightly different
# landmark coordinates for the same image, and using the wrong one at
# inference time measurably hurts accuracy (verified: 1/3 wrong on real
# samples with solutions.hands vs 3/3 correct with HandLandmarker).
HAND_LANDMARKER_MODEL_PATH = "models/hand_landmarker.task"

# Ignore predictions below this confidence. Validation-set accuracy was 99%
# (see ../training/asl_landmark/DATASET_LOG.md), but that was measured on
# dataset images, not webcam frames, so this still guards against low-
# confidence predictions once real webcam domain shift comes into play.
LANDMARK_CONFIDENCE_THRESHOLD = 0.6

# This classifier is lightweight (small MLP on 63 numbers), so it runs every
# frame — this is how many consecutive frames must agree before a command is
# sent, to avoid accidental triggers from a hand passing through the frame.
LANDMARK_STABLE_FRAMES = 8

# Minimum seconds between two commands sent to the same device, to avoid
# relay chatter if the prediction flickers right at the confidence threshold.
COMMAND_COOLDOWN_SECONDS = 1.5

# ASL letter -> command name. Edit this to redefine the sign vocabulary.
LETTER_TO_COMMAND = {
    "L": "light_on",     # Light
    "D": "light_off",    # Dark
    "F": "fan_on",       # Fan
    "S": "fan_off",      # Stop
    "O": "door_unlock",  # Open
    "C": "door_lock",    # Close
}
