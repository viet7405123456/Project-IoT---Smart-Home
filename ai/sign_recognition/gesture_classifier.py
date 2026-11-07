"""Rule-based gesture classifier: counts extended fingers from MediaPipe
hand landmarks. No training data required.
"""

# MediaPipe landmark indices: https://google.github.io/mediapipe/solutions/hands.html
THUMB_TIP, THUMB_IP = 4, 3
FINGER_TIPS = [8, 12, 16, 20]   # index, middle, ring, pinky
FINGER_PIPS = [6, 10, 14, 18]
WRIST = 0


def _is_finger_extended(landmarks, tip_idx, pip_idx):
    # A finger is extended if its tip is clearly above (smaller y) its PIP joint.
    return landmarks[tip_idx].y < landmarks[pip_idx].y - 0.02


def _is_thumb_extended(landmarks, handedness_label):
    # Thumb moves sideways rather than up/down, so compare x instead of y.
    tip_x = landmarks[THUMB_TIP].x
    ip_x = landmarks[THUMB_IP].x
    if handedness_label == "Right":
        return tip_x < ip_x - 0.02
    return tip_x > ip_x + 0.02


def count_extended_fingers(landmarks, handedness_label="Right"):
    """landmarks: list of 21 MediaPipe NormalizedLandmark objects."""
    count = 0
    if _is_thumb_extended(landmarks, handedness_label):
        count += 1
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        if _is_finger_extended(landmarks, tip_idx, pip_idx):
            count += 1
    return count


def classify(landmarks, handedness_label, gesture_to_command):
    """Returns the mapped command name, or None if the finger count has no mapping."""
    finger_count = count_extended_fingers(landmarks, handedness_label)
    return gesture_to_command.get(finger_count), finger_count
