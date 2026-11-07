"""ASL letter classifier trained on MediaPipe hand landmarks (MLP, trained
on Kaggle — see ../training/asl_landmark/).

Loads raw weights from a .npz file and runs the MLP forward pass by hand in
numpy, rather than unpickling the scikit-learn estimator directly. This is
deliberate: the model was trained on Kaggle with numpy>=2, but this project's
mediapipe==0.10.21 pin requires numpy<2 (see requirements.txt — the same
pin finger-count/SigLIP recognition needs, since mediapipe>=0.10.30 removed
the `solutions.hands` API this project relies on). Unpickling a scikit-learn
model saved under one major numpy version and loaded under a different one
raises `ValueError: ... MT19937 is not a known BitGenerator`. Plain numpy
arrays in an .npz file have no such version coupling. The forward pass below
was verified to match the original model's predict_proba() exactly (max abs
diff 0.0) on real sample images before switching to it.
"""

import numpy as np

_weights = None
_classes = None

CANONICAL_HANDEDNESS = "Right"


def _load_weights(weights_path):
    global _weights, _classes
    if _weights is None:
        print(f"Loading landmark classifier weights from '{weights_path}'...")
        data = np.load(weights_path, allow_pickle=True)
        _weights = {
            "W0": data["W0"], "b0": data["b0"],
            "W1": data["W1"], "b1": data["b1"],
            "W2": data["W2"], "b2": data["b2"],
        }
        _classes = data["classes"]
    return _weights, _classes


def normalize_landmarks(landmarks, handedness_label):
    """landmarks: 21 MediaPipe landmark objects (.x, .y, .z).
    MUST stay in sync with ../training/asl_landmark/train_on_kaggle.py's
    normalize_landmarks — this is the exact preprocessing the model was
    trained on (mirror left hand to canonical right-hand orientation,
    translate to wrist origin, scale by hand size).
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    if handedness_label != CANONICAL_HANDEDNESS:
        pts[:, 0] = -pts[:, 0]
    pts -= pts[0]
    scale_ref = np.linalg.norm(pts[9])
    if scale_ref < 1e-6:
        scale_ref = 1.0
    pts /= scale_ref
    return pts.flatten()


def _softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _forward(feat, weights):
    """feat: (1, 63) float array. Mirrors sklearn MLPClassifier's forward
    pass: relu hidden layers, softmax output (verified to match exactly)."""
    h = np.maximum(0, feat @ weights["W0"] + weights["b0"])
    h = np.maximum(0, h @ weights["W1"] + weights["b1"])
    logits = h @ weights["W2"] + weights["b2"]
    return _softmax(logits)


def classify(landmarks, handedness_label, weights_path):
    """Returns (letter, confidence), e.g. ('L', 0.97)."""
    weights, classes = _load_weights(weights_path)

    feat = normalize_landmarks(landmarks, handedness_label).astype(np.float64).reshape(1, -1)
    probs = _forward(feat, weights)[0]
    best_idx = int(np.argmax(probs))
    confidence = float(probs[best_idx])
    letter = str(classes[best_idx])
    return letter, confidence
