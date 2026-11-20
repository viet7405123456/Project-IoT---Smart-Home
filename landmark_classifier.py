"""ASL letter classifier on MediaPipe hand landmarks (trained on ASL-HG, see
../training/asl_landmark/experiments/).

Loads raw weights from a .npz file and runs the forward pass by hand in
numpy, rather than unpickling a scikit-learn estimator. Pickled estimators
are tied to the numpy/scikit-learn versions they were saved with (loading
across numpy major versions raised `MT19937 is not a known BitGenerator`),
and the app would also need scikit-learn installed. Plain arrays have no such
coupling.

Two model kinds are supported; the kind is detected from the arrays present:
  - MLP (W0,b0,W1,b1,W2,b2): relu, relu, softmax — matches
    sklearn MLPClassifier.predict_proba.
  - Logistic regression (W,b): softmax(x W + b) — matches multinomial
    sklearn LogisticRegression.predict_proba.
Both exports were checked against predict_proba (max abs diff ~1e-7).
"""

import numpy as np

_cache = {}

CANONICAL_HANDEDNESS = "Right"


def _load_weights(weights_path):
    if weights_path not in _cache:
        print(f"Loading landmark classifier weights from '{weights_path}'...")
        data = np.load(weights_path, allow_pickle=True)
        if "W0" in data:
            kind = "mlp"
            weights = {k: data[k] for k in ("W0", "b0", "W1", "b1", "W2", "b2")}
        elif "W" in data:
            kind = "logreg"
            weights = {"W": data["W"], "b": data["b"]}
        else:
            raise ValueError(f"Unknown weights format in {weights_path}: {list(data.keys())}")
        print(f"  model kind: {kind}")
        _cache[weights_path] = (kind, weights, data["classes"])
    return _cache[weights_path]


def normalize_landmarks(landmarks, handedness_label):
    """landmarks: 21 MediaPipe landmark objects (.x, .y, .z).
    MUST stay identical to the preprocessing used for training
    (../training/asl_landmark/experiments/loso_on_kaggle.py): mirror the
    left hand to a right-hand orientation, translate to the wrist, scale by
    the wrist-to-middle-MCP distance.
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


def predict_proba(feat, kind, weights):
    """feat: (n, 63) float array -> (n, n_classes) probabilities."""
    if kind == "mlp":
        h = np.maximum(0, feat @ weights["W0"] + weights["b0"])
        h = np.maximum(0, h @ weights["W1"] + weights["b1"])
        return _softmax(h @ weights["W2"] + weights["b2"])
    return _softmax(feat @ weights["W"] + weights["b"])


def classify(landmarks, handedness_label, weights_path):
    """Returns (letter, confidence), e.g. ('L', 0.97)."""
    kind, weights, classes = _load_weights(weights_path)
    feat = normalize_landmarks(landmarks, handedness_label).astype(np.float64).reshape(1, -1)
    probs = predict_proba(feat, kind, weights)[0]
    best_idx = int(np.argmax(probs))
    return str(classes[best_idx]), float(probs[best_idx])
