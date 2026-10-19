import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 28
MODEL_PATH = "model/sketch_model.h5"
CLASSES_PATH = "model/classes.txt"

_model = None
_classes = None

def load():
    global _model, _classes
    if _model is None:
        _model = load_model(MODEL_PATH)
        with open(CLASSES_PATH) as f:
            _classes = [l.strip() for l in f.readlines() if l.strip()]
    return _model, _classes

def preprocess(canvas_gray):
    """canvas_gray : np.array uint8 (fond blanc, trait noir)."""
    # On cherche la bounding box du dessin
    _, thresh = cv2.threshold(canvas_gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    crop = thresh[y:y+h, x:x+w]
    # Redimension avec padding pour garder proportions
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    ox, oy = (size - w)//2, (size - h)//2
    square[oy:oy+h, ox:ox+w] = crop
    img = cv2.resize(square, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

def predict(canvas_gray):
    model, classes = load()
    x = preprocess(canvas_gray)
    if x is None:
        return "Vide", 0.0
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])