import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

IMG_SIZE = 28
DATA_DIR = "data"
MODEL_PATH = "model/sketch_model.h5"

def load_data():
    X, y = [], []
    classes = sorted(os.listdir(DATA_DIR))
    print("Classes trouvées :", classes)
    for idx, cls in enumerate(classes):
        folder = os.path.join(DATA_DIR, cls)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.bitwise_not(img)  # fond noir, trait blanc
            img = img / 255.0
            X.append(img)
            y.append(idx)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    return X, y, classes

def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    X, y, classes = load_data()
    print(f"{len(X)} images chargées, {len(classes)} classes.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(len(classes))
    model.fit(X_train, y_train, epochs=15, batch_size=32,
              validation_data=(X_test, y_test))

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)

    with open("model/classes.txt", "w") as f:
        f.write("\n".join(classes))

    print("Modèle enregistré :", MODEL_PATH)