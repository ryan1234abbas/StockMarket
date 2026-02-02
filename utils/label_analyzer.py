import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import argparse

IMG_SIZE = 32
LABELS = ["HH", "HL", "LH", "LL"]
MODEL_PATH = "label_cnn.h5"


def load_multi_source_dataset(folders):
    X, y = [], []
    for base_dir in folders:
        for idx, label in enumerate(LABELS):
            label_dir = os.path.join(base_dir, label)
            if not os.path.exists(label_dir):
                continue
            for file in os.listdir(label_dir):
                path = os.path.join(label_dir, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0
                X.append(img.reshape(IMG_SIZE, IMG_SIZE, 1))
                y.append(idx)
    X = np.array(X)
    y = to_categorical(y, num_classes=len(LABELS))
    return X, y


def build_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(LABELS), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model():
    folders = ["templates", "templates_windows"]
    print("Loading dataset from multiple sources...")
    X, y = load_multi_source_dataset(folders)
    print(f"Dataset loaded: {len(X)} samples")
    model = build_model()
    print("Training model...")
    model.fit(X, y, epochs=25, batch_size=16, validation_split=0.2)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def test_model(image_path, templates_dir_list=["templates", "templates_windows"], model=None,threshold=0.8):
    if not model:
        if not os.path.exists(MODEL_PATH):
            print("No trained model found, training first...")
            model = train_model()
        else:
            model = load_model(MODEL_PATH)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image not found:", image_path)
        return

    # --- Load templates from multiple sources ---
    templates = []
    for base_dir in templates_dir_list:
        for label in LABELS:
            label_dir = os.path.join(base_dir, label)
            if not os.path.exists(label_dir):
                continue
            for file in os.listdir(label_dir):
                path = os.path.join(label_dir, file)
                temp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if temp is not None:
                    templates.append((label, temp))

    # --- Find the rightmost template match ---
    max_x = -1
    rightmost_crop = None
    for label, temp in templates:
        if temp.shape[0] > img.shape[0] or temp.shape[1] > img.shape[1]:
            continue
        res = cv2.matchTemplate(img, temp, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= threshold:
            x_right = max_loc[0] + temp.shape[1]
            if x_right > max_x:
                max_x = x_right
                # crop the template region
                rightmost_crop = img[max_loc[1]:max_loc[1]+temp.shape[0], max_loc[0]:max_loc[0]+temp.shape[1]]

    if rightmost_crop is None:
        print("No label found above threshold.")
        return

    # --- Resize crop and predict ---
    crop_resized = cv2.resize(rightmost_crop, (IMG_SIZE, IMG_SIZE)) / 255.0
    crop_resized = crop_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    preds = model.predict(crop_resized, verbose=0)[0]
    label_idx = np.argmax(preds)
    conf = preds[label_idx]
    #print(f"Rightmost Predicted label: {LABELS[label_idx]}, Confidence: {conf:.2f}")
    return LABELS[label_idx], conf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test label CNN")
    parser.add_argument("--train", action="store_true", help="Train the CNN model")
    parser.add_argument("--test", type=str, help="Path to image for rightmost label prediction")
    args = parser.parse_args()

    if args.train:
        train_model()
    if args.test:
        test_model(args.test)
