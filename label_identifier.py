import os
import platform
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import cv2

class PatternClassifier:
    def __init__(self, template_dir_mac="templates", template_dir_win="templates_windows",
                 model_path="pattern_model.h5", img_size=(64,64)):
        # Detect OS and choose template folder
        system = platform.system()
        self.template_dir = template_dir_mac if system == "Darwin" else template_dir_win

        self.model_path = model_path
        self.img_size = img_size
        self.classes = ["HH", "HL", "LH", "LL"]

        # Load existing model if present
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"[INFO] Loaded model from {self.model_path}")
        else:
            self.model = None

    def train(self, epochs=20, batch_size=8):
        """Train CNN with automatic data augmentation"""
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            brightness_range=(0.8,1.2),
            fill_mode='nearest',
            validation_split=0.2
        )

        train_gen = datagen.flow_from_directory(
            self.template_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_gen = datagen.flow_from_directory(
            self.template_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )

        # Build simple CNN
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(4, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_gen, validation_data=val_gen, epochs=epochs)

        model.save(self.model_path)
        print(f"[INFO] Model saved to {self.model_path}")
        self.model = model

    def predict(self, img):
        """Predict pattern of a single cropped image"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        img_resized = cv2.resize(img, self.img_size)
        if len(img_resized.shape) == 2:  # grayscale → 3 channels
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        img_array = img_resized / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        probs = self.model.predict(img_array, verbose=0)[0]
        label = self.classes[np.argmax(probs)]
        confidence = np.max(probs)
        return label, confidence
