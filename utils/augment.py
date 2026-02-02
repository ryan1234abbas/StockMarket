import albumentations as A
import cv2
import os
from glob import glob

# Folder with original images
img_folder = "images/"
images = glob(os.path.join(img_folder, "*.jpg")) + glob(os.path.join(img_folder, "*.png"))

# Subtle augmentations
transform = A.Compose([
    # Slight brightness/contrast
    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.8),
    # Slight color shifts
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.5),
    # Slight blur/sharpen
    A.OneOf([
        A.GaussianBlur(blur_limit=(3,5), p=0.5),
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5)
    ], p=0.3),
])

for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Create 2 augmented versions per image
    for i in range(2):
        augmented = transform(image=img)['image']
        base, ext = os.path.splitext(img_path)
        save_path = f"{base}_aug{i+1}{ext}"
        cv2.imwrite(save_path, augmented)

print("Subtle brightness/contrast + hue/saturation + slight blur augmentation done!")
