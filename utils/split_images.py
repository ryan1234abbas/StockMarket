import os, random, shutil, glob

img_folder = "images/"
train_folder = "images/train/"
val_folder = "images/val/"

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

images = glob.glob(os.path.join(img_folder, "*.png"))
random.shuffle(images)

split_idx = int(0.7 * len(images))
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

for f in train_imgs:
    shutil.move(f, train_folder)
for f in val_imgs:
    shutil.move(f, val_folder)
