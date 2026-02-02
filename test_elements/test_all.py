import os

splits = ['train', 'val']
img_dir = "images"
label_dir = "combined_labels"

for split in splits:
    missing = []
    img_path = os.path.join(img_dir, split)
    lbl_path = os.path.join(label_dir, split)
    for img_file in os.listdir(img_path):
        if img_file.endswith((".png", ".jpg")):
            label_file = os.path.splitext(img_file)[0] + ".txt"
            if not os.path.exists(os.path.join(lbl_path, label_file)):
                missing.append(img_file)
    print(f"{split}: {len(missing)} images missing labels")
    if missing:
        print("Missing:", missing)
