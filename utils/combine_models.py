from ultralytics import YOLO
import os, random, shutil

old_model = YOLO('/Users/ryanabbas/Desktop/work/StockMarket/runs/content/StockMarket/runs/detect2/new_model12/weights/best.pt')
yellow_model = YOLO('/Users/ryanabbas/Desktop/work/StockMarket/runs/detect/final_model/weights/best.pt')

SRC = "all_images"
IMG_OUT = {"train": "images/train", "val": "images/val"}
LBL_OUT = {"train": "all_txt/train", "val": "all_txt/val"}

for p in IMG_OUT.values(): os.makedirs(p, exist_ok=True)
for p in LBL_OUT.values(): os.makedirs(p, exist_ok=True)

imgs = [f for f in os.listdir(SRC) if f.endswith(".png")]
random.shuffle(imgs)

split_idx = int(0.7 * len(imgs))
splits = {
    "train": imgs[:split_idx],
    "val": imgs[split_idx:]
}

for split, files in splits.items():
    for img_file in files:
        src_img = os.path.join(SRC, img_file)
        dst_img = os.path.join(IMG_OUT[split], img_file)
        shutil.copy(src_img, dst_img)

        txt_path = os.path.join(LBL_OUT[split], img_file.replace(".png", ".txt"))
        labels = []

        # OLD MODEL (classes 0–4)
        r1 = old_model(dst_img, conf=0.3)[0]
        if r1.boxes:
            for b in r1.boxes:
                cls = int(b.cls[0])
                x,y,w,h = b.xywhn[0].tolist()
                labels.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        # YELLOW MODEL → FORCE class 5
        r2 = yellow_model(dst_img, conf=0.5)[0]
        if r2.boxes:
            for b in r2.boxes:
                x,y,w,h = b.xywhn[0].tolist()
                labels.append(f"5 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        with open(txt_path, "w") as f:
            f.writelines(labels)

        print(f"{split}: {img_file} → {len(labels)} labels")

print("✅ 70/30 split + combined pseudo-labels done")
