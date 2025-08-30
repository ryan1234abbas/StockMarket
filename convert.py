import os
import xml.etree.ElementTree as ET

# Mapping your labels to YOLO IDs
class_map = {
    "LL": 0,
    "HH": 1,
    "HL": 2,
    "LH": 3
}

folder = "/Users/ryanabbas/Desktop/work/StockMarket/backup_label_imgs2"
output_dir = folder
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(folder):
    if not filename.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(folder, filename))
    root = tree.getroot()

    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        if class_name not in class_map:
            print(f"⚠️ Skipping unknown class: {class_name}")
            continue

        class_id = class_map[class_name]
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # Normalize
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    base_name = os.path.splitext(filename)[0]
    with open(os.path.join(output_dir, f"{base_name}.txt"), "w") as f:
        f.write("\n".join(yolo_lines))

print("✅ Conversion done. YOLO labels saved in:", output_dir)
