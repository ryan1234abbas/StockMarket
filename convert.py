import os
import xml.etree.ElementTree as ET

'''Converts xml files into txt files for model processing'''

# Paths
folder = '/Users/koshabbas/Desktop/unchecked_stock_imgs'
output_dir = os.path.join(folder, "new_labels")
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(folder):
    if not filename.endswith(".xml") or filename.startswith("~$"):
        continue
    print(f"Parsing {filename}")
    tree = ET.parse(os.path.join(folder, filename))

    tree = ET.parse(os.path.join(folder, filename))
    root = tree.getroot()

    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    yolo_lines = []
    for obj in root.findall("object"):
        class_name = obj.find("name").text
        class_id = 0  # use 0 if you only have 1 class 

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        x_center = (xmin + xmax) / 2 / image_width
        y_center = (ymin + ymax) / 2 / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save YOLO annotation
    base_name = os.path.splitext(filename)[0]
    with open(os.path.join(output_dir, f"{base_name}.txt"), "w") as f:
        f.write("\n".join(yolo_lines))
