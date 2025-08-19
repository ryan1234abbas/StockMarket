import cv2
import os
import numpy as np
import platform

def load_templates():
    """
    Load all template images from subfolders in 'templates/'.
    Returns a dict: {label: [list of images]}
    """
    
    if platform.system() == "Darwin":
        base_dir = "templates"
    elif platform.system() == "Windows":
        base_dir = "templates_windows"
    else:
        return "Unknown OS"
    
    templates = {}
    for label in os.listdir(base_dir):
        label_dir = os.path.join(base_dir, label)
        if os.path.isdir(label_dir):
            templates[label] = []
            for file in os.listdir(label_dir):
                path = os.path.join(label_dir, file)
                img = cv2.imread(path)
                if img is not None:
                    templates[label].append(img)
    return templates

def get_rightmost_label(img, templates, threshold=0.8):
    """
    Returns the rightmost label found in the image based on the largest x-coordinate,
    along with the confidence score.
    """
    max_x = -1
    rightmost_label = None
    rightmost_conf = 0.0  # Keep track of the confidence of the chosen label

    for label, temp_imgs in templates.items():
        for template in temp_imgs:
            if template.shape[0] > img.shape[0] or template.shape[1] > img.shape[1]:
                continue  # skip templates bigger than image
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val >= threshold:
                x = max_loc[0] + template.shape[1]  # right edge
                if x > max_x:
                    max_x = x
                    rightmost_label = label
                    rightmost_conf = max_val  # save the confidence

    return rightmost_label, rightmost_conf


def main(screenshot_path):
    img = cv2.imread(screenshot_path)
    if img is None:
        print("Error: image not found")
        exit(1)

    templates = load_templates()
    label, confidence = get_rightmost_label(img, templates)
    return label, confidence
