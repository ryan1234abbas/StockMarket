import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from replica_screen import ReplicaScreen
import mss
import cv2
import pytesseract
import re
import os
import pyautogui

# execute python3 labelImg.py to run labeling GUI