import cv2
import numpy as np
import os

def load_data(path:str = 'data/raw'):
    imgs = []
    for img_path in os.listdir(path):
        img = cv2.imread(os.path.join(path,img_path))
        imgs.append(img)
    return imgs

# feature detection