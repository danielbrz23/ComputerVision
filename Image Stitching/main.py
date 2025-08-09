import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_data(path:str):
    imgs = []
    greys = []
    for img_path in sorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path,img_path))
        img = cv2.resize(img, (0,0),fx=.2, fy=.2)
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        greys.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    return imgs, greys

# feature detection
load_data('Image Stitching/data/raw')