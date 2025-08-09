import cv2
import numpy as np
import os
import glob

def load_data(path:str):
    imgs = []
    for img_path in os.listdir(path):
        img = cv2.imread(os.path.join(path,img_path))
        img = cv2.resize(img, (0,0),fx=.2, fy=.2)
        imgs.append(img)
        cv2.imshow("Image",img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return imgs

# feature detection
load_data('Image Stitching/data/raw')