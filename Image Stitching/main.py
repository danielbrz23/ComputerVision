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
def detectAndDescribe(image, method=None):
    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"
    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
    elif method == 'akaze':
        descriptor = cv2.AKAZE_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    return (kps, features)