import cv2
import numpy as np
def preprocess(image_path,image_size,single=False):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size,image_size), interpolation = cv2.INTER_AREA)
    if single == True:
        image = np.expand_dims(image,axis=0)
    return image

def augment_img(image_path,image_size,single=False):
    image = cv2.imread(image_path)
    image = occulude(image)
    image = cv2.resize(image, (image_size,image_size), interpolation = cv2.INTER_AREA)
    if single == True:
        image = np.expand_dims(image,axis=0)
    return image

def occulude(image):
    w = int(image.shape[1])
    h = int(image.shape[0])
    w_p1 = int(w/3.5)
    w_p2 = w-(w_p1)
    h_p1 = int(h/3)
    h_p2 = h-(h_p1)
    x1 = np.random.randint(w_p1,w*3/4) #w
    x2 = np.random.randint(w*1/4,w_p2) #0
    y2 = np.random.randint(h*1/4,h_p2) #0
    y1 = np.random.randint(h_p1,h*3/4) #h
    rand = np.random.randint(4)
    if rand == 0:
        image = cv2.rectangle(image, (x1,y1), (w,h), list(np.random.random(size=3) * 256), -1)
    if rand == 1:
        image = cv2.rectangle(image, (x1,0), (w,y2), list(np.random.random(size=3) * 256), -1)
    if rand == 2:
        image = cv2.rectangle(image, (0,y1), (x2,h), list(np.random.random(size=3) * 256), -1)
    if rand == 3:
        image = cv2.rectangle(image, 1(0,0),(x1,y2), list(np.random.random(size=3) * 256), -1)
    return image
