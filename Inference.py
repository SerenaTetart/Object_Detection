import os
import numpy as np
from PIL import Image
import tensorflow as tf

import cv2

PATH_TO_SAVED_MODEL = 'C:/Users/Seren/Desktop/Projets_Info/FaceKP/saved_model' #the directory saved_model

TARGET_SHAPE = (250, 250)

def preprocess_image(image):
    img = tf.math.divide(image, 255)
    img = tf.image.resize(img, TARGET_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img
    
#I- Load model
tf.keras.backend.clear_session()
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#II- Get Webcam view
cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)
if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
while rval:
    #III- Detect faces and crop them on each frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        coef_x = (w+100)/250
        coef_y = (h+100)/250
    #IV- On each face use algorithm
        img = tf.image.resize(frame[y-50:y+h+50, x-50:x+w+50, 0:3], TARGET_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        img = np.array(img)
        img_tmp = preprocess_image(frame[y-50:y+h+50, x-50:x+w+50, 0:3])
        img_tmp = np.expand_dims(img_tmp, axis=0)
        pred = model(img_tmp)[0]
        for i in range(5):
            cv2.line(frame, (int(pred[i*2]*coef_x)+x-50-5, int(pred[(i*2)+1]*coef_y)+y-50-5), (int(pred[i*2]*coef_x)+x-50+5, int(pred[(i*2)+1]*coef_y)+y-50+5), (255, 0, 0), 2)
            cv2.line(frame, (int(pred[i*2]*coef_x)+x-50-5, int(pred[(i*2)+1]*coef_y)+y-50+5), (int(pred[i*2]*coef_x)+x-50+5, int(pred[(i*2)+1]*coef_y)+y-50-5), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #V- Draw and show the result
    cv2.imshow("preview", frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break