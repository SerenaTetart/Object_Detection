import os
import numpy as np
from PIL import Image
import tensorflow as tf

PATH_TO_TEST_IMAGES = 'C:/Users/Seren/Desktop/Projets_Info/Segmentation/test'                           #the directory where your test images are
PATH_TO_SAVED_MODEL = 'C:/Users/Seren/Desktop/Projets_Info/Segmentation/saved_model'                    #the directory saved_model

TARGET_SHAPE = (512, 512)

def preprocess_image(filename):
    """
    Load the specified file as a JPG image, preprocess it and
    resize it to the target shape.
    """
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.math.divide(img, 255)
    img = tf.image.resize(img, TARGET_SHAPE, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img
    
def MakeMask(imgMask, ind):
  imgMask = tf.where(imgMask > ind, 255, 0)
  imgMask = np.array(imgMask)
  imgMask1 = np.zeros((imgMask.shape[0],imgMask.shape[1],3),dtype = np.uint8)
  for i in range(imgMask.shape[0]):
    for j in range(imgMask.shape[1]):
      if imgMask[i][j][0] == 0:
        imgMask1[i][j][0] = imgMask1[i][j][1] = imgMask1[i][j][2] = 0
      else:
        imgMask1[i][j][0] = imgMask1[i][j][1] = imgMask1[i][j][2] = 255
  return imgMask1
    
tf.keras.backend.clear_session()
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

for file in os.listdir(PATH_TO_TEST_IMAGES):
    randomImage = preprocess_image(PATH_TO_TEST_IMAGES+'/'+file)
    randomImage = np.expand_dims(randomImage, axis=0)
    pred = model(randomImage)[0]
    pred1 = MakeMask(pred, 0.5)
    print(pred1)
    img = Image.fromarray(pred1, "RGB")
    img.save(PATH_TO_TEST_IMAGES+'/result.jpg')
    
tmp = input('Program paused...')