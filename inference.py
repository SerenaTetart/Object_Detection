import io
import os
import re
import numpy as np
from PIL import Image
import tensorflow as tf

PATH_TO_TEST_IMAGES = './images/test'                    #the directory where your test images are
PATH_TO_SAVED_MODEL = './saved_model'                    #the directory saved_model
PATH_TO_LABEL_MAP = './saved_model/label_map.pbtxt'      #label_map.pbtxt

def load_image_into_numpy_array(path):
  img_data = tf.io.gfile.GFile(path, 'rb').read()
  image = Image.open(io.BytesIO(img_data))
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
  
def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  input_tensor = tf.convert_to_tensor(image)
  input_tensor = input_tensor[tf.newaxis,...]

  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)
  
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
    
  return output_dict
  
listClass = []
with open(PATH_TO_LABEL_MAP) as file:
  txt = file.read()
  indices = [i.start() for i in re.finditer('name:', txt)]
  for indice in indices:
    ind = (indice+7)+txt[indice+7::].find('}')
    listClass.append(txt[indice+7:ind-2])
    
tf.keras.backend.clear_session()
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

for file in os.listdir(PATH_TO_TEST_IMAGES):
  if(os.path.splitext(file)[1] != '.xml'):
    image_np = load_image_into_numpy_array(PATH_TO_TEST_IMAGES+'/'+file)
    output_dict = run_inference_for_single_image(model, image_np)
    for i in range(len(output_dict['detection_boxes'])):
      if(output_dict['detection_scores'][i] > 0.5):
        xmin = output_dict['detection_boxes'][i][1]*image_np.shape[1]
        ymin = output_dict['detection_boxes'][i][0]*image_np.shape[0]
        xmax = output_dict['detection_boxes'][i][3]*image_np.shape[1]
        ymax = output_dict['detection_boxes'][i][2]*image_np.shape[0]
        box = [xmin, ymin, xmax, ymax]
        classe = listClass[output_dict['detection_classes'][i]-1]
        print(box, classe)
tmp = input('Program paused...')
