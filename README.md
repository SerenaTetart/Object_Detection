# Object_Detection

## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Project 1: Object Detection using Efficientdet](#project-1---object-detection-using-efficientdet)
   * [Training the model](#training-the-model)
   * [Testing locally the model](#testing-locally-the-model)

## General info
In this repository you will learn the basics for detecting objects and keypoints on images using models such as: EfficientDet, Mobilenet-SSD or even YoloV4...

## Requirements
For the first project you only need a Google account with <a href=https://colab.research.google.com/>Colab</a> and <a href=https://drive.google.com/drive/my-drive>Drive</a>. *(I am using Colab pro for the training)*

But if you want to train locally you need to install manually Tensorflow Object Detection, you'll find a good tutorial on this <a href=https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html>link</a>

For the inference you only need Tensorflow and the files: *saved_model.pb* and *label_map.pbtxt*.

## Project 1 - Object Detection using EfficientDet

### Training the model:

1. First open the file ObjectDetection.ipynb of this repository in Collab.
2. Then you need images and annotations in Pascal VOC format (xml files). *(You can use <a href=https://github.com/Serenalyw/FastAnnotation>FastAnnotations</a>, a framework that I made :smile:)*
3. Once you have them simply put them in a zip file named data.zip, don't bother making a train/test or annotation folder everything will be handled automatically to make the process easier.
4. And now you can upload data.zip to your Drive.
5. Finally just run the code, it will train an EfficientDet0 model on the data you sent to Drive.

If you want to change the model to let's say EfficientDet5 or Mobilenet-SSD you need to download the model from <a href=https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md>Tensorflow Object Detection Zoo</a>, for instance the changes needed for EfficientDet5 will be:

```
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz
!tar -xzvf efficientdet_d0_coco17_tpu-32.tar.gz
```

to

```
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d5_coco17_tpu-32.tar.gz
!tar -xzvf efficientdet_d5_coco17_tpu-32.tar.gz
```

and

``` python
ModelName = 'efficientdet_d0_coco17_tpu-32'
```

to

``` python
ModelName = 'efficientdet_d5_coco17_tpu-32'
```

After the training this is the result that I get for the dataset <a href='https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset'>Stanford Dogs</a> *(with 9 classes only)*:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/168272854-8bf7bef0-d01c-4ad6-a4af-ce6115954451.PNG">
</p>

And these are some test made with the new model trained:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/168273236-32152f5e-cbaa-438b-ae8b-32c033e7f269.PNG">
<img src="https://user-images.githubusercontent.com/65224852/168273292-b2063602-d790-492a-9493-1889c1e412e5.png">
</p>

### Testing locally the model:
