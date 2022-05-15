# Object_Detection

## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Project 1: Object Detection using Efficientdet](#project-1---object-detection-using-efficientdet)
   * [What is EfficientDet ?](#what-is-efficientdet-)
   * [Training the model](#training-the-model)
   * [Testing locally the model](#testing-locally-the-model)
* [Project 2: Segmentation using U-Net](#project-2---segmentation-using-u-net)

## General info
In this repository you will learn the basics for detecting objects, keypoints or even making segmentation on images using models such as: EfficientDet, Mobilenet-SSD, U-Net...

## Requirements
For the first project you only need a Google account with <a href=https://colab.research.google.com/>Colab</a> and <a href=https://drive.google.com/drive/my-drive>Drive</a>. *(I am using Colab pro for the training)*

But if you want to train locally you need to install manually Tensorflow Object Detection, you'll find a good tutorial on this <a href=https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html>link</a>

For the inference you need **inference.py** shared in this repository, the directory **saved_model** and the file **label_map.pbtxt**. *(you can automatically download both by running the code in colab)*

## Project 1 - Object Detection using EfficientDet

### What is EfficientDet ?

EfficientDet is a deep learning model designed for object detection, it achieved state of the art results on COCO dataset, it is both scalable and efficient meaning that it can recognize objects at vastly different scales and need fewer computational performance than the other models.

To understand EfficientDet we need to understand two key improvements made:
1. Bi-directional Feature Pyramid Network (BiFPN)
2. Compound Scaling

#### Feature Pyramid Network:

Recognizing objects at different scales is a fundamental challenge in computer vision.

Different authors have tried to solve this differently, there were three main categories of solutions that existed before the introduction of FPN:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/168466672-11fabea6-c628-4dd8-99c0-52b89fec237c.png">
</p>

But they all had some issues:

* **Featurized image pyramid** is too long to train and is infeasible in terms of memory because you need to train a CNN for every scales of an image.
* **Single feature map** is actually used by Faster RCNN but lose representational capacity for object detection in the first layers with low level features embedding.
* **Pyramidal feature hierarchy** is used by SSD, it avoids using low level features in the first levels of a CNN by directly using the high level feature at the end of a CNN and then adds several new layers but by doing so it misses the opportunity to reuse the earlier layers which are important for detecting small objects.

What **Feature Pyramid Network** does is to combine low-resolution, semantically strong features in the later layers with high-resolution and semantically weak features in the earlier layers via a top-down pathway and lateral connections. Thus, leading to **Multi-scale feature fusion**.

*It is somehow similar to the architecture of U-Net when you think about it.*

#### Bi-directional Feature Pyramid Network (BiFPN):

<p align="center"> <i> Feature network design and evolution of FPN </i>
<img src="https://user-images.githubusercontent.com/65224852/168467759-58196966-3795-4e5a-9dd4-5d9ea02f60ed.png">
</p>

What EfficientDet and BiFPN in particular did is to:

1. Remove all nodes that have only one input edge. The intuition is that if a node has only one input edge with no feature fusion, then it will have less contribution to the feature network.
2. Add an extra edge from the original input to the output node if they are at the same level in order to fuse more features without adding much cost
3. Treat each bidirectional path as one single layer and have multiple of these to enable more high-level feature fusion.

#### Compound Scaling:

### Training the model:

1. First open the file ObjectDetection.ipynb of this repository in Colab.
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

In order to use the model locally there are a few steps:

#### I- Download the trained model by running this code in the notebook in colab:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/168441258-50c0e90a-cefb-4d54-94ba-9248d61d4b47.PNG">
</p>

#### II - Copy and paste where you want the files:

<p align="center"> <i>this is where your files are downloaded</i>
<img src="https://user-images.githubusercontent.com/65224852/168441338-e755f771-4e68-49e6-ab82-2695a093de69.PNG">
</p>


#### III - Modify the PATH variables and run inference.py or your custom code and enjoy:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/168441339-06cdfebf-aa49-4af0-a04a-fe126e1b33bb.png">
</p>

## Project 2 - Segmentation using U-Net
