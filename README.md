# Object_Detection

## Table of contents
* [General info](#general-info)
* [Requirements](#requirements)
* [Project 1: Object Detection using Efficientdet](#project-1---object-detection-using-efficientdet)
   * [What is EfficientDet ?](#what-is-efficientdet-)
   * [Training the model](#training-the-model)
   * [Testing locally the model](#testing-locally-the-model)
* [Project 2: Water Segmentation using U-Net](#project-2---water-segmentation-using-u-net)
  * [What is Segmentation ?](#what-is-segmentation-)
  * [Training the model](#training-the-model-1)
  * [Testing locally the model](#testing-locally-the-model-1)
* [Project 3: Face key-point recognition using CNN](#project-3---face-key-point-recognition-using-cnn)
  * [How does it work ?](#how-does-it-work-)
  * [Training the model](#training-the-model-2)
  * [Testing locally the model](#testing-locally-the-model-2)

## General info
In this repository you will learn the basics for detecting objects, keypoints or even making segmentation on images using models such as: EfficientDet, Mobilenet-SSD, U-Net...

## Requirements
For the first project you only need a Google account with <a href=https://colab.research.google.com/>Colab</a> and <a href=https://drive.google.com/drive/my-drive>Drive</a>. *(I am using Colab pro for the training)*

But if you want to train locally you need to install manually Tensorflow Object Detection, you'll find a good tutorial on this <a href=https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html>link</a>

For the inference you need **inference.py** shared in this repository, the directory **saved_model** and the file **label_map.pbtxt**. *(you can automatically download both by running the code in colab)*

## Project 1 - Object Detection using EfficientDet

### What is EfficientDet ?

*I will assume you have some knowledge in computer vision and CNNs, if not you can skip this part.*

EfficientDet is a family of deep learning models designed for object detection, EfficientDet7 achieved state of the art results on COCO dataset, it is both scalable and efficient meaning that it can recognize objects at vastly different scales and need fewer computational performance than the other models.

<p align="center"><a href=https://arxiv.org/pdf/1911.09070v7.pdf><i>link to EfficientDet paper</i></a>

To understand EfficientDet we need to understand two key improvements made:
1. Bi-directional Feature Pyramid Network (BiFPN)
2. Compound Scaling

#### Feature Pyramid Network:

Recognizing objects at different scale is a fundamental challenge in computer vision.

Different authors have tried to solve this differently, there were three main categories of solutions that existed before the introduction of FPN:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/168466672-11fabea6-c628-4dd8-99c0-52b89fec237c.png">
</p>

But they all have some issues:

* **Featurized image pyramid** is too long to train and is infeasible in terms of memory because you need to train a CNN for every scales of an image.
* **Single feature map** is actually used by Faster RCNN but lose representational capacity for object detection in the first layers with low level features embedding.
* **Pyramidal feature hierarchy** is used by SSD, it avoids using low level features in the first levels of a CNN by directly using the high level feature at the end of a CNN and then adds several new layers but by doing so it misses the opportunity to reuse the earlier layers which are important for detecting small objects.

What **Feature Pyramid Network** does is to combine low-resolution, semantically strong features in the later layers with high-resolution and semantically weak features in the earlier layers via a top-down pathway and lateral connections. Thus, leading to **Multi-scale feature fusion**.

*It is somehow similar to the architecture of U-Net when you think about it.*

#### Bi-directional Feature Pyramid Network (BiFPN):

<p align="center"> <i> Feature network design and evolution of FPN </i>
<img src="https://user-images.githubusercontent.com/65224852/168467759-58196966-3795-4e5a-9dd4-5d9ea02f60ed.png">
</p>

What EfficientDet, and BiFPN in particular, did was to:

1. Add bottum-up path aggregation network, conventional top-down FPN is limited by the one-way information flow. (Making it bidirectional)
2. Remove all nodes that have only one input edge. The intuition is that if a node has only one input edge with no feature fusion, then it will have less contribution to the feature network.
3. Add an extra edge from the original input to the output node if they are at the same level in order to fuse more features without adding much cost
4. Treat each bidirectional path as one single layer and have multiple of these to enable more high-level feature fusion.

#### Compound Scaling:

The second key improvement was made by EfficientNet (the backbone of EfficientDet) with **compound scaling**.

Previous work mostly scale up a baseline detector by employing bigger backbone networks (ResNets, AmoebaNet..) using larger input images or stacking more FPN layers. These methods are usually ineffective since they only focus on a single or limited scaling dimensions.

They proposed to use a single compound coefficient to jointly scale up all three dimensions while mantaining a balance between all dimensions of the network.

<p align="center"> <i> Model Scaling </i>
<img src="https://user-images.githubusercontent.com/65224852/168483264-b0b7c6d8-f2df-4355-b7dc-7dc26897112c.png">
</p>

#### EfficientDet final architecture:

To conclude by combining an EfficientNet backbone, a Bi directionnal Feature Pyramid Network and convolutions we get this:

<p align="center"> <i> EfficientDet model architecture </i>
<img src="https://user-images.githubusercontent.com/65224852/168483386-81252296-fc3f-4517-ab22-5b7681f19abf.png">
</p>

### Training the model:

1. First open the file ObjectDetection.ipynb of this repository in Colab.
2. Then you need images in **JPG** format and annotations in **Pascal VOC** format (xml files). *(You can use <a href=https://github.com/Serenalyw/FastAnnotation>FastAnnotations</a>, a framework that I made :smile:)*
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

## Project 2 - Water Segmentation using U-Net

### What is segmentation ?

In this project we will try to identify water on images thanks to a  <a href="https://www.kaggle.com/datasets/gvclsu/water-segmentation-dataset"> dataset from Kaggle </a> using a technique called Segmentation.

Segmentation is made with the use of an autoencoder which is an unsupervised Artificial Neural Network that attempts to encode the data by compressing it into the lower dimensions (bottleneck layer or code) and then decode the data to construct the targeted mask.

A mask is an image made of numbers or colors corresponding to the different classes present in the image.

<p align="center"> <b>Exemple of mask</b> </p>
<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/184553362-a974b54f-bf3b-4292-a739-ad82bc9a11f6.png">
<img src="https://user-images.githubusercontent.com/65224852/184553212-96e977a6-1bb6-4f97-b628-95ecb05420dd.png">
</p>

Here we are going to use U-Net as our autoencoder which is a model generally used in medical segmentation in order to detect diseases or certain parts of the body in order to operate surgeries.

Below you can see the U-Net architecture:
<p align="center"> <b>U-Net architecture</b> </p>
<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/136661667-c1bf2334-38ec-4737-9442-bfde615e1d3a.png"/>
<br><a href="https://arxiv.org/pdf/1505.04597v1.pdf">(From 2015 U-Net paper)</a>
</p>

*In addition, if you want to improve the performances of your model you can add a pretrained model such as ResNet50 or VGG19 as the encoder at the start of your U-Net model and then attach decoder at the end*

### Training the model:

1. First open the file Segmentation.ipynb of this repository in Colab.
2. Then you need images and masks *(jpg or png or other type of images...)*.
3. Once you have them simply upload them to your Drive.
4. Finally modify the DATASET_PATH variable and the different paths where the dataset is made.
``` python
for img in os.listdir(DATASET_PATH+'/Annotations/ADE20K'):
  for img2 in os.listdir(DATASET_PATH+'/JPEGImages/ADE20K'):
```
5. Run the code and enjoy.

And these are some results I got after running the algorithm:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/184554364-42737472-c94c-45cb-9c0e-647c1dc1db8f.PNG">
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/184554440-49f9b9e1-7b34-409f-b86a-c49591df7ad1.png">
</p>

The model has some difficulties around the edges of the water but he has the idea, the reason is because I didn't use a pre-trained model and made everything from scratch with a few epochs.

### Testing locally the model:

Just like the object detection above, we need to repeat the same steps:

#### I- Download the trained model by running this code in the notebook in colab:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/189476421-243e6ccb-6411-4c5e-a202-c78a089d692b.png">
</p>

#### II - Copy and paste where you want the files:

<p align="center"> <i>this is where your files are downloaded</i>
<img src="https://user-images.githubusercontent.com/65224852/189476423-3d2dc32d-bd45-4817-bc8e-66801937904d.png">
</p>

#### III - Modify the PATH variables and run inference_segmentation.py or your custom code and enjoy:

<p align="center">
<img src="https://user-images.githubusercontent.com/65224852/189476424-a21eff7f-b0c8-43c3-b5ab-26d7a9b27fb6.png">
</p>

## Project 3 - Face key-point recognition using CNN

### How does it work ?

Key points can be used for a variety of tasks:
* Apply a filter on a face
* Detect emotions on a face
* Identify someone based on their traits
* Identify what action a person is doing... (Used for sports, or thiefs in a supermarket)

In order to achieve that we need a neural network composed of a CNN and fully connected layers that predict (x, y) coordinates for each key-points, for instance if we have 5 key-points we'll need a linear layer of 10 outputs *(x1, y1, x2, y2, ..., x5, y5)*.

<p align="center"> <b>Neural network architecture (CNN + fully connected layers)</b>
<img src="https://user-images.githubusercontent.com/65224852/208777742-9efea372-5af4-4c87-85e9-06eb1dd11a7a.png">
</p>

### Training the model:

* First open the file FacekeyPoint.ipynb of this repository in Colab.
* Then you need images and annotations (jpg and you might use <a href='https://github.com/Serenalyw/FastAnnotation'>FastAnnotations</a>).
* Once you have them simply upload them to your Drive.
* Finally modify the different paths where the dataset is made and run.

### Testing locally the model:
