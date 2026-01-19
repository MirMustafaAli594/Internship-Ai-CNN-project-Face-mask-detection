#  AICTE Internship-Ai-CNN-project-Face-mask-detection 
Google Collab notebook link - https://colab.research.google.com/github/MirMustafaAli594/Internship-Ai-CNN-project-Face-mask-detection/blob/main/Project_face_Mask_Detection_using_CNN.ipynb
# Face Mask Detection Using CNN
# Project Overview:
This project implements a Face Mask Detection system using Convolutional Neural Networks (CNN) to classify facial images into With Mask and Without Mask categories. The model learns facial features through deep learning and performs binary image classification.This project demonstrates the application of CNNs in computer vision for public safety and health-related projects.

 # Objectives:
-Develop a CNN-based model for face mask detection

-Perform image classification using deep learning

-Apply convolution, pooling, and dense layers effectively

-Evaluate model performance using accuracy and loss metrics

 # Technologies & Tools Used:
-Python

-TensorFlow / Keras

-Convolutional Neural Networks (CNN)

-NumPy

-Matplotlib

-OpenCV

-Google Colab

# Architecture Diagram:

Explanation of Architecture

-Input Layer – Facial images resized to a fixed shape

-Preprocessing – Normalization and array conversion

-Convolutional Layers – Extract spatial facial features

-MaxPooling Layers – Reduce spatial dimensions

-Flatten Layer – Convert feature maps into vectors

-Dense Layers – Classification layers

-Output Layer – Softmax activation for Mask / No Mask prediction

# Dataset:

Dataset consists of labeled facial images:

-With Mask

-Without Mask

Images are preprocessed for consistency and optimal model learning.

 # Model Details:

-Multiple Convolution + MaxPooling layers

-ReLU activation for hidden layers

-Flatten layer for vector conversion

-Dense layers for classification

-Softmax activation for final prediction

-Model compiled with standard Keras optimizer and loss functions

# Results

-The model achieved good training and validation accuracy

-The CNN correctly classifies masked and unmasked faces

-Performance on  test images confirms that the model generalizes well for binary image classification.

Overall, the results demonstrate that the CNN architecture is effective for face mask detection tasks.




