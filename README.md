# MNIST Dataset with Deep Convolutional Neural Network (DCNN)

1. What kind of architecture do you use and describe your reason(s) for choosing the architecture?

Answer: 
The architecture I use is the Deep Convolutional Neural Network (DCNN). I use DCNN because its purpose is to classify images from the MNIST dataset, which contains handwritten digit images. DCNN is one of the most effective types of neural networks in processing visual data such as images, and is widely used in computer vision tasks such as image classification, object detection, and segmentation.

2. What kind of dataset do you use? Based on your analysis, please describe the features in the dataset.

The dataset I use is MNIST, which contains handwritten digit images. Each image in the MNIST dataset has a size of 28 x 28 pixels, so each image is represented as a 1D vector with 784 features. Each pixel in the image is represented as a grayscale value between 0 and 255.
The MNIST dataset has 10 classes, each representing a digit from 0 to 9. The goal of the image classification task performed in the code above is to predict the digit appearing in the image, using the DCNN model that has been trained on the training data.

3. Based on your analysis, what kind of data can be chosen as a target? Why do you use it as the target?

The target data that can be chosen for the MNIST dataset is the class label of each digit image. This is because the goal of the MNIST dataset is to perform image classification of digits, making the class label a suitable target to measure the accuracy of the classification model. The class label has discrete values from 0 to 9, representing the digit shown in the image, making it possible to perform multi-class classification using machine learning or deep learning models. Additionally, the class labels are already provided in the MNIST dataset, so there is no need to perform label encoding or manual target label creation.


<h3>How does the Deep Convolutional Neural Network (DCNN) architecture work?</h3>

<img src="https://github.com/skyradez/MNISTWIthDCNN-ANN-Qualification-2023-/blob/main/dcnn.png" />
