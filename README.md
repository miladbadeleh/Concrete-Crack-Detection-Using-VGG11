# VGG11 Image Classification

This repository contains code for training and evaluating a VGG11 model for image classification tasks. The model is trained on a custom dataset and can be used for tasks such as classifying images into different categories.

## Table of Contents
1. [Introduction](#introduction)
2. [VGG11 Architecture](#architecture)
3. [Code Overview](#code-overview)
4. [Training and Evaluation](#training-and-evaluation)
5. [Results](#results)
6. [Usage](#usage)
7. [Dependencies](#dependencies)

---

## Introduction
The goal of this project is to demonstrate how to use the VGG11 architecture for image classification. VGG11 is a convolutional neural network (CNN) architecture that is part of the VGG family of models. It is known for its simplicity and effectiveness in image classification tasks. This code uses PyTorch, a popular deep learning framework, to implement the model.

---

## Architecture
VGG11 is a CNN architecture that consists of 11 layers, including convolutional layers, max-pooling layers, and fully connected layers. The architecture is as follows:

1. **Convolutional Layers**: VGG11 uses a series of convolutional layers with small 3x3 filters to extract features from the input images.
2. **Max-Pooling Layers**: After every few convolutional layers, max-pooling is applied to reduce the spatial dimensions of the feature maps.
3. **Fully Connected Layers**: The final layers of the network are fully connected layers, which are used to classify the extracted features into different categories.

The key feature of VGG11 is its simplicity and depth, which allows it to learn complex patterns in the data.

---

## Code Overview
The code is divided into the following sections:

1. **Data Loading and Preprocessing**:
   - The dataset is loaded using `torchvision.datasets.ImageFolder`.
   - Images are resized to 224x224 pixels and normalized using ImageNet mean and standard deviation values.
   - Data loaders are created for training, validation, and testing.

2. **Model Definition**:
   - The VGG11 model is loaded using `torchvision.models.vgg11(pretrained=True)`.
   - The final fully connected layer is modified to match the number of classes in the dataset.

3. **Training**:
   - The model is trained using the Adam optimizer and CrossEntropyLoss.
   - Training and validation loops are implemented with progress bars using `tqdm`.

4. **Testing**:
   - The trained model is evaluated on a test dataset to measure its performance.

5. **Saving the Model**:
   - The trained model is saved to a file for future use.
---

## Training and Evaluation
### Training
- The model is trained for 10 epochs by default.
- Training and validation losses are monitored to ensure the model is learning effectively.
- The Adam optimizer is used with a learning rate of 0.001.

### Evaluation
- The model is evaluated on a separate test set to measure its generalization performance.
- Test accuracy and loss are reported.

---

## Results
After training, the model achieves the following results:
- **Training Accuracy**: [Insert training accuracy]
- **Validation Accuracy**: [Insert validation accuracy]
- **Test Accuracy**: [Insert test accuracy]

---

## Usage
To use this code, follow these steps:

1. **Set Up the Environment**:
   - Install the required dependencies (see [Dependencies](#dependencies)).
   - Mount Google Drive (if using Google Colab) and organize your dataset into `train`, `validation`, and `test` folders.

2. **Run the Code**:
   - Execute the notebook or script to train the model.
   - The trained model will be saved to `/content/drive/MyDrive/densenet121_model.pth`.

3. **Test the Model**:
   - Use the `test` function to evaluate the model on the test set.

---

## Dependencies
The following Python libraries are required to run this code:
- `torch`
- `torchvision`
- `tqdm`
- `matplotlib`
- `PIL`

Install the dependencies using:
```bash
pip install torch torchvision tqdm matplotlib pillow
