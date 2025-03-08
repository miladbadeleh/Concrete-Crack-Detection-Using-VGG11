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
The VGG11 model was trained for **10 epochs** on a custom dataset, achieving excellent performance on both the training and validation sets. Below is a summary of the results:

### Training and Validation Performance
- **Training Accuracy**: The model achieved a training accuracy of **99.18%** by the final epoch, with a training loss of **0.0377**.
- **Validation Accuracy**: The validation accuracy reached **99.45%** by the final epoch, with a validation loss of **0.0151**.
- **Test Accuracy**: After training, the model was evaluated on the test set, achieving an impressive **99.74% accuracy** with a test loss of **0.0103**.

### Detailed Epoch-wise Performance
| Epoch | Train Loss | Train Accuracy | Val Loss | Val Accuracy |
|-------|------------|----------------|----------|--------------|
| 1     | 0.3499     | 93.26%         | 0.0387   | 99.22%       |
| 2     | 0.0558     | 99.18%         | 0.0332   | 99.29%       |
| 3     | 0.1053     | 98.03%         | 0.0161   | 99.55%       |
| 4     | 0.0200     | 99.48%         | 0.0191   | 99.49%       |
| 5     | 0.1474     | 97.76%         | 0.0963   | 97.12%       |
| 6     | 0.0358     | 99.14%         | 0.0196   | 99.29%       |
| 7     | 0.0170     | 99.53%         | 0.0173   | 99.53%       |
| 8     | 0.0281     | 99.44%         | 0.0134   | 99.61%       |
| 9     | 0.0478     | 99.11%         | 0.0181   | 99.47%       |
| 10    | 0.0377     | 99.18%         | 0.0151   | 99.45%       |

### Key Observations
1. **High Accuracy**: The model consistently achieved high accuracy on both the training and validation sets, indicating that it learned the dataset effectively.
2. **Low Loss**: The training and validation losses decreased significantly over the epochs, demonstrating that the model minimized errors effectively.
3. **Generalization**: The high test accuracy (**99.74%**) indicates that the model generalizes well to unseen data, making it suitable for real-world applications.

### Model Performance Visualization
Below is a visualization of the training and validation accuracy over the epochs:

---

### Conclusion
The VGG11 model demonstrated exceptional performance on the custom dataset, achieving **99.74% accuracy** on the test set. This makes it a reliable choice for image classification tasks. The model's ability to generalize well to unseen data highlights its robustness and effectiveness.

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
