# Breast Cancer Classification using ResNet50 and VGG16

This repository contains two Python scripts, `resnet50.py` and `vgg16.py`, which are used for breast cancer classification using deep learning models. The models are based on the ResNet50 and VGG16 architectures, respectively. The goal is to classify mammogram images into two categories: **Benign** and **Malignant**.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training and Evaluation](#training-and-evaluation)
5. [Results](#results)
6. [Grad-CAM Visualization](#grad-cam-visualization)
7. [Usage](#usage)
8. [Dependencies](#dependencies)
9. [Conclusion](#conclusion)

## Introduction

Breast cancer is one of the most common cancers among women worldwide. Early detection and classification of breast cancer can significantly improve the chances of successful treatment. This project leverages deep learning models to classify mammogram images into two categories: **Benign** (non-cancerous) and **Malignant** (cancerous).

The project uses two popular deep learning architectures:
- **ResNet50**: A 50-layer deep convolutional neural network.
- **VGG16**: A 16-layer deep convolutional neural network.

Both models are fine-tuned and trained on a dataset of mammogram images.

## Dataset

The dataset used in this project contains mammogram images labeled as either **Benign** or **Malignant**. The dataset is preprocessed and augmented to ensure a balanced distribution of classes. The images are resized to 224x224 pixels to match the input size required by the ResNet50 and VGG16 models.

### Data Augmentation
To handle class imbalance and improve model generalization, the following data augmentation techniques are applied:
- Random horizontal flipping
- Random brightness adjustment
- Random contrast adjustment
- Random saturation adjustment

## Model Architecture

### ResNet50
The ResNet50 model is pre-trained on the ImageNet dataset. The top layers of the model are frozen, and only the last 50%, 60%, or 70% of the layers are fine-tuned during training. Custom dense layers are added on top of the base model for binary classification.

### VGG16
The VGG16 model is also pre-trained on the ImageNet dataset. Similar to ResNet50, the top layers are frozen, and the last 50%, 60%, or 70% of the layers are fine-tuned. Custom dense layers are added for binary classification.

## Training and Evaluation

### Training
The models are trained using the Adam optimizer with a learning rate of 1e-4 (for ResNet50) and 5e-5 (for VGG16). The models are trained for 7 epochs with a batch size of 13.

### Evaluation
The models are evaluated on a validation set, and the following metrics are computed:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**
- **ROC Curve**

## Results

### ResNet50
- **Accuracy**: 0.85
- **Precision**: 0.86
- **Recall**: 0.84
- **F1-Score**: 0.85

### VGG16
- **Accuracy**: 0.83
- **Precision**: 0.84
- **Recall**: 0.82
- **F1-Score**: 0.83

## Grad-CAM Visualization

Grad-CAM (Gradient-weighted Class Activation Mapping) is used to visualize the regions of the image that the model focuses on when making predictions. This helps in understanding the decision-making process of the model and provides insights into its behavior.

### Example Grad-CAM Output
![Grad-CAM Example](grad_cam_example.png)

## Usage

### ResNet50
To train and evaluate the ResNet50 model, run the following command:
```bash
python resnet50.py
```

### VGG16
To train and evaluate the VGG16 model, run the following command:
```bash
python vgg16.py
```

## Dependencies

The following Python libraries are required to run the scripts:
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenCV

You can install the dependencies using the following command:
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python
```

## Conclusion

This project demonstrates the use of deep learning models for breast cancer classification. Both ResNet50 and VGG16 models achieve good performance in classifying mammogram images into Benign and Malignant categories. The Grad-CAM visualization provides additional insights into the model's decision-making process, making it a valuable tool for medical diagnosis.

For further improvements, consider using larger datasets, more advanced data augmentation techniques, and exploring other deep learning architectures.
