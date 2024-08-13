### Bachelor's Final Project

ViT - Bone image classification and fracture image detection

# Entry
Nowadays, Internet platforms and digital health technologies, which are adopted by a wide range of users, play an important role in medical diagnosis and treatment processes. Especially in recent years, the development of artificial intelligence and deep learning techniques has enabled the use of new methods in medical image analysis. In this project, the findings of the study conducted using the Vision Transformer (ViT) model for the classification of bone fractures are presented.
The Vision Transformer (ViT) model consists of a combination of artificial neural networks and layers that form the basis of image processing with the transformer architecture. This model aims to classify bone fracture images by analyzing them. The ViT model plays an effective role in the accurate detection and classification of bone fractures. The model learns by analyzing the images of broken and non-broken bones in the data set given during the training process. In this learning process, fine details in the bone structure and the characteristics of the fracture sites are determined. Using these characteristics, the model learns to understand the differences between Decayed and intact bones.

# Model and Description
ViT-B/16;
Image Recognition: ViT models are used for the purpose of recognizing and classifying images. For example, it can identify objects in an image and determine which class they belong to.
Transformer Architecture: ViT has adapted the Transformer architecture, which is successfully used in natural language processing tasks, to image recognition tasks. This approach allows images to be treated as sequences of pixels.
Learning Features: The model learns important features in the images and uses these features in classification tasks.

In summary, the ViT-B/16 model, as a member of the Vision Transformer family, is a model used for image recognition and classification tasks. Using the power of transformer architecture, it learns important features in images and uses these features in recognition and classification processes.

## Model Architecture: Vision Transformer (ViT)

### Vision Transformer (ViT)
Vision Transformer (ViT) is a cutting-edge model for image recognition that leverages transformer architecture, typically used in natural language processing. Developed by Google Research, ViT surpasses traditional convolutional neural networks (CNNs) in various benchmarks.

### How ViT Works
ViT divides an image into fixed-size patches, which are then linearly embedded. Positional embeddings are added, and the sequences are fed into a transformer encoder. This process allows ViT to capture global context and recognize complex patterns.

### Key Features
- **Patch Embedding:** Splits images into smaller patches.
- **Positional Encoding:** Retains spatial information.
- **Transformer Encoder:** Uses self-attention to understand relationships between patches.

### Advantages
- **Scalability:** Efficiently scaled by increasing model size and training data.
- **Performance:** Achieves higher accuracy in image classification tasks compared to CNNs.

  
<img width="321" alt="vittttt" src="https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/b695f4d8-ce33-4a0e-8116-b2d0f5e09fed">






# Augmentation
Since there is a class imbalance in the data set, data augmentation (data augmentation operations have been performed.)
Popular data augmentation techniques have been used; these include methods such as mosaic and muxip. Additionally,
augmentations like random rotation, horizontal flipping, brightness adjustment, hue and saturation modification,
crop and padding, and Gaussian blur have been applied.

popular data augmentation techniques have been used.

![augmentationn](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/ba60dd58-cb0a-45b6-8a3f-eeb60bbc5035)

# Dataset
### Original Data Distribution

This table shows the number of fractured and non-fractured data in the original dataset, separated by training, validation, and test sets.

| Set    | Fractured | Non-Fractured | Total |
|----------|----------------|------------------|----------|
|Train(70%)|     501        |      2296        | 2797 |
|Val(10%)  |     72         |      339         |  411 |
|Test(%20) |     148        |      674         |  822 |
| **Total**| **721** | **3309** | **4030** |

### Data Distribution After Augmentation

This table shows the number of data after augmentation, separated by training, validation, and test sets.

| Set    | Fractured | Non-Fractured | Total |
|--------|-----------|---------------|-------|
| Train(70%)  |    2296    |     2296     | 4592 |
| Val(10%)     |     72    |     339      | 441  |
| Test(%20)    |    148     |     674      | 822 |
| **Total**| **2516** | **3309** | **6618** |


[The original data set](https://figshare.com/articles/dataset/The_dataset/22363012/6#:~:text=FracAtlas%20is%20a%20musculoskeletal%20bone,freely%20available%20for%20any%20purpose)

[Augmentation applied data set](https://drive.google.com/drive/folders/1NE0g2E59HRR8-kuToFSlJst1KZKVH6fW?usp=sharing)


# TRAİN
i did the first training without applying augmentation, the first results I received:
![first_education](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/829f5be9-fc05-4c5e-8f17-381a083b93f4)

The first test result :

![aaaa](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/f4d5a38d-96df-4092-8126-68922c2b23d7)


When I ran the project through google colap, it was constantly interrupted by gridsearch i didn't get the result I wanted with gridsearch. 

It was not interrupted when running the Bayesian algorithm The maximum training data obtained at the end of 17 different trainings is given as 75%.    
Bayesian Algorithm In these trainings, it was determined that the problem was during magnification and the magnification was applied again from the very beginning. there was an improvement in the results after the augmentation.




The results I got after eliminating the problem

![Testacc](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/3aa04300-e7f7-42be-bb2f-f4ab67b16350)

![trainOrVal](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/1ebfdb9b-06e6-4d83-bca4-facf36ade0b9)

# Fine Tune

![fine tune acc](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/42dfdca2-f291-4a21-bf63-f707a1a2b5ab)


![fine tune](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/34387ecb-a3cd-43b0-b281-13d1c388879a)



# Evaluation Metrics


### Confusion Matrix

A confusion matrix is a tool used to evaluate the performance of a classification model. It compares the actual values with the predicted values, providing a detailed breakdown of model performance.

#### Key Components:
- **True Positives (TP):** Correctly predicted positive instances.
- **True Negatives (TN):** Correctly predicted negative instances.
- **False Positives (FP):** Incorrectly predicted positive instances.
- **False Negatives (FN):** Incorrectly predicted negative instances.

#### Why It Matters:
The confusion matrix helps to:
1. **Understand Model Accuracy:** It provides insights beyond simple accuracy, highlighting specific types of errors.
2. **Evaluate Performance on Imbalanced Data:** It offers a balanced evaluation, crucial for datasets with unequal class distributions.

Below is the confusion matrix for our model, showing its performance in classifying fractured and non-fractured instances:

![cnfsion](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/ecd537f2-7707-48d7-bfba-630fc70b898b)



This matrix helps us identify areas where the model performs well and areas that need improvement, ensuring a more reliable and accurate classification system.

## Model Evaluation Metrics

### Precision
Precision measures how many of the instances predicted as positive are actually positive. In other words, it is the ratio of true positive predictions to the total number of positive predictions made by the model.

**Formula:**
Precision = True Positives (TP) / (True Positives (TP) + False Positives (FP))


### Recall
Recall measures how many of the actual positive instances the model correctly identified. It indicates the model's ability to capture all positive instances.

**Formula:**
Recall = True Positives (TP) / (True Positives (TP) + False Negatives (FN))

### F1 Score
The F1 score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall, and is particularly useful when you need to consider both false positives and false negatives.

**Formula:**
F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

Each of these metrics provides a different perspective on the model's performance. Precision indicates the accuracy of the positive predictions, recall indicates the model's ability to identify all positive instances, and the F1 score balances both precision and recall to give a comprehensive measure of performance.

### Specificity

Specificity measures how many of the actual negative instances the model correctly identified. It indicates the model's ability to capture all negative instances.


**Formula:**
Specificity = True Negatives (TN) / (True Negatives (TN) + False Positives (FP))



![metrg](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/745bd092-1d53-4297-9eae-d44519d04c36)


## ROC (Receiver Operating Characteristic) Curve
The ROC curve is a graphical tool used to evaluate the performance of a classification model. It visualizes how well the model performs across different threshold values.

### Key Components:
- **True Positive Rate (TPR) / Recall / Sensitivity:** The ratio of correctly predicted positive instances to the total actual positive instances.
  - **Formula:** TPR = TP / (TP + FN)

- **False Positive Rate (FPR):** The ratio of incorrectly predicted positive instances to the total actual negative instances.
  - **Formula:** FPR = FP / (FP + TN)
  
![ROC](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/4e83a3c7-8e07-46cd-8bcf-02d3bc4fddb3)



# Model Prediction

 Model prediction is the process where a trained machine learning model is used to make predictions on new, unseen data.

 ![PREDİCT GİTHUB](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/b4833cfc-89b9-4e88-8480-8196361c81f0)
 
# Weight & Biases
Weight & Biases (W&B) is an experiment tracking and optimization platform developed for machine learning projects. Researchers and data scientists can use W&B to track their experiments, perform hyperparameter optimization, and visualize model performance. The platform enhances the efficiency and transparency of the model training process, making it easier to determine which hyperparameter settings yield the best results.

 ### HyperParametre code
 
![hyperparametre](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/4820a255-d825-42fa-a7bf-372dd6047a68)


HyperParametre  graphs

![w B2](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/8cc3c2ab-9bb4-4c10-acc6-5d048f83a166)


### model evaluation metrics and graphs


![w b metik1](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/9ff264a4-4600-493e-a660-64b3488b09e1)


![WB3](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/49df225a-2124-44f9-a7bd-0f376707fb0f)











# Results
| Vision Transformer | Accuracy | Precision | F1 Score | Recall | Specificity | Sensitivity |
|------------------|----------|-----------|----------|--------|-------------|--------------|
| Augmentationsuz  | 0.53     | 0.41      | 0.25     | 0.16   | 0.81        |    0.16     |
| Augmentationlu   | 0.95     | 0.96      | 0.97     |   1    | 0.88        |      1       |
| Fine-tune        | 0.97     | 0.98      | 0.98     |   1    | 0.88        |      1       |

| Model          | Accuracy | Precision | F1 Score | Recall | Specificity |
|----------------|----------|-----------|----------|--------|-------------|
| Vision Transormer| 0.89     | 0.87      | 0.93     |  1     | 0.39      | 
| Swin Transformer | 0.90   | 0.88      | 0.94     |  1     | 0.48        | 
| ResNet         | 0.90     | 0.88      | 0.94     |  1     | 0.48        |











