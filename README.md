ViT - Bone image classification and fracture image detection

# Entry
Nowadays, Internet platforms and digital health technologies, which are adopted by a wide range of users, play an important role in medical diagnosis and treatment processes. Especially in recent years, the development of artificial intelligence and deep learning techniques has enabled the use of new methods in medical image analysis. In this project, the findings of the study conducted using the Vision Transformer (ViT) model for the classification of bone fractures are presented.
The Vision Transformer (ViT) model consists of a combination of artificial neural networks and layers that form the basis of image processing with the transformer architecture. This model aims to classify bone fracture images by analyzing them. The ViT model plays an effective role in the accurate detection and classification of bone fractures. The model learns by analyzing the images of broken and non-broken bones in the data set given during the training process. In this learning process, fine details in the bone structure and the characteristics of the fracture sites are determined. Using these characteristics, the model learns to understand the differences between Decayed and intact bones.

# Augmentation
Since there is a class imbalance in the data set, data augmentation (data augmentation operations have been performed.)
popular data augmentation techniques have been used.
[Data Augmentation](https://github.com/MuhammedDilli/bone-fracture-clasification/blob/main/Augmentation.ipynb).

# Dataset
[the original data set:](https://figshare.com/articles/dataset/The_dataset/22363012/6#:~:text=FracAtlas%20is%20a%20musculoskeletal%20bone,freely%20available%20for%20any%20purpose).



Veri dengesizliği giderilmiş (augmentation uygulanmış) [data set:](https://drive.google.com/drive/folders/1NE0g2E59HRR8-kuToFSlJst1KZKVH6fW?usp=sharing).


# Model and Description
ViT-B/16;
Image Recognition: ViT models are used for the purpose of recognizing and classifying images. For example, it can identify objects in an image and determine which class they belong to.
Transformer Architecture: ViT has adapted the Transformer architecture, which is successfully used in natural language processing tasks, to image recognition tasks. This approach allows images to be treated as sequences of pixels.
Learning Features: The model learns important features in the images and uses these features in classification tasks.

In summary, the ViT-B/16 model, as a member of the Vision Transformer family, is a model used for image recognition and classification tasks. Using the power of transformer architecture, it learns important features in images and uses these features in recognition and classification processes.

# education
i did the first training without applying augmentation, the first results I received:
![first_education](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/829f5be9-fc05-4c5e-8f17-381a083b93f4)

the first test result :

![test1](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/e8b18945-d0df-463b-9398-86f2f076b2c1)



When I ran the project through google colap, it was constantly interrupted by gridsearch
i didn't get the result I wanted with gridsearch.
At the end of 17 different trainings with Bayesian, the maximum obtained training data was given 75%.
in these trainings, it was determined that the problem was during the augmentation and the augmentation was applied again from the very beginning.
there was an improvement in the results after augmentation.

# Results
While the result is obtained over 96% if the data increase is applied [you can click here](https://github.com/MuhammedDilli/bone-fracture-clasification/blob/main/SONUCC_adl%C4%B1_not_defterinin_kopyas%C4%B1.ipynb)
![accuracy](https://github.com/MuhammedDilli/bone-fracture-clasification/assets/100585981/8f0f65a8-01f2-4c45-9791-372fa0a95674)



if the data increase was not applied, the test data gave a result of 53%.[you can click here](https://github.com/MuhammedDilli/bone-fracture-clasification/blob/main/ham_sonuc_.ipynb)




