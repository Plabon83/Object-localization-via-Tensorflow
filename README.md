# **Object Localization with TensorFlow**

![image](https://github.com/user-attachments/assets/afaa5cb7-deb8-413f-a2df-a3618929a726)



This repository demonstrates the implementation of an **Object Localization Model** using TensorFlow. The project predicts both the **class** and **bounding box** of objects in images and evaluates performance using a custom **Intersection over Union (IoU)** metric.

---

## **Table of Contents**

1. [Overview](#overview)  
2. [Model Architecture](#model-architecture)  
3. [Dataset](#dataset)  
4. [Training and Evaluation](#training-and-evaluation)  
5. [Custom IoU Metric](#custom-iou-metric)  
6. [Results](#results)  
7. [Usage](#usage)  
8. [Requirements](#requirements)  
9. [References](#references)

---

## **Overview**

This project is part of the Coursera Guided Project: *Object Localization with TensorFlow*. The objective is to classify objects into 9 categories and predict their bounding box coordinates.

### **Classes**
The model detects and classifies emojis into the following categories:

| Class Name     | Emoji File      |
|----------------|-----------------|
| Happy          | 😀 (`1F642.png`) |
| Laughing       | 😂 (`1F602.png`) |
| Skeptical      | 🤨 (`1F928.png`) |
| Sad            | 😰 (`1F630.png`) |
| Cool           | 😎 (`1F60E.png`) |
| Whoa           | 😯 (`1F62F.png`) |
| Crying         | 😭 (`1F62D.png`) |
| Puking         | 🤮 (`1F92E.png`) |
| Nervous        | 😬 (`1F62C.png`) |

---

## **Model Architecture**

The model consists of convolutional layers for feature extraction, followed by fully connected layers for classification and bounding box regression.

### **Model Outputs**
1. **`class_out`**: Predicts the class of the object (softmax output with 9 categories).
2. **`box_out`**: Predicts the bounding box coordinates (`x`, `y`).

### **Model Summary**
```plaintext
Model: "Object Localization"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
image (InputLayer)              [(None, 144, 144, 3) 0                                            
__________________________________________________________________________________________________
conv2d_15 (Conv2D)              (None, 142, 142, 16) 448         image[0][0]                      
...
class_out (Dense)               (None, 9)            2313        dense_3[0][0]                    
box_out (Dense)                 (None, 2)            514         dense_3[0][0]                    
==================================================================================================
Total params: 659,819
Trainable params: 658,827
Non-trainable params: 992

```
---

## **Dataset**

The dataset used in this project consists of:
- **Input Images**: RGB images of size `144x144x3`.
- **Labels**:
  - **Classes**: One-hot encoded labels for the 9 emoji categories.
  - **Bounding Boxes**: Normalized coordinates (`x`, `y`) representing the center of the bounding box.

---

### **Emoji Classes**
The dataset includes the following emoji types:

| Class Name     | Emoji File      |
|----------------|-----------------|
| Happy          | 😀 (`1F642.png`) |
| Laughing       | 😂 (`1F602.png`) |
| Skeptical      | 🤨 (`1F928.png`) |
| Sad            | 😰 (`1F630.png`) |
| Cool           | 😎 (`1F60E.png`) |
| Whoa           | 😯 (`1F62F.png`) |
| Crying         | 😭 (`1F62D.png`) |
| Puking         | 🤮 (`1F92E.png`) |
| Nervous        | 😬 (`1F62C.png`) |

---

## **Training and Evaluation**

### **Loss Functions**
- **Classification Loss (`class_out`)**:  
  - *Categorical Crossentropy*: Measures how well the model predicts the emoji class.
- **Bounding Box Loss (`box_out`)**:  
  - *Mean Squared Error (MSE)*: Calculates the error between the predicted and actual bounding box coordinates.

### **Optimizer**
- The **Adam optimizer** is used with a learning rate of `1e-3`.

### **Metrics**
- **Classification Accuracy**: Measures the percentage of correctly predicted emoji classes.
- **Custom IoU (Intersection over Union)**: Evaluates the overlap between predicted and ground truth bounding boxes.

---

## **Custom IoU Metric**

### **Purpose**
The IoU (Intersection over Union) metric is used to measure the accuracy of bounding box predictions. It calculates the overlap between the predicted and ground truth bounding boxes.

### **Implementation**
The IoU is implemented as a custom TensorFlow metric.

#### **Code Snippet**
```python
class IoU(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super(IoU, self).__init__(**kwargs)
        self.iou = self.add_weight(name='iou', initializer='zeros')
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.num_ex = self.add_weight(name='num_ex', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Intersection and union area calculations
        ...

    def result(self):
        return self.iou

```


## **Results**
The model achieves:

High classification accuracy for predicting emoji classes.
Reasonable IoU values for bounding box predictions, indicating accurate object localization.

## **Usage**
### Setup
Clone the repository and navigate to the project directory:
#### bash
Copy code
git clone https://github.com/username/object-localization-tensorflow.git
cd object-localization-tensorflow
Install the required dependencies:
#### bash
Copy code
pip install -r requirements.txt
Training
Train the model by running:

#### bash
Copy code
python train.py
Evaluation
Evaluate the trained model:

#### bash
Copy code
python evaluate.py
Inference
Run inference to make predictions on a new image:

#### python
Copy code
from tensorflow.keras.models import load_model

#### Load the trained model
model = load_model('path_to_model', custom_objects={'IoU': IoU})

#### Preprocess the input image
image = preprocess_image('path_to_image')

### Make predictions
predictions = model.predict(image)
print("Class:", predictions['class_out'])
print("Bounding Box:", predictions['box_out'])

## **Requirements**
Python 3.7+
TensorFlow 2.0+
NumPy
Matplotlib
## **References**
Coursera Guided Project: Object Localization with TensorFlow
TensorFlow Documentation: Custom Metrics


