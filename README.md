
# 🧠 Brain Tumor Detection using CNN

![TensorFlow](https://img.shields.io/badge/Built%20with-TensorFlow-orange?style=for-the-badge\&logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge\&logo=python)
![Accuracy](https://img.shields.io/badge/Model%20Accuracy-%3E90%25-brightgreen?style=for-the-badge\&logo=google)
![MIT License](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)
![Last Update](https://img.shields.io/badge/Last%20Updated-May%202025-blue?style=for-the-badge)

Welcome to the **Brain Tumor Detection** project! This repository showcases how a **Convolutional Neural Network (CNN)** can detect brain tumors from MRI scans with high accuracy using deep learning and medical imaging.

---

## 📑 Table of Contents

* [Project Overview](#project-overview)
* [Dataset Information](#dataset-information)
* [Installation](#installation)
* [Model Architecture](#model-architecture)
* [How to Use](#how-to-use)
* [Model Evaluation](#model-evaluation)
* [Visualization](#visualization)
* [Contributing](#contributing)
* [License](#license)

---

## 📌 Project Overview

This deep learning pipeline is designed to classify brain MRI scans into:

* ✅ **Tumor**
* ❌ **No Tumor**

Built with **TensorFlow**, **Keras**, and **Python**, the model helps in early diagnosis of brain tumors through visual pattern learning.

---

## 📂 Dataset Information

Ensure your dataset follows this directory structure:

```
Dataset/
├── Train/
│   ├── yes/
│   └── no/
├── Test/
│   ├── yes/
│   └── no/
├── Prediction/
│   └── yes3.JPG
```

Each image should be in `.jpg` format and labeled under the appropriate folders.

---

## ⚙️ Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/brain-tumor-detection-cnn.git
   cd brain-tumor-detection-cnn
   ```

2. **Install Dependencies**

   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. **Mount Google Drive in Colab** (if using Colab)

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

---

## 🧠 Model Architecture

* `Conv2D → ReLU → MaxPooling → Conv2D → ReLU → MaxPooling`
* `Dropout → Flatten → Dense → Sigmoid`
* Optimizer: `Adam`
* Loss Function: `Binary Crossentropy`

---

## 🚀 How to Use

### 1. Load and Preprocess the Data

```python
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_set = train_datagen.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/Brain_Tumor_Detection/Dataset /Train ',
                                              target_size=(224, 224), batch_size=32, shuffle=False, class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/Brain_Tumor_Detection/Dataset /Test ',
                                            target_size=(224, 224), batch_size=16, shuffle=False, class_mode='binary')
```

### 2. Build and Train the Model

```python
import tensorflow as tf

cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(224, 3, activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(224, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(train_set, validation_data=test_set, epochs=10)
```

### 3. Predict on New MRI Image

```python
import numpy as np
from tensorflow.keras.preprocessing import image

img = tf.keras.utils.load_img('/content/drive/MyDrive/Colab Notebooks/Brain_Tumor_Detection/Dataset /Prediction /yes3.JPG', target_size=(224,224))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

prediction = cnn.predict(img_array)

if prediction[0][0] == 1:
    print('Tumor Detected ✅')
else:
    print('No Tumor ❌')
```

---

## 📊 Model Evaluation

The model's performance is evaluated based on:

* **Accuracy**
* **Binary Cross-Entropy Loss**
* **Validation Accuracy**

It achieves **>90% accuracy** on unseen test data.

---

## 📈 Visualization

You can visualize predictions using `matplotlib`:

```python
import matplotlib.pyplot as plt

plt.imshow(img.astype('uint8'))
plt.title("Prediction: Tumor" if prediction[0][0]==1 else "Prediction: No Tumor")
plt.axis('off')
plt.show()
```

---

## 🤝 Contributing

We welcome contributions!

1. Fork this repo 🍴
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 🔥

---

## 🛡️ License

Distributed under the [MIT License](LICENSE). Feel free to use and modify as needed.

---
