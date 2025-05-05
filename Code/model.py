import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = training_datagen.flow_from_directory(
'/content/drive/MyDrive/Colab Notebooks/Brain_Tumor_Detection/Dataset /Train ',
    target_size=(224, 224),
    batch_size=32,
    shuffle=False,
    class_mode='binary'
)

from google.colab import drive
drive.mount('/content/drive')

test_datagen =ImageDataGenerator(rescale=1./255)
test_set=test_datagen.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/Brain_Tumor_Detection/Dataset /Test ',target_size=(224,224),batch_size=16,shuffle=False,class_mode='binary')


cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=224 , kernel_size=3 , activation='relu' , input_shape=[224,224,3]))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

cnn.add(tf.keras.layers.Conv2D(filters=224 , kernel_size=3 , activation='relu' ))

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2 , strides=2))

cnn.add(tf.keras.layers.Dropout(0.5))

cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=1 , activation='sigmoid'))

cnn.compile(optimizer = 'Adam',loss ='binary_crossentropy',metrics = ['accuracy'])

cnn.fit(x = training_set , validation_data = test_set , epochs = 10)

from keras.preprocessing import image

test_image = tf.keras.utils.load_img('/content/drive/MyDrive/Colab Notebooks/Brain_Tumor_Detection/Dataset /Prediction /yes3.JPG',target_size=(224,224))
test_image = tf.keras.utils.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = cnn.predict(test_image)
training_set.class_indices
print(result)

if result[0][0] == 1:
    print('yes')
else:
    print('no')
     



