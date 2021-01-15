# -*- coding: utf-8 -*-
"""
@author: Abdou
"""

import pandas as pd
import numpy as np
from glob import glob
import cv2
from skimage import io
from tqdm import tqdm
import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

from google.colab import drive
drive.mount('/content/gdrive')


data_dir = pathlib.Path('/content/gdrive/My Drive/Data Final/ABC/')

test_dir = pathlib.Path('/content/gdrive/My Drive/CrossVal/ABC/')



batch_size = 32
img_height =  224
img_width =  224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.25,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
  test_dir,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)



class_names = train_ds.class_names
print(class_names)


num_classes = 2


"""
data_augmentation = keras.Sequential(
  [
    layers.experimental.preprocessing.RandomFlip("horizontal"),
    layers.experimental.preprocessing.RandomRotation(0.1),
  ]
)


my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

"""

iter=30
sommeAC=0
sommeLs=0

for i in range(0,iter):
      print('########Start########:', i+1)

      model = Sequential([
      layers.experimental.preprocessing.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(0.2),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
      ])
      
      model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

      epochs = 15
      history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=epochs,
      )

      score = model.evaluate(test_ds, verbose=2)
      print('Test loss:', score[0])
      print('Test accuracy:', score[1])
      sommeLs=sommeLs+score[0]
      sommeAC=sommeAC+score[1]
      print('########End########:')


print('######################## Moyenne du loss:', (sommeLs/iter))
print('######################## Moyenne de Accuracy:', (sommeAC/iter))

