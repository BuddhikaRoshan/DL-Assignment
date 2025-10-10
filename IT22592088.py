#Mount to google drive
from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from zipfile import ZipFile # To unzip the zip folder
import os

#import path 
data_path='/content/drive/MyDrive/DL/lung_images.zip'
#unzip the folder
with ZipFile(data_path,'r') as zip:
  zip.extractall()
  print('The data set has been extracted.')

#Setting Up Dataset Directories
  train_dir = '/path/to/train'
  val_dir = '/path/to/validation'

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

#Loading and Augmenting Images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
