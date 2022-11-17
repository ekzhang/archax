import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device_str = '/CPU:0'

cached_dset_path = 'flower_photos.pickle'
if not os.path.isfile(cached_dset_path):
    import pathlib
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    with open(cached_dset_path, 'wb') as fout:
        pickle.dump(data_dir, fout)
else:
    with open(cached_dset_path, 'rb') as fin:
        data_dir = pickle.load(fin)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# define the model
num_classes = len(class_names)

model = Sequential([
  layers.RandomFlip("horizontal",
                    input_shape=(img_height,
                                img_width,
                                3),
                    name='random_flip'),
  layers.RandomRotation(0.1, name='random_rotation'),
  layers.RandomZoom(0.1, name='random_zoom'),
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3), name='rescaling'),
  layers.Conv2D(16, 3, padding='same', activation='relu', name='first_conv'),
  layers.MaxPooling2D(name='first_maxpool'),
  layers.Conv2D(32, 3, padding='same', activation='relu', name='second_conv'),
  layers.MaxPooling2D(name='second_maxpool'),
  layers.Conv2D(64, 3, padding='same', activation='relu', name='third_conv'),
  layers.MaxPooling2D(name='third_maxpool'),
  layers.Dropout(0.2, name='drop_out'),
  layers.Flatten(name='flatten'),
  layers.Dense(128, activation='relu', name='dense'),
  layers.Dense(num_classes, name='output')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

with open('cnn_flowers.pickle', 'wb') as fout:
  pickle.dump(model, fout)
