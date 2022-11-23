# Importing libraries
import os
import numpy as np
import tensorflow as tf
import subprocess 
import pandas as pd 
import pickle

# Downloading the dataset.
if not os.path.exists('Sunspots.csv'):
    subprocess.call(['wget', 'https://raw.githubusercontent.com/adib0073/TimeSeries-Using-TensorFlow/main/Data/Sunspots.csv'])
else:
    print('File already exists. Skipping download.')

# Reading in and processing the dataset.
df = pd.read_csv('Sunspots.csv')
index, sunspots = np.array(df['Date']), np.array(df['Monthly Mean Total Sunspot Number'])

# Preparing the data for training.
SPLIT_PROP, WIN_LENGTH, BATCH_SIZE = 0.9, 30, 32
split_i = int(len(sunspots) * SPLIT_PROP)
time_train, time_valid = index[:split_i], index[split_i:]
series_train, series_valid = sunspots[:split_i], sunspots[split_i:]

# Creating the windowed dataset. Citation: https://github.com/lmoroney/dlaicourse
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer).map(lambda w: (w[:-1], w[1:]))
    ds = ds.batch(batch_size).prefetch(1)
    return ds

# Creating the model.
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.LSTM(100),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compiling the model.
opt = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=opt,
              metrics=['mae'])

EPOCHS = 10
hist = model.fit(windowed_dataset(series_train, WIN_LENGTH, BATCH_SIZE, 1000),
                 epochs=EPOCHS,
                 validation_data=windowed_dataset(series_valid, WIN_LENGTH, BATCH_SIZE, 1000))

# Saving the model. 
with open('lstm.pickle', 'wb') as fout:
    pickle.dump(model, fout)
