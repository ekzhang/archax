# Importing Libraries
import matplotlib.pyplot as plt
import os 
import re 
import shutil
import string 
import tensorflow as tf
import pickle

from tensorflow.keras import layers, losses

# Importing dataset (caching is automatically handled by Keras)
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", URL, 
                                  untar=True, cache_dir='.', 
                                  cache_subdir='')
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
train_dir = os.path.join(dataset_dir, 'train')

# Removing additional folders from the IMDB dataset.
remove_dir = os.path.join(train_dir, "unsup")
shutil.rmtree(remove_dir)

# Setting variables for ML training pipeline.
BATCH_SIZE, EPOCHS = 128, 5
SEED, VALIDATION_RATIO = 243, 0.2

# Training data.
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_RATIO,
    subset='training',
    seed=SEED)

# Validation data.
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train',
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_RATIO,
    subset='validation',
    seed=SEED)

# Testing data.
test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test',
    batch_size=BATCH_SIZE)

# Function designed to standardize the text data.
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(
        stripped_html,
        '[%s]' % re.escape(string.punctuation),
    '')

# Text Vectorization Layer.
MAX_FEATURES = 10000
SEQUENCE_LENGTH = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=MAX_FEATURES,
    output_mode='int',
    output_sequence_length=SEQUENCE_LENGTH)

train_text = train_ds.map(lambda x, y : x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# Appling TextVectorization to dataset.
train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)
test_ds = test_ds.map(vectorize_text)

# Configuring dataset for performance.
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Creating the model.
EMBEDDING_DIM = 16

model = tf.keras.Sequential([
  layers.Embedding(MAX_FEATURES + 1, EMBEDDING_DIM),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

# Compiling the model.
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',   
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Training the model.
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS)

# Evaluating the model.
loss, accuracy = model.evaluate(test_ds)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Saving the model.
with open('lm.pickle', 'wb') as fout:
    pickle.dump(model, fout)



