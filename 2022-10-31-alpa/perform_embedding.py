import io
import os
import re
import shutil
import string
import tensorflow as tf
import numpy as np
import pickle
import time
import argparse

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization
from tensorflow.python.client import device_lib

def time_imdb_embedding(device, count):
    """
    Measures the time needed to vectorize and embed IMDB datasets
    - device: either "cpu" or "gpu"
    - count: number of test samples to use, between 1 and 20000 (inclusively)
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='gpu')
    parser.add_argument('-c', '--count', default='20000', type=int)
    args = parser.parse_args()

    # mask GPU device to run on CPU
    if device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device_str = '/CPU:0'
    else:
        device_str = '/GPU:0'
    print(device_lib.list_local_devices())
    
    # load test data
    with tf.device(device_str):
        dataset_dir = './aclImdb'
        test_dir = os.path.join(dataset_dir, 'test')
        test_ds = tf.keras.utils.text_dataset_from_directory(test_dir, batch_size=1,)
        test_data = np.vstack([x for x, y in test_ds])
        test_data = test_data[:count]
        
        # load model
        model_path = 'imdb_classification_model.pkl'
        with open(model_path, 'rb') as fin:
            model = pickle.load(fin)
        
        # extract layers
        vectorization_layer = model.get_layer("text_vectorization")
        embedding_layer = model.get_layer("embedding")

        # measure time for vectorization and embedding layer
        start_time = time.time()
        vectorization_output = vectorization_layer(test_data)
        finish_time = time.time()
        vectorization_time = finish_time - start_time

        start_time = time.time()
        embedded_output = embedding_layer(vectorization_output)
        finish_time = time.time()
        embedding_time = finish_time - start_time

        print('{}\t{}'.format(vectorization_time, embedding_time))
        