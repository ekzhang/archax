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
import pandas as pd
import json

from tensorflow.keras.models import Model, load_model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization
from tensorflow.python.client import device_lib

if __name__ == '__main__':
    """
    Measures the time needed to vectorize and embed IMDB datasets
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('-c', '--count', type=int)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()
    device, count, output = str(args.device), int(args.count), args.output

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
        selected_idxs = np.random.choice(test_data.shape[0], size=count, replace=False)
        test_data = test_data[selected_idxs]
        
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

        measurements = {
            "device": device,
            "sample_size": count,
            "vectorization_time": vectorization_time,
            "embedding_time": embedding_time
        }
        print(measurements)

        if output is not None:
            with open(output, "w") as fout:
                json.dump(measurements, fout, indent=2)
        