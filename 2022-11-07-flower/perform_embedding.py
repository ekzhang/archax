if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import PIL
    import tensorflow as tf

    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.data import Dataset

    import os
    import pickle
    import time
    import argparse
    import pandas as pd
    import json

    import sklearn
    from sklearn.utils import resample
    from tensorflow.python.client import device_lib

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=str, choices=['cpu', 'gpu'])
    parser.add_argument('-s', '--sample', type=int)
    parser.add_argument('-o', '--output', type=str)
    parser.add_argument('-t', '--trial', type=int)
    args = parser.parse_args()
    device, sample, output, trial = str(args.device), int(args.sample), args.output, int(args.trial)

    # mask GPU device to run on CPU
    if device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        device_str = '/CPU:0'
    else:
        device_str = '/GPU:0'
    print(device_lib.list_local_devices())

    # load dataset
    with tf.device(device_str):
        test_data_path = 'test_data.pickle'
        batch_size = 32
        if not os.path.isfile(test_data_path):
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

            img_height = 180
            img_width = 180

            test_ds = tf.keras.utils.image_dataset_from_directory(
                data_dir,
                seed=123,
                image_size=(img_height, img_width),
                batch_size=None
            )

            test_datas = [x for x in test_ds]
            X_data, Y_data = [x[0] for x in test_datas], [x[1] for x in test_datas]

            with open(test_data_path, "wb") as fout:
                pickle.dump((X_data, Y_data), fout)
        else:
            with open(test_data_path, "rb") as fin:
                X_data, Y_data = pickle.load(fin)
        
        X_data_new, Y_data_new = resample(
            X_data, Y_data,
            replace=True,
            random_state=10,
            n_samples=sample
        )

        test_ds_new = Dataset.from_tensor_slices(
            (np.array(X_data_new).astype(np.float32), np.array(Y_data_new).astype(np.int32))
        )

        test_ds_new = test_ds_new.cache().shuffle(1000).batch(batch_size=batch_size).prefetch(tf.data.AUTOTUNE)

        # load model
        with open('cnn_flowers.pickle', 'rb') as fin:
            model = pickle.load(fin)

        records = list()
        for _ in range(trial):
            current = np.array(X_data_new).astype(np.float32)

            trial_record = dict()
            for layer in model.layers:
                start_time = time.time()
                current = layer(current)
                time_taken = time.time() - start_time
                trial_record[layer.name] = time_taken
            records.append(trial_record)
        
        output_json = {
            "device": device,
            "sample": sample,
            "trial": records
        }

        with open(output, "w") as fout:
            json.dump(output_json, fout, indent=2)
