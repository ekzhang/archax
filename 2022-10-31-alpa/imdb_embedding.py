import io
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.layers import TextVectorization


if __name__ == '__main__':

    dataset_dir = './aclImdb'
    if not os.path.isdir(dataset_dir):
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                        untar=True, cache_dir='.',
                                        cache_subdir='')

    train_dir = os.path.join(dataset_dir, 'train')
    remove_dir = os.path.join(train_dir, 'unsup')
    if os.path.isdir(remove_dir):
        shutil.rmtree(remove_dir)

    batch_size = 1024
    seed = 123
    train_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train', batch_size=batch_size, validation_split=0.2,
        subset='training', seed=seed)
    val_ds = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/train', batch_size=batch_size, validation_split=0.2,
        subset='validation', seed=seed)
    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    embedding_dim=16

    # Create a custom standardization function to strip HTML break tags '<br />'.
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation), '')


    # Vocabulary size and number of words in a sequence.
    vocab_size = 10000
    sequence_length = 100

    # Use the text vectorization layer to normalize, split, and map strings to
    # integers. Note that the layer uses the custom standardization defined above.
    # Set maximum_sequence length as all samples are not of the same length.
    vectorize_layer = TextVectorization(
        standardize=custom_standardization,
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Make a text-only dataset (no labels) and call adapt to build the vocabulary.
    text_ds = train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    # Create a classification model
    model = Sequential([
        vectorize_layer,
        Embedding(vocab_size, embedding_dim, name="embedding"),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1)
    ])

    # compile and run model
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs")
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=[tensorboard_callback]
    )
    
