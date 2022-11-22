# Importing Libraries

import os 
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import tensorflow_gnn as tfgnn
import subprocess

# Downloading and unzipping dataset.
if not os.path.exists("mutag.zip"):
    try:
        subprocess.run(["unzip"])
    except:
        raise Exception("Please sudo-apt install unzip.")

    subprocess.run(["wget", "https://storage.googleapis.com/download.tensorflow.org/data/mutag.zip"])
    subprocess.run(["unzip", "mutag.zip"])
    
else:
    print("File already exists, so not downloading.\n")

# Greating the graph tensor specification for Tensorflow.
graph_tensor = tfgnn.GraphTensorSpec.from_piece_specs(
    context_spec=tfgnn.ContextSpec.from_field_specs(features_spec={
                  'label': tf.TensorSpec(shape=(1,), dtype=tf.int32)
    }),
    node_sets_spec={
        'atoms':
            tfgnn.NodeSetSpec.from_field_specs(
                features_spec={
                    tfgnn.HIDDEN_STATE:
                        tf.TensorSpec((None, 7), tf.float32)
                },
                sizes_spec=tf.TensorSpec((1,), tf.int32))
    },
    edge_sets_spec={
        'bonds':
            tfgnn.EdgeSetSpec.from_field_specs(
                features_spec={
                    tfgnn.HIDDEN_STATE:
                        tf.TensorSpec((None, 4), tf.float32)
                },
                sizes_spec=tf.TensorSpec((1,), tf.int32),
                adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                    'atoms', 'atoms'))
    })


# Function to decode a TF graph.
def decode_graph(data):
    G = tfgnn.parse_single_example(
        graph_tensor, data, validate=True
    )

    # extract label from context and remove from input graph
    features = G.context.get_features_dict()
    label = features['label']
    del features['label']
    G = G.replace_features(context=features)

    return G, label

# Establishing path.
train_dir = os.path.join(os.getcwd(), 'mutag', 'train.tfrecords')
val_dir = os.path.join(os.getcwd(), 'mutag', 'val.tfrecords')

# Decoding the dataset.
train_data = tf.data.TFRecordDataset([train_dir]).map(decode_graph)
val_data = tf.data.TFRecordDataset([val_dir]).map(decode_graph)

# Batching the dataset.
BATCH_SIZE, EPOCHS = 32, 1000
train_data = train_data.batch(BATCH_SIZE)
val_data = val_data.batch(BATCH_SIZE)

# Creating the model itself
def build_model(
    graph_tensor,
    num_classes=2,
    num_message_passing=3,
    l2_regularization=5e-4,
    dropout_rate=0.5,
    node_dim=16,
    edge_dim=16,
    message_dim=64,
    next_state_dim=64,
):

    input_G = tf.keras.layers.Input(type_spec = graph_tensor)
    G = input_G.merge_batch_to_components()

    def set_node_state(node_set, *, node_set_name):
        return tf.keras.layers.Dense(node_dim)(node_set[tfgnn.HIDDEN_STATE])
    
    def set_edge_state(edge_set, *, edge_set_name):
        return tf.keras.layers.Dense(edge_dim)(edge_set[tfgnn.HIDDEN_STATE])

    G = tfgnn.keras.layers.MapFeatures(
        node_sets_fn=set_node_state,
        edge_sets_fn=set_edge_state,
    )(G)

    def custom_dense(units, activation="relu"):
        regularizer = tf.keras.regularizers.l2(l2_regularization)
        return tf.keras.Sequential([
            tf.keras.layers.Dense(
                units, 
                activation=activation, 
                kernel_regularizer=regularizer,
                bias_regularizer=regularizer),
            tf.keras.layers.Dropout(dropout_rate),
        ])
    
    for _ in range(num_message_passing):
        G = tfgnn.keras.layers.GraphUpdate(
            node_sets={
                "atoms": tfgnn.keras.layers.NodeSetUpdate(
                    {"bonds": tfgnn.keras.layers.SimpleConv(
                        sender_edge_feature=tfgnn.HIDDEN_STATE,
                        message_fn=custom_dense(message_dim),
                        reduce_type="sum",
                        receiver_tag=tfgnn.TARGET)},
                    tfgnn.keras.layers.NextStateFromConcat(custom_dense(next_state_dim)))}
        )(G)

    rfeatures = tfgnn.keras.layers.Pool(
        tfgnn.CONTEXT, "mean", node_set_name="atoms")(G)
    logits = tf.keras.layers.Dense(1)(rfeatures)
    return tf.keras.Model(inputs=[input_G], outputs=[logits])

# Compiling the model.
model_g_spec, _ = train_data.element_spec
model = build_model(model_g_spec)

model.compile(tf.keras.optimizers.Adam(),
              loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics = [tf.keras.metrics.BinaryAccuracy(threshold=0.0), 
                         tf.keras.metrics.BinaryCrossentropy(from_logits=True)]
            )

# Fitting the model.
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data = val_data
)

with open('gnn.pickle', 'wb') as fout:
    pickle.dump(model, fout)



