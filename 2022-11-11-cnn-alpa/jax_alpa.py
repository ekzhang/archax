import haiku as hk
import jax.numpy as jnp
import numpy as np
import optax
import time
import alpa
import numpy.random as npr
import linecache
import os 
import tracemalloc

from jax import grad, nn, jit, tree_map
from jax import random as jax_random
from keras.datasets import mnist
from jax.lib import xla_bridge

# Checking if alpa recognizes your GPU
assert xla_bridge.get_backend().platform == 'gpu'

# Python profiler (insert citation here later)
def display_top(snapshot, key_type='lineno', limit=7):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    #print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        #print("#%s: %s:%s: %.1f KiB"
        #      % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        #if line:
        #    print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        #print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    #print("Total allocated size: %.1f KiB" % (total / 1024))
    return total

CLASSES, LR, BATCH_SIZE = 10, 0.001, 1024
IMG_ROWS, IMG_COLS = 28, 28
RNG = jax_random.PRNGKey(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], IMG_ROWS, IMG_COLS, 1)
X_test = X_test.reshape(X_test.shape[0], IMG_ROWS, IMG_COLS, 1)
X_train, X_test, = X_train.astype('float32'), X_test.astype('float32')

def data_stream(images, labels, repeat=True):
    num_data = images.shape[0]
    num_complete_batches, leftover = divmod(num_data, BATCH_SIZE)
    num_batches = num_complete_batches + bool(leftover)

    rng = npr.RandomState(0)
    while True:
        perm = rng.permutation(num_data)
        for i in range(num_batches):
            batch_idx = perm[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            yield images[batch_idx], labels[batch_idx]
        if not repeat:
            break

class CNN(hk.Module):
    def __init__(self):
        super().__init__(name="CNN")
        self.conv1 = hk.Conv2D(output_channels=32, kernel_shape=(3, 3), padding="SAME")
        self.conv2 = hk.Conv2D(output_channels=16, kernel_shape=(3, 3), padding="SAME")
        self.flatten = hk.Flatten()
        self.linear = hk.Linear(CLASSES)

    def __call__(self, x_batch):
        fp = self.conv1(x_batch)
        fp = nn.relu(fp)
        fp = self.conv2(fp)
        fp = nn.relu(fp)
        fp = self.flatten(fp)
        fp = self.linear(fp)
        fp = nn.softmax(fp)
        return fp

def ConvNet(x):
    return CNN()(x)

@jit
def loss_fn(weights, x, y):
    logits = cnn.apply(weights, RNG, x)
    y = nn.one_hot(y, num_classes=CLASSES)
    return jnp.mean(optax.softmax_cross_entropy(logits, y))

@jit
def acc_loss_fn(weights, x, y):
    logits = cnn.apply(weights, RNG, x)
    y = nn.one_hot(y, num_classes=CLASSES)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(y, axis=-1))
    loss = jnp.mean(optax.softmax_cross_entropy(logits, y))
    return acc, loss

batches = data_stream(X_train, y_train)
optimizer = optax.adam(learning_rate=LR)

rng, cnn = hk.PRNGSequence(jax_random.PRNGKey(243)), hk.transform(ConvNet)
weights = cnn.init(next(rng), next(batches)[0])
opt_state = optimizer.init(weights)

step_count = 0
MAX_STEPS = 20

@alpa.parallelize
def train_step(weights, opt_state, x_batch, y_batch):
    grads = grad(loss_fn)(weights, x_batch, y_batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    weights = optax.apply_updates(weights, updates)
    return weights, opt_state

start_time = time.time()
tracemalloc.start()
memory_lst, time_lst = [], []
for i, (x_batch, y_batch) in enumerate(batches):
    weights, opt_state = train_step(weights, opt_state, x_batch, y_batch)

    if (i + 1) % 50 == 0:
        time_lst.append(time.time() - start_time)
        start_time = time.time()
        batch_losses = []
        batch_accs = []
        for x_t_batch, y_t_batch in data_stream(X_test, y_test, repeat=False):
            batch_acc, batch_loss = acc_loss_fn(weights, x_t_batch, y_t_batch)
            batch_losses.append(batch_loss)
            batch_accs.append(batch_acc)
        loss, acc = np.mean(batch_losses), np.mean(batch_accs)
        print(f"Loss: {loss} - Accuracy: {acc}")
        step_count += 1
        snapshot = tracemalloc.take_snapshot()
        mem_total = display_top(snapshot)
        memory_lst.append(mem_total)
    if step_count == MAX_STEPS:
        break
print(f"Memory: {round(np.mean(memory_lst) / 1000000, 4)}, Time: {round(np.mean(time_lst, 4))}")






    

