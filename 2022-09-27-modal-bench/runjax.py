import subprocess
import time
import modal

import datasets

GPU = False
SEED = 1701
BATCH_SIZE = 1024

image_cpu = (
    modal.Image.conda()
    # .conda_install(["jax"], channels=["conda-forge"])
    .run_commands(["conda install jax -c conda-forge --yes"]).pip_install(
        ["dm-haiku", "optax", "numpy"]
    )
)
image_gpu = (
    modal.Image.conda().run_commands(
        [
            'CONDA_OVERRIDE_CUDA="11.2" conda install jax cuda-nvcc -c conda-forge -c nvidia --yes'
        ]
    )
    # .conda_install(["jax", "cuda-nvcc"], channels=["conda-forge", "nvidia"])
    .pip_install(["dm-haiku", "optax", "numpy"])
)

stub = modal.Stub()
stub.image = image_cpu if not GPU else image_gpu

if stub.is_inside():
    import optax
    import numpy as np
    import numpy.random as npr
    import haiku as hk
    import jax.numpy as jnp
    from jax import grad, jit, nn, random, tree_util


@stub.function(gpu=GPU)
def bench_matmul():
    if GPU:
        subprocess.run(["nvidia-smi"])

    key = random.PRNGKey(SEED)
    key, subkey = random.split(key)
    A = random.normal(subkey, shape=(8000, 64))

    key, subkey = random.split(key)
    B = random.normal(subkey, shape=(64, 8000))

    (A @ B).block_until_ready()

    start_time = time.time()
    for _ in range(500):
        result = (A @ B).block_until_ready()
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print(result)


@stub.function(gpu=GPU)
def train_mnist():
    train_images, train_labels, test_images, test_labels = datasets.mnist()

    def get_model():
        return hk.Sequential(
            [
                hk.Linear(5000),
                nn.relu,
                hk.Linear(3000),
                nn.relu,
                hk.Linear(10),
            ]
        )

    def loss_fn(images, labels):
        mlp = get_model()
        logits = mlp(images)
        return jnp.mean(optax.softmax_cross_entropy(logits, labels))

    def accuracy_fn(images, labels):
        mlp = get_model()
        logits = mlp(images)
        return jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(labels, axis=-1))

    loss_fn_t = hk.transform(loss_fn)
    loss_fn_t = hk.without_apply_rng(loss_fn_t)

    accuracy_fn_t = hk.transform(accuracy_fn)
    accuracy_fn_t = hk.without_apply_rng(accuracy_fn_t)

    def data_stream(images, labels, repeat=True):
        num_data = images.shape[0]
        num_complete_batches, leftover = divmod(num_data, BATCH_SIZE)
        num_batches = num_complete_batches + bool(leftover)

        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_data)
            for i in range(num_batches):
                batch_idx = perm[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
                yield images[batch_idx], labels[batch_idx]
            if not repeat:
                break

    batches = data_stream(train_images, train_labels)
    optimizer = optax.adam(1e-3)

    rng = random.PRNGKey(42)
    params = loss_fn_t.init(rng, *next(batches))
    opt_state = optimizer.init(params)

    for i, (images, labels) in enumerate(batches):
        print(f"Starting next training iteration {i + 1}...")
        start_time = time.time()
        grads = grad(loss_fn_t.apply)(params, images, labels)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        print(f"  time taken: {time.time() - start_time:.2f} seconds")
        if i % 50 == 49:
            print("Computing loss:")
            batch_losses = []
            batch_accs = []
            for (images, labels) in data_stream(test_images, test_labels, repeat=False):
                batch_loss = loss_fn_t.apply(params, images, labels)
                batch_acc = accuracy_fn_t.apply(params, images, labels)
                batch_losses.append(batch_loss)
                batch_accs.append(batch_acc)
            test_loss = np.mean(batch_losses)
            print(f"  test loss = {test_loss:.4f}")
            test_acc = np.mean(batch_accs)
            print(f"  test accuracy = {test_acc:.4f}")


if __name__ == "__main__":
    with stub.run():
        # bench_matmul()
        train_mnist()
