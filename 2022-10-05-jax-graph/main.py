import jax.numpy as jnp
from jax import grad, xla_computation


def tanh(x):
    y = jnp.exp(-2.0 * x)
    return (1.0 - y) / (1.0 + y)


def lfn(x):
    return jnp.log(tanh(x).sum())


def dlfn(x):
    return grad(lfn)(x)


z = xla_computation(dlfn)(jnp.ones(256))

with open("t.hlo", "w") as f:
    f.write(z.as_hlo_text())

with open("t.dot", "w") as f:
    f.write(z.as_hlo_dot_graph())
