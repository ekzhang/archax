from archax import add

import numpy as np
import jax
import jax.numpy as jnp


def test_add():
    assert add(1, 2) == 3


def test_jax_works():
    assert np.add(1, 2) == 3
    np.testing.assert_allclose(
        jax.grad(jnp.sum)(jnp.array([1.0, 2, 3])),
        np.array([1.0, 1, 1]),
    )
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda x: x[0] - x[1]))(jnp.array([1.0, 2, 3])),
        np.array([1.0, -1, 0]),
    )
