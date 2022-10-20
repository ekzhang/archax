# pipejax

[![PyPI - Version](https://img.shields.io/pypi/v/pipejax.svg)](https://pypi.org/project/pipejax)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pipejax.svg)](https://pypi.org/project/pipejax)

A network-optimized flexible parallelism library for JAX, supporting an automatically optimized mix of pipeline and operator parallelism on accelerated devices.

Currently a work-in-progress. Written in Python, with a focus on performance and composability.

## Installation

**TODO: This command doesn't work until we publish the first version of the library. See the "Development" section instead.**

```
pip install pipejax
```

## Development

This project has dependencies managed by `hatch`. To get started, install Python 3.8+ and Hatch with:

```
pip install --upgrade hatch
```

That's it! Then, you can open a virtual environment with the project. For example, to get a Python shell and start playing with the library:

```
hatch run python

>>> import pipejax
>>> ...
```

You can run tests using `hatch run pytest`, or use `hatch run cov` to measure code coverage in tests.

There are some top-level folders that start with dates. These are used for early one-off experiments and not intended to be part of the library.

### GPU and TPU development

**TODO: Create development / test environments for running the library on GPUs and TPUs.**

## License

`pipejax` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
