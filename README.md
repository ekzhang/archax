# archax

**Experiments in multi-architecture parallelism for deep learning with JAX.**

![Example JAX computation graph](https://gist.githubusercontent.com/ekzhang/146eb9d1a09fd264da9f6a177e970146/raw/a8165a2a1e1da4a7b6a75eccb89f75cf191430c8/optimized_hlo.svg)

What if we could create a new kind of multi-architecture parallelism library for deep learning compilers, supporting expressive frontends like JAX? This would optimize a mix of pipeline and operator parallelism on accelerated devices. Use both CPU, GPU, and/or TPU in the same program, and automatically interleave between them.

Experiments are given in this repository, dated and annotated with brief descriptions.

## License

All code and notebooks in this repository are distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
