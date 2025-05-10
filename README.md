# Batch Tensorsolve

<p align="center">
  <a href="https://github.com/34j/batch-tensorsolve/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/batch-tensorsolve/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://batch-tensorsolve.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/batch-tensorsolve.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/batch-tensorsolve">
    <img src="https://img.shields.io/codecov/c/github/34j/batch-tensorsolve.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://github.com/astral-sh/uv">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/batch-tensorsolve/">
    <img src="https://img.shields.io/pypi/v/batch-tensorsolve.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/batch-tensorsolve.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/batch-tensorsolve.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://batch-tensorsolve.readthedocs.io" target="_blank">https://batch-tensorsolve.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/batch-tensorsolve" target="_blank">https://github.com/34j/batch-tensorsolve </a>

---

Batched tensorsolve() for NumPy / PyTorch / JAX. ([numpy/numpy#28099](https://github.com/numpy/numpy/issues/28099))

## Installation

Install this via pip (or your favourite package manager):

```shell
pip install batch-tensorsolve
```

## Usage

```python
import numpy as np
from numpy.testing import assert_allclose

from batch_tensorsolve import btensorsolve

a = np.random.randn(2, 2, 3, 6)
b = np.random.randn(2, 2, 3)
assert_allclose(np.einsum("...ijk,...k->...ij", a, btensorsolve(a, b)), b)
```

## Advanced Usage

It is recommended to explicitly specify the batch axes, as the desired shape will be ambiguous if axes of size 1 are present.

```python
import numpy as np

from batch_tensorsolve import btensorsolve

a = np.random.randn(2, 1, 2, 2)
b = np.random.randn(2, 1, 2)
# 2 possibilities:
assert btensorsolve(a, b, num_batch_axes=1).shape == (2, 2) # 1st axis is batch
assert btensorsolve(a, b, num_batch_axes=2).shape == (2, 1, 2) # 1st and 2nd axes are batch
```

Broadcasting-like behavior is also supported:

```python
import numpy as np
from numpy.testing import assert_allclose

from batch_tensorsolve import btensorsolve

a = np.random.randn(1, 2, 3, 6) # -> (2, 2, 3, 6)
b = np.random.randn(2, 1, 1) # -> (2, 2, 3)
left = np.einsum("...ijk,...k->...ij", a, btensorsolve(a, b))
assert_allclose(left, np.broadcast_to(b, left.shape))
```

Note that broadcasting (repeating) `a` for non-batch axes will result in `numpy.linalg.LinAlgError: Singular matrix` because the matrix representation of `a` has duplicate rows.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-orange.json)](https://github.com/copier-org/copier)

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
