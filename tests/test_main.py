import numpy as np
import pytest
from numpy.testing import assert_allclose

from batch_tensorsolve import (
    AmbiguousBatchAxesWarning,
    broadcast_without_repeating,
    btensorsolve,
)


def test_btensorsolve():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (1, 1, 3, 4, 1, 1, 2, 2, 3, 2, 6))
    b = rng.normal(0, 1, (2, 1, 3, 1, 5, 1, 1, 2, 3))
    with pytest.warns(AmbiguousBatchAxesWarning):
        sol = btensorsolve(a, b)
    assert sol.shape == (2, 1, 3, 4, 5, 1, 2, 6)
    asol = np.einsum("...ijklm,...lm->...ijk", a, sol)
    assert_allclose(asol, np.broadcast_to(b, asol.shape))


def test_broadcast_without_repeating():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (3, 1))
    b = rng.normal(0, 1, (2, 1, 4))
    a, b = broadcast_without_repeating(a, b)
    assert_allclose(a, np.broadcast_to(a, (1, 3, 1)))
    assert_allclose(b, np.broadcast_to(b, (2, 1, 4)))


def test_broadcast_without_repeating_error():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (2,))
    b = rng.normal(0, 1, (3,))
    with pytest.raises(ValueError):
        broadcast_without_repeating(a, b)
