import numpy as np
from numpy.testing import assert_allclose

from batch_tensorsolve import btensorsolve


def test_btensorsolve():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, (1, 1, 2, 2, 3, 2, 6))
    b = rng.normal(0, 1, (2, 1, 1, 2, 3))
    sol = btensorsolve(a, b)
    asol = np.einsum("...ijklm,...lm->...ijk", a, sol)
    assert_allclose(asol, np.broadcast_to(b, asol.shape))
