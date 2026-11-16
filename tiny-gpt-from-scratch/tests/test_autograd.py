"""Numerical gradient checks — the proof the autograd engine is correct.

Every primitive is validated against central differences in float64:
``(f(x+h) - f(x-h)) / 2h`` must match the analytic gradient to ~1e-6.
If a backward closure is wrong, these tests fail loudly — that is the
whole point of writing autograd by hand.
"""
from __future__ import annotations

import numpy as np
from src.autograd import Tensor


def numeric_grad(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient of the scalar-valued f at x."""
    g = np.zeros_like(x)
    flat = x.reshape(-1)
    out = g.reshape(-1)
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + h
        fp = f(x.copy())
        flat[i] = orig - h
        fm = f(x.copy())
        flat[i] = orig
        out[i] = (fp - fm) / (2 * h)
    return g


def _check(op_name: str, build, x: np.ndarray, tol: float = 1e-6, seed: int = 0):
    """Generic check: build(x) -> Tensor; compare d(sum(build * W))/dx."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(build(Tensor(x)).shape)

    def f(arr: np.ndarray) -> float:
        return float((build(Tensor(arr)).data * w).sum())

    x_t = Tensor(x, requires_grad=True)
    (build(x_t) * Tensor(w)).sum().backward()
    analytic = x_t.grad
    numeric = numeric_grad(f, x)
    diff = float(np.abs(analytic - numeric).max())
    assert diff < tol, f"{op_name}: max |analytic-numeric| = {diff:.2e}"
    return diff


class TestPrimitiveGradients:
    def test_add_broadcast(self):
        x = np.random.default_rng(1).standard_normal((4, 3))
        diff = _check("add", lambda t: t + Tensor(np.ones((1, 3))), x)
        assert diff < 1e-9  # linear op: exact

    def test_mul_broadcast(self):
        x = np.random.default_rng(2).standard_normal((4, 3))
        _check("mul", lambda t: t * Tensor(np.full((1, 3), 1.7)), x)

    def test_matmul_2d(self):
        x = np.random.default_rng(3).standard_normal((5, 4))
        w = np.random.default_rng(4).standard_normal((4, 6))
        _check("matmul2d", lambda t: t @ Tensor(w), x)

    def test_matmul_batched_3d(self):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((2, 3, 4))
        w = rng.standard_normal((2, 4, 5))
        _check("matmul3d", lambda t: t @ Tensor(w), x)

    def test_matmul_transposed_rhs(self):
        rng = np.random.default_rng(6)
        x = rng.standard_normal((2, 3, 4))
        w = rng.standard_normal((2, 5, 4))
        _check("matmulT", lambda t: t @ Tensor(w).transpose((0, 2, 1)), x)

    def test_relu(self):
        x = np.random.default_rng(7).standard_normal((6, 4))
        _check("relu", Tensor.relu, x)

    def test_tanh(self):
        x = np.random.default_rng(8).standard_normal((6, 4))
        _check("tanh", Tensor.tanh, x)

    def test_exp_log(self):
        x = np.abs(np.random.default_rng(9).standard_normal((5, 4))) + 0.5
        _check("exp", Tensor.exp, x)
        _check("log", Tensor.log, x)

    def test_sum_and_mean(self):
        x = np.random.default_rng(10).standard_normal((4, 3, 2))
        _check("sum-axis", lambda t: t.sum(axis=1), x)
        _check("mean-axis", lambda t: t.mean(axis=2), x)

    def test_softmax(self):
        x = np.random.default_rng(11).standard_normal((4, 6))
        _check("softmax", lambda t: t.softmax(axis=-1), x)

    def test_softmax_of_scores(self):
        # attention-shaped: softmax over last axis of a 4D score tensor
        rng = np.random.default_rng(12)
        x = rng.standard_normal((2, 2, 3, 3))
        _check("softmax4d", lambda t: t.softmax(axis=-1), x)

    def test_layer_norm(self):
        x = np.random.default_rng(13).standard_normal((4, 5)) * 2 + 1
        w = np.random.default_rng(14).standard_normal((4, 5))
        gamma = Tensor(np.full((5,), 1.3), requires_grad=True)
        beta = Tensor(np.full((5,), -0.2), requires_grad=True)

        def f_x(arr: np.ndarray) -> float:
            return float((Tensor(arr).layer_norm(gamma, beta).data * w).sum())

        def f_gamma(arr: np.ndarray) -> float:
            return float((Tensor(x).layer_norm(Tensor(arr), beta).data * w).sum())

        def f_beta(arr: np.ndarray) -> float:
            return float((Tensor(x).layer_norm(gamma, Tensor(arr)).data * w).sum())

        x_t = Tensor(x, requires_grad=True)
        (x_t.layer_norm(gamma, beta) * Tensor(w)).sum().backward()
        assert np.abs(x_t.grad - numeric_grad(f_x, x)).max() < 1e-6
        assert np.abs(gamma.grad - numeric_grad(f_gamma, gamma.data)).max() < 1e-6
        assert np.abs(beta.grad - numeric_grad(f_beta, beta.data)).max() < 1e-6

    def test_cross_entropy(self):
        rng = np.random.default_rng(15)
        logits = rng.standard_normal((6, 9))
        targets = rng.integers(0, 9, size=6)

        def f(arr):
            return float(Tensor(arr).cross_entropy(targets).data)

        x_t = Tensor(logits, requires_grad=True)
        x_t.cross_entropy(targets).backward()
        numeric = numeric_grad(f, logits)
        assert np.abs(x_t.grad - numeric).max() < 1e-6

    def test_gather_embedding(self):
        rng = np.random.default_rng(16)
        table = rng.standard_normal((7, 4))
        idx = np.array([[0, 3], [6, 1]])
        w = np.random.default_rng(17).standard_normal((2, 2, 4))

        def f(arr):
            return float((Tensor(arr).gather(idx).data * w).sum())

        t_t = Tensor(table, requires_grad=True)
        (t_t.gather(idx) * Tensor(w)).sum().backward()
        assert np.abs(t_t.grad - numeric_grad(f, table)).max() < 1e-6


class TestGraphSemantics:
    def test_reuse_accumulates_gradients(self):
        """y = x*x + x  ->  dy/dx = 2x + 1 (x appears twice)."""
        x = Tensor(np.array([1.5, -2.0]), requires_grad=True)
        y = x * x + x
        y.sum().backward()
        assert np.allclose(x.grad, 2 * x.data + 1)

    def test_residual_style_reuse(self):
        x = Tensor(np.random.default_rng(18).standard_normal((3, 3)),
                   requires_grad=True)
        w = Tensor(np.random.default_rng(19).standard_normal((3, 3)),
                   requires_grad=True)
        y = (x + x @ w) @ w
        y.sum().backward()
        numeric = numeric_grad(
            lambda a: float((((Tensor(a) + Tensor(a) @ w) @ w).data).sum()), x.data)
        assert np.abs(x.grad - numeric).max() < 1e-6

    def test_backward_requires_grad_only_graph_is_fine(self):
        t = Tensor(np.ones((2, 2)))
        assert t.data.sum() == 4  # inference tensors carry no graph overhead

    def test_scalar_seed(self):
        """An explicit seed overrides the ones-vector (weighted loss)."""
        x = Tensor(np.array([2.0, 3.0]), requires_grad=True)
        (x * x).backward(seed=np.array([1.0, 2.0]))
        assert np.allclose(x.grad, 2 * x.data * np.array([1.0, 2.0]))
