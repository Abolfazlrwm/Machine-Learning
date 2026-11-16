"""A tape-based reverse-mode autograd engine — NumPy only.

Every :class:`Tensor` wraps an ``np.ndarray`` and knows how to propagate
gradients to the tensors that produced it. Building an op records a
backward closure; ``Tensor.backward()`` topologically sorts the graph and
replays the closures in reverse, so composed ops (attention, MLP blocks)
need no hand-derived gradient of their own.

Design notes
------------
* float64 everywhere: the tests check every primitive against central
  differences, and training is small enough that float64 costs nothing.
* Broadcasting is undone by ``_unbroadcast`` — grads are summed back to
  each parent's shape, which is what makes the (1,1,T,T) causal mask and
  shared positional embeddings work transparently.
* Gradients *accumulate* (``+=``) so reusing a tensor in several places
  (residual streams, weight tying) is correct.
"""
from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Sum ``grad`` down to ``shape`` after a broadcasting op."""
    # 1) sum away prepended broadcast dims
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) sum dims that were 1 in the parent
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _reduce_to(param_shape: tuple[int, ...], arr: np.ndarray) -> np.ndarray:
    """Sum leading axes so ``arr`` matches a trailing-aligned parameter shape.

    LayerNorm's gamma/beta live on the normalised (last) axis, so their
    gradient is ``arr`` summed over every leading batch/time axis — e.g.
    (B, T, D) -> (D,). Handles parameters with the same rank as ``arr``.
    """
    if arr.ndim > len(param_shape):
        return arr.sum(axis=tuple(range(arr.ndim - len(param_shape))))
    return arr


class Tensor:
    """A node in the computation graph: ndarray + gradient + parents."""

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        parents: tuple["Tensor", ...] = (),
        backward=None,
        name: str | None = None,
    ):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self.requires_grad = requires_grad
        self._parents = parents
        self._backward = backward  # closure: reads self.grad, writes parents' grads
        self.name = name

    # ------------------------------------------------------------- helpers
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy())

    def _accumulate(self, grad: np.ndarray) -> None:
        if self.grad is None:
            self.grad = np.zeros_like(self.data)
        self.grad += grad

    def zero_grad(self) -> None:
        self.grad = None

    # ------------------------------------------------------------- topsort
    def backward(self, seed: np.ndarray | None = None) -> None:
        """Reverse-mode autodiff from this (scalar or array) tensor."""
        topo: list[Tensor] = []
        visited: set[int] = set()
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in visited:
                    stack.append((p, False))
        self._accumulate(
            np.ones_like(self.data) if seed is None else np.asarray(seed, dtype=np.float64)
        )
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward()
        # Free the graph: backward closures close over their output tensor,
        # so the graph is a reference cycle that only cyclic GC would reap.
        # Training builds a fresh graph every step — without this, memory
        # grows until the process is OOM-killed. Breaking the edges here
        # makes refcounting reclaim every intermediate immediately.
        for node in topo:
            node._backward = None
            node._parents = ()

    # ------------------------------------------------------------- ops
    def __add__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(self.data * 0 + other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad,
                     (self, other))
        if out.requires_grad:
            def _bw():
                self._accumulate(_unbroadcast(out.grad, self.shape))
                other._accumulate(_unbroadcast(out.grad, other.shape))
            out._backward = _bw
        return out

    __radd__ = __add__

    def __mul__(self, other) -> "Tensor":
        other = other if isinstance(other, Tensor) else Tensor(self.data * 0 + other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad,
                     (self, other))
        if out.requires_grad:
            def _bw():
                self._accumulate(_unbroadcast(out.grad * other.data, self.shape))
                other._accumulate(_unbroadcast(out.grad * self.data, other.shape))
            out._backward = _bw
        return out

    __rmul__ = __mul__

    def __neg__(self) -> "Tensor":
        return self * -1.0

    def __sub__(self, other) -> "Tensor":
        return self + (-other if isinstance(other, Tensor) else -other)

    def __matmul__(self, other) -> "Tensor":
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad,
                     (self, other))
        if out.requires_grad:
            def _bw():
                g = out.grad
                self._accumulate(_unbroadcast(np.matmul(g, np.swapaxes(other.data, -1, -2)),
                                              self.shape))
                other._accumulate(_unbroadcast(np.matmul(np.swapaxes(self.data, -1, -2), g),
                                               other.shape))
            out._backward = _bw
        return out

    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,))
        if self.requires_grad:
            out._backward = lambda: self._accumulate(out.grad * (out.data > 0))
        return out

    def tanh(self) -> "Tensor":
        out = Tensor(np.tanh(self.data), self.requires_grad, (self,))
        if self.requires_grad:
            out._backward = lambda: self._accumulate(out.grad * (1.0 - out.data**2))
        return out

    def exp(self) -> "Tensor":
        out = Tensor(np.exp(self.data), self.requires_grad, (self,))
        if self.requires_grad:
            out._backward = lambda: self._accumulate(out.grad * out.data)
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), self.requires_grad, (self,))
        if self.requires_grad:
            out._backward = lambda: self._accumulate(out.grad / self.data)
        return out

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,))
        if self.requires_grad:
            def _bw():
                g = out.grad
                if axis is not None and not keepdims:
                    g = np.expand_dims(g, axis)
                self._accumulate(np.broadcast_to(g, self.shape).copy())
            out._backward = _bw
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)

    def softmax(self, axis: int = -1) -> "Tensor":
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        s = e / e.sum(axis=axis, keepdims=True)
        out = Tensor(s, self.requires_grad, (self,))
        if self.requires_grad:
            def _bw():
                g = out.grad
                inner = (g * s).sum(axis=axis, keepdims=True)
                self._accumulate(s * (g - inner))
            out._backward = _bw
        return out

    def reshape(self, shape) -> "Tensor":
        out = Tensor(self.data.reshape(shape), self.requires_grad, (self,))
        if self.requires_grad:
            out._backward = lambda: self._accumulate(out.grad.reshape(self.shape))
        return out

    def transpose(self, axes) -> "Tensor":
        out = Tensor(self.data.transpose(axes), self.requires_grad, (self,))
        if self.requires_grad:
            inverse = tuple(np.argsort(axes))
            out._backward = lambda: self._accumulate(out.grad.transpose(inverse))
        return out

    # ------------------------------------------------------- fused primitives
    def layer_norm(self, gamma: "Tensor", beta: "Tensor",
                   axis: int = -1, eps: float = 1e-5) -> "Tensor":
        """LayerNorm over ``axis`` with trainable gamma/beta — hand-derived."""
        mu = self.data.mean(axis=axis, keepdims=True)
        var = self.data.var(axis=axis, keepdims=True)
        inv_std = 1.0 / np.sqrt(var + eps)
        x_hat = (self.data - mu) * inv_std
        out = Tensor(x_hat * gamma.data + beta.data,
                     self.requires_grad or gamma.requires_grad or beta.requires_grad,
                     (self, gamma, beta))
        if out.requires_grad:
            def _bw():
                g = out.grad
                ax = axis if axis is not None else -1
                if gamma.requires_grad:
                    gamma._accumulate(_reduce_to(gamma.shape, g * x_hat))
                if beta.requires_grad:
                    beta._accumulate(_reduce_to(beta.shape, g))
                if self.requires_grad:
                    gg = g * gamma.data
                    m1 = gg.mean(axis=ax, keepdims=True)
                    m2 = (gg * x_hat).mean(axis=ax, keepdims=True)
                    self._accumulate(inv_std * (gg - m1 - x_hat * m2))
            out._backward = _bw
        return out

    def cross_entropy(self, targets: np.ndarray) -> "Tensor":
        """Softmax cross-entropy over the last axis, mean over all rows.

        ``targets`` is an int array matching every leading dim of the
        logits. Hand-derived backward: ``(softmax - onehot) / N``.
        """
        x = self.data
        shifted = x - x.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        n = targets.size
        nll = -log_probs.reshape(-1, x.shape[-1])[np.arange(n), targets.reshape(-1)]
        out = Tensor(nll.mean(), self.requires_grad, (self,))
        if self.requires_grad:
            def _bw():
                probs = np.exp(log_probs)
                flat = probs.reshape(-1, x.shape[-1]).copy()
                flat[np.arange(n), targets.reshape(-1)] -= 1.0
                self._accumulate(_unbroadcast((flat / n).reshape(x.shape), self.shape))
            out._backward = _bw
        return out

    def gather(self, indices: np.ndarray) -> "Tensor":
        """Embedding lookup: ``table[indices]`` — backward scatter-adds."""
        out = Tensor(self.data[indices], self.requires_grad, (self,))
        if self.requires_grad:
            def _bw():
                grad_table = np.zeros_like(self.data)
                np.add.at(grad_table, indices, out.grad)
                self._accumulate(grad_table)
            out._backward = _bw
        return out

    # ------------------------------------------------------------- repr
    def __repr__(self) -> str:
        tag = f", name={self.name}" if self.name else ""
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}{tag})"


def parameters(*tensors: Tensor) -> list[Tensor]:
    """Collect all tensors that require grad (a tiny leaf collector)."""
    seen: dict[int, Tensor] = {}
    stack = list(tensors)
    while stack:
        t = stack.pop()
        if id(t) in seen:
            continue
        seen[id(t)] = t
        if t.requires_grad:
            stack.extend(t._parents)
    return [t for t in seen.values() if t.requires_grad]
