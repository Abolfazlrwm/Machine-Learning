"""Adam optimizer operating on autograd Tensors.

Standard Kingma & Ba updates (m, v, bias correction) applied in-place to
each parameter's ``.data`` using its accumulated ``.grad``.
"""
from __future__ import annotations

import numpy as np

from .autograd import Tensor


class Adam:
    def __init__(self, params: dict[str, Tensor], lr: float = 3e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.v = {k: np.zeros_like(p.data) for k, p in params.items()}
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for key, p in self.params.items():
            if p.grad is None:
                continue
            g = p.grad
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g * g
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params.values():
            p.zero_grad()
