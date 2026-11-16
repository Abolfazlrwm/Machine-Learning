"""A tiny decoder-only transformer (GPT) built from autograd primitives.

Pre-LN blocks, multi-head causal self-attention, ReLU FFN, learned
positional embeddings, a final LayerNorm and a linear LM head. No
module framework — every layer is a thin class holding
:class:`src.autograd.Tensor` parameters, and the forward pass is plain
composition of primitive ops, so the reverse-mode engine derives all
gradients end-to-end.
"""
from __future__ import annotations

import numpy as np

from .autograd import Tensor


class Module:
    """Minimal parameter container — the interface every layer implements."""

    def __init__(self) -> None:
        self._params: dict[str, Tensor] = {}

    def register(self, name: str, tensor: Tensor) -> Tensor:
        self._params[f"{type(self).__name__}.{name}"] = tensor
        return tensor

    def parameters(self) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for name, p in self._params.items():
            out[name] = p
        for name, value in self.__dict__.items():
            if isinstance(value, Module) and value is not self:
                out.update({f"{name}.{k}": v for k, v in value.parameters().items()})
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        out.update({f"{name}[{i}].{k}": v for k, v in item.parameters().items()})
        return out

    def state_dict(self) -> dict[str, np.ndarray]:
        return {k: p.data.copy() for k, p in self.parameters().items()}

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        params = self.parameters()
        missing = set(params) - set(state)
        if missing:
            raise KeyError(f"missing keys in state dict: {sorted(missing)}")
        for k, p in params.items():
            p.data[...] = state[k]


class Embedding(Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.vocab, self.dim = vocab, dim
        self.weight = self.register("weight", Tensor(np.random.randn(vocab, dim) * 0.02,
                                                     requires_grad=True))

    def __call__(self, indices: np.ndarray) -> Tensor:
        return self.weight.gather(np.asarray(indices, dtype=np.int64))


class Linear(Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        bound = 1.0 / np.sqrt(dim_in)
        self.weight = self.register(
            "weight", Tensor(np.random.uniform(-bound, bound, (dim_in, dim_out)),
                             requires_grad=True))
        self.bias = self.register("bias", Tensor(np.zeros(dim_out), requires_grad=True))

    def __call__(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = self.register("gamma", Tensor(np.ones(dim), requires_grad=True))
        self.beta = self.register("beta", Tensor(np.zeros(dim), requires_grad=True))

    def __call__(self, x: Tensor) -> Tensor:
        return x.layer_norm(self.gamma, self.beta, axis=-1, eps=self.eps)


class MultiHeadAttention(Module):
    """Causal multi-head self-attention; the mask is a broadcast constant."""

    def __init__(self, dim: int, n_heads: int, max_t: int):
        super().__init__()
        if dim % n_heads:
            raise ValueError("dim must be divisible by n_heads")
        self.n_heads, self.dh = n_heads, dim // n_heads
        self.max_t = max_t
        self.w_q = Linear(dim, dim)
        self.w_k = Linear(dim, dim)
        self.w_v = Linear(dim, dim)
        self.w_o = Linear(dim, dim)

    def _split_heads(self, x: Tensor, batch: int, t: int) -> Tensor:
        return x.reshape((batch, t, self.n_heads, self.dh)).transpose((0, 2, 1, 3))

    def _mask(self, t: int) -> np.ndarray:
        # (1, 1, T, T): future positions get a large negative constant so the
        # softmax zeroes them; -1e9 keeps exp() stable in float64.
        return np.triu(np.full((t, t), -1e9), k=1).reshape(1, 1, t, t)

    def __call__(self, x: Tensor) -> Tensor:
        batch, t = x.shape[0], x.shape[1]
        q = self._split_heads(self.w_q(x), batch, t)
        k = self._split_heads(self.w_k(x), batch, t)
        v = self._split_heads(self.w_v(x), batch, t)
        scores = (q @ k.transpose((0, 1, 3, 2))) * (1.0 / np.sqrt(self.dh))
        att = (scores + self._mask(t)).softmax(axis=-1)
        out = (att @ v).transpose((0, 2, 1, 3)).reshape((batch, t, self.n_heads * self.dh))
        return self.w_o(out)


class MLP(Module):
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or 4 * dim
        self.fc1 = Linear(dim, hidden)
        self.fc2 = Linear(hidden, dim)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())


class Block(Module):
    """Pre-LN transformer block: x + attn(ln(x)), then x + ffn(ln(x))."""

    def __init__(self, dim: int, n_heads: int, max_t: int):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads, max_t)
        self.ln2 = LayerNorm(dim)
        self.mlp = MLP(dim)

    def __call__(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(Module):
    """Decoder-only transformer over char tokens — the whole point."""

    def __init__(self, vocab: int, dim: int = 96, n_layers: int = 2,
                 n_heads: int = 3, max_t: int = 64):
        super().__init__()
        self.vocab, self.dim, self.max_t = vocab, dim, max_t
        self.embed = Embedding(vocab, dim)
        self.pos = Embedding(max_t, dim)
        self.blocks = [Block(dim, n_heads, max_t) for _ in range(n_layers)]
        self.ln_f = LayerNorm(dim)
        self.head = Linear(dim, vocab)

    def __call__(self, indices: np.ndarray) -> Tensor:
        indices = np.asarray(indices, dtype=np.int64)
        batch, t = indices.shape
        if t > self.max_t:
            raise ValueError(f"sequence length {t} exceeds max_t={self.max_t}")
        pos_ids = np.arange(t, dtype=np.int64)
        x = self.embed(indices) + self.pos(pos_ids)  # (B,T,d) + (T,d) broadcast
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))

    def loss(self, indices: np.ndarray) -> Tensor:
        """Next-token cross-entropy for a (B, T+1) token matrix."""
        inputs, targets = indices[:, :-1], indices[:, 1:]
        logits = self(inputs)                       # (B, T, V)
        flat = logits.reshape((-1, self.vocab))     # (B*T, V)
        return flat.cross_entropy(targets.reshape(-1))

    def sample(self, prompt: list[int], n_new: int, temperature: float = 0.8,
               top_k: int | None = None, rng: np.random.Generator | None = None) -> list[int]:
        """Autoregressive sampling — pure NumPy, no graph needed."""
        rng = rng or np.random.default_rng(0)
        tokens = list(prompt)
        for _ in range(n_new):
            window = np.array(tokens[-self.max_t:]).reshape(1, -1)
            logits = self(window).data[0, -1] / max(temperature, 1e-6)
            if top_k:
                cutoff = np.sort(logits)[-top_k]
                logits = np.where(logits < cutoff, -1e9, logits)
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            tokens.append(int(rng.choice(len(probs), p=probs)))
        return tokens
