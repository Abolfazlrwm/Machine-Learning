"""Model-level tests: shapes, causal masking, optimisation sanity, determinism."""
from __future__ import annotations

import numpy as np
import pytest
from src.data import random_batch
from src.model import TinyGPT
from src.optimizer import Adam
from src.tokenizer import CharTokenizer


def tiny_model(vocab: int = 11, dim: int = 24, layers: int = 1, heads: int = 2,
               max_t: int = 16) -> TinyGPT:
    np.random.seed(0)
    return TinyGPT(vocab, dim=dim, n_layers=layers, n_heads=heads, max_t=max_t)


class TestShapesAndMasking:
    def test_logits_shape(self):
        model = tiny_model()
        idx = np.random.default_rng(1).integers(0, 11, size=(3, 16))
        logits = model(idx)
        assert logits.shape == (3, 16, 11)

    def test_sequence_too_long_raises(self):
        model = tiny_model(max_t=8)
        with pytest.raises(ValueError):
            model(np.zeros((1, 9), dtype=np.int64))

    def test_causal_masking(self):
        """Changing a *future* token must not change earlier logits."""
        model = tiny_model()
        base = np.random.default_rng(2).integers(0, 11, size=(1, 16))
        perturbed = base.copy()
        perturbed[0, -1] = (perturbed[0, -1] + 1) % 11
        l1 = model(base).data
        l2 = model(perturbed).data
        assert np.allclose(l1[0, :15], l2[0, :15])
        assert not np.allclose(l1[0, 15], l2[0, 15])

    def test_heads_must_divide_dim(self):
        with pytest.raises(ValueError):
            TinyGPT(11, dim=10, n_heads=3)


class TestTraining:
    def test_overfit_toy_corpus(self):
        """Adam + hand-derived gradients must actually learn — fast."""
        np.random.seed(3)
        text = "یادگیری ماشین از مثال‌ها الگو استخراج می‌کند. " * 8
        tok = CharTokenizer.from_text(text)
        data = np.array(tok.encode(text), dtype=np.int64)
        model = TinyGPT(tok.vocab_size, dim=24, n_layers=1, n_heads=2, max_t=16)
        opt = Adam(model.parameters(), lr=1e-2)
        rng = np.random.default_rng(4)
        losses = []
        for _ in range(150):
            batch = random_batch(data, 16, 8, rng)
            loss = model.loss(batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.data))
        assert losses[-1] < 0.5, f"did not converge: {losses[-1]:.3f}"
        assert losses[-1] < losses[0] / 2

    def test_gradients_flow_to_every_parameter(self):
        model = tiny_model()
        idx = np.random.default_rng(5).integers(0, 11, size=(2, 17))
        model.loss(idx).backward()
        missing = [k for k, p in model.parameters().items() if p.grad is None]
        assert not missing, f"no gradient reached: {missing}"

    def test_param_count_matches_expectation(self):
        model = tiny_model(vocab=11, dim=24, layers=1, heads=2, max_t=16)
        n = sum(p.data.size for p in model.parameters().values())
        # embed 11*24 + pos 16*24 + block(ln 48 + attn 4*576 + mlp 2*24*96+96+24)
        # + final ln 48 + head 24*11 + 11
        assert 24 * (11 + 16) < n < 12_000


class TestSamplingAndPersistence:
    def test_sampling_is_seeded_deterministic(self):
        model = tiny_model()
        prompt = [1, 2, 3]
        a = model.sample(prompt, n_new=10, rng=np.random.default_rng(42))
        b = model.sample(prompt, n_new=10, rng=np.random.default_rng(42))
        assert a == b

    def test_top_k_never_leaves_the_alphabet(self):
        model = tiny_model()
        ids = model.sample([0], n_new=20, top_k=3, rng=np.random.default_rng(7))
        assert all(0 <= i < 11 for i in ids)

    def test_state_dict_round_trip(self):
        model = tiny_model()
        idx = np.random.default_rng(6).integers(0, 11, size=(1, 16))
        before = model(idx).data.copy()
        clone = TinyGPT(11, dim=24, n_layers=1, n_heads=2, max_t=16)
        clone.load_state_dict(model.state_dict())
        assert np.allclose(clone(idx).data, before)

    def test_loss_decreases_on_one_repeated_batch(self):
        """Single-batch sanity: one Adam step reduces the loss."""
        np.random.seed(8)
        model = tiny_model()
        batch = np.random.default_rng(9).integers(0, 11, size=(4, 17))
        loss1 = model.loss(batch)
        opt = Adam(model.parameters(), lr=1e-2)
        opt.zero_grad()
        loss1.backward()
        opt.step()
        loss2 = model.loss(batch)
        assert float(loss2.data) < float(loss1.data)


class TestTokenizer:
    def test_round_trip(self):
        tok = CharTokenizer.from_text("سلام دنیا abc")
        ids = tok.encode("سلام abc")
        assert tok.decode(ids) == "سلام abc"

    def test_unknown_chars_dropped(self):
        tok = CharTokenizer.from_text("abc")
        assert tok.encode("abcxy") == [0, 1, 2]

    def test_vocab_size(self):
        assert CharTokenizer.from_text("aabbcc").vocab_size == 3
