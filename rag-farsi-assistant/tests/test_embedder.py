import numpy as np

from src.embedder import FakeHashEmbedder, create_embedder


class TestFakeHashEmbedder:
    def test_deterministic(self):
        e1 = FakeHashEmbedder(dimension=64).embed(["سلام دنیا"])
        e2 = FakeHashEmbedder(dimension=64).embed(["سلام دنیا"])
        np.testing.assert_array_equal(e1, e2)

    def test_normalized(self):
        vectors = FakeHashEmbedder(dimension=64).embed(["یک", "دو", "سه"])
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)

    def test_shape_and_dtype(self):
        embedder = FakeHashEmbedder(dimension=128)
        vectors = embedder.embed(["a", "b", "c"])
        assert vectors.shape == (3, 128)
        assert vectors.dtype == np.float32

    def test_different_texts_differ(self):
        embedder = FakeHashEmbedder(dimension=64)
        v1, v2 = embedder.embed(["شبکه عصبی", "سیستم توصیه‌گر"])
        assert v1 @ v2 < 0.99  # not identical

    def test_similar_texts_share_words(self):
        embedder = FakeHashEmbedder(dimension=64)
        v1, v2 = embedder.embed(["مکانیزم توجه مهم است", "مکانیزم توجه"])
        assert v1 @ v2 > 0.5  # share tokens -> high similarity


def test_factory_unknown_backend():
    import pytest

    with pytest.raises(ValueError, match="Unknown embedding backend"):
        create_embedder("nope", "any-model")


def test_factory_fake_backend():
    embedder = create_embedder("fake", "ignored", dimension=32)
    assert embedder.dimension == 32
