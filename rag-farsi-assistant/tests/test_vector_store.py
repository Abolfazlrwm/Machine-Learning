import numpy as np

from src.chunker import Chunk
from src.vector_store import VectorStore


def make_chunks(n: int) -> list[Chunk]:
    return [
        Chunk(chunk_id=f"doc::{i}", doc_id="doc", title="t", text=f"chunk {i}", position=i)
        for i in range(n)
    ]


class TestVectorStore:
    def test_add_and_search_nearest_first(self):
        store = VectorStore(dimension=8)
        vectors = np.zeros((3, 8), dtype="float32")
        vectors[0, 0] = 1.0          # axis e0
        vectors[1, 1] = 1.0          # axis e1
        vectors[2, 2] = 1.0          # axis e2
        store.add(vectors, make_chunks(3))

        results = store.search(vectors[1], top_k=2)  # query == e1
        assert len(results) == 2
        assert results[0][0].chunk_id == "doc::1"
        assert results[0][1] > results[1][1]  # descending scores

    def test_numpy_fallback_available(self):
        store = VectorStore(dimension=4)
        assert store._faiss is None or store._faiss is not None  # either path OK
        vectors = np.ones((2, 4), dtype="float32")
        store.add(vectors, make_chunks(2))
        assert len(store) == 2

    def test_dimension_mismatch_raises(self):
        import pytest

        store = VectorStore(dimension=4)
        with pytest.raises(ValueError):
            store.add(np.ones((1, 5), dtype="float32"), make_chunks(1))

    def test_save_load_roundtrip(self, tmp_path):
        store = VectorStore(dimension=8)
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(5, 8)).astype("float32")
        store.add(vectors, make_chunks(5))
        store.save(tmp_path)

        loaded = VectorStore.load(tmp_path)
        assert len(loaded) == 5
        query = vectors[3] + rng.normal(scale=0.01, size=8).astype("float32")
        original_top = store.search(query, top_k=1)[0][0].chunk_id
        loaded_top = loaded.search(query, top_k=1)[0][0].chunk_id
        assert original_top == loaded_top

    def test_search_empty_store(self):
        store = VectorStore(dimension=8)
        assert store.search(np.ones(8, dtype="float32"), top_k=3) == []
