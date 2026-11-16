"""tiny-gpt-from-scratch — a Persian char-level GPT with zero ML frameworks.

The whole stack is NumPy: a tape-based reverse-mode autograd engine
(``src.autograd``), the transformer modules built from those primitives
(``src.model``), an Adam optimizer, a char-level tokenizer, and a
training / sampling CLI. No PyTorch, no JAX, no TF — every gradient is
either hand-derived in a primitive's backward closure or composed
automatically through the graph, and the tests verify every primitive
against central-difference numerical gradients.
"""
