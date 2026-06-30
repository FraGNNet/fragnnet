"""dgl_compat.py

Safe compatibility shim for optional DGL usage. This module centralizes the
try/except import logic so other modules can `from fragnnet.utils.dgl_compat
import dgl, DGL_AVAILABLE, DGLGraph, dgl_function, dgl_nn, expand_as_pair`
and handle the absence of DGL gracefully.
"""

try:
    import dgl  # type: ignore

    DGL_AVAILABLE = True
    DGLGraph = getattr(dgl, "DGLGraph", type("DGLGraph", (), {}))
    dgl_function = getattr(dgl, "function", None)
    dgl_nn = getattr(dgl, "nn", None)
    expand_as_pair = getattr(dgl_nn, "expand_as_pair", None) if dgl_nn is not None else None
except Exception:
    dgl = None  # type: ignore
    DGL_AVAILABLE = False
    DGLGraph = type("DGLGraph", (), {})
    dgl_function = None
    dgl_nn = None
    expand_as_pair = None

__all__ = [
    "dgl",
    "DGL_AVAILABLE",
    "DGLGraph",
    "dgl_function",
    "dgl_nn",
    "expand_as_pair",
]
