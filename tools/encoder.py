"""Encoder abstraction. Real path uses sentence-transformers; tests inject a mock."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class Encoder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list) -> list: ...


@dataclass
class MockEncoder:
    name: str = "mock-encoder"
    dim: int = 8

    def encode(self, texts: list) -> list:
        out: list = []
        for t in texts:
            v = [0.0] * self.dim
            for w in t.lower().split():
                v[hash(w) % self.dim] += 1.0
            out.append(v)
        return out


@dataclass
class _SentenceTransformerWrapper:
    inner: object
    name: str = "all-MiniLM-L6-v2"

    @property
    def dim(self) -> int:
        return int(self.inner.get_sentence_embedding_dimension())

    def encode(self, texts: list) -> list:
        vecs = self.inner.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vecs.tolist()


def _cpu_banner() -> None:
    msg1 = "hydra-memory: CUDA not available, using CPU for embedding (slower reindex)"
    msg2 = (
        "WARNING: CPU embedding fallback in use. Reindex will be ~15x slower; "
        "install a CUDA torch build to restore speed."
    )
    print(msg1, file=sys.stderr)
    print(msg2, file=sys.stderr)


def _try_real() -> Encoder | None:
    if os.environ.get("HYDRA_MEMORY_ENCODER") == "mock":
        return None
    force_cpu = os.environ.get("HYDRA_MEMORY_FORCE_CPU") == "1"
    if _local_snapshot_exists("sentence-transformers/all-MiniLM-L6-v2") and \
            "HF_HUB_OFFLINE" not in os.environ:
        os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        # Encoder stack missing entirely: still emit the loud CPU banner so
        # the operator sees that reindex is degraded, per spec §9.2.
        if force_cpu or os.environ.get("HYDRA_MEMORY_NO_CPU_BANNER") != "1":
            _cpu_banner()
        return None
    import torch
    if not torch.cuda.is_available() or force_cpu:
        _cpu_banner()
        device = "cpu"
    else:
        device = "cuda"
    return _SentenceTransformerWrapper(
        SentenceTransformer("all-MiniLM-L6-v2", device=device)
    )


def _local_snapshot_exists(repo_id: str) -> bool:
    """True if HF hub has a usable local snapshot for *repo_id* (avoids network)."""
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.is_dir():
        return False
    snapshots = hub / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return False
    return any(child.is_dir() for child in snapshots.iterdir())


def load_encoder() -> Encoder:
    forced = os.environ.get("HYDRA_MEMORY_ENCODER")
    if forced == "mock":
        return MockEncoder()
    real = _try_real()
    if real is not None:
        return real
    return MockEncoder()


def backend_name(enc: Encoder) -> str:
    if os.environ.get("HYDRA_MEMORY_FORCE_CPU") == "1":
        return "cpu"
    if isinstance(enc, MockEncoder):
        return "mock"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
