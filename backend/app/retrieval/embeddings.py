from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str
    revision: str
    dimension: int
    preprocessing: dict[str, Any]


class HashEmbeddingAdapter:
    """Deterministic CPU-only feature-hash embedding used as the Phase 07 baseline.

    The adapter is intentionally dependency-free so indexing and retrieval can be reproduced
    in CI and clean Compose environments. The exact algorithm identity is carried by the
    model_id/revision pair in the KB config and manifest.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        if config.dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.config = config

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed(self, text: str) -> list[float]:
        normalized = text.lower() if self.config.preprocessing.get("lowercase", True) else text
        tokens = _TOKEN_RE.findall(normalized)
        vector = [0.0] * self.dimension
        features = list(tokens)
        if self.config.preprocessing.get("include_bigrams", True):
            features.extend(f"{left}::{right}" for left, right in zip(tokens, tokens[1:]))

        for feature in features:
            digest = hashlib.sha256(feature.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self.dimension
            sign = -1.0 if digest[8] & 1 else 1.0
            vector[bucket] += sign

        if self.config.preprocessing.get("l2_normalize", True):
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
        return vector


def validate_embedding_dimension(vector: list[float], expected_dimension: int) -> None:
    if len(vector) != expected_dimension:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected_dimension}, got {len(vector)}"
        )
