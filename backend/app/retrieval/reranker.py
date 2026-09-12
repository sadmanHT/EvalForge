from __future__ import annotations

import re
from dataclasses import replace
from typing import Protocol

from app.retrieval.search import RetrievalResult

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


class Reranker(Protocol):
    reranker_id: str
    revision: str

    def rerank(
        self,
        query: str,
        results: tuple[RetrievalResult, ...],
    ) -> tuple[RetrievalResult, ...]: ...


class NoOpReranker:
    reranker_id = "none"
    revision = "phase7-noop-v1"

    def rerank(
        self,
        query: str,
        results: tuple[RetrievalResult, ...],
    ) -> tuple[RetrievalResult, ...]:
        del query
        return tuple(sorted(results, key=lambda item: (-item.score, item.chunk_id)))


class TokenOverlapReranker:
    reranker_id = "evalforge/token-overlap"
    revision = "phase7-token-overlap-v1"

    def rerank(
        self,
        query: str,
        results: tuple[RetrievalResult, ...],
    ) -> tuple[RetrievalResult, ...]:
        query_tokens = set(_TOKEN_RE.findall(query.lower()))
        rescored = []
        for result in results:
            result_tokens = set(_TOKEN_RE.findall(result.text.lower()))
            union = query_tokens | result_tokens
            overlap = len(query_tokens & result_tokens) / len(union) if union else 0.0
            rescored.append(replace(result, score=result.score + overlap))
        return tuple(sorted(rescored, key=lambda item: (-item.score, item.chunk_id)))
