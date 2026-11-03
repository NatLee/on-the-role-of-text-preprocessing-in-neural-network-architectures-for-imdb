"""Word ids: the most frequent training words get the smallest ids.

Ids 0 and 1 are reserved, as in the original code: 0 pads short reviews, 1 stands for every word that
is not in the vocabulary.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

PAD = 0
UNK = 1
RESERVED = 2


@dataclass(frozen=True)
class Vocabulary:
    words: tuple[str, ...]  # words[i] has id i + RESERVED
    index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", {word: i + RESERVED for i, word in enumerate(self.words)})

    @classmethod
    def build(cls, docs: Iterable[list[str]], size: int) -> Vocabulary:
        """Keep the ``size - 2`` most frequent words, so that every id is below ``size``.

        Ties are broken alphabetically, which makes the ids the same on every run and platform.
        """
        if size <= RESERVED:
            raise ValueError(f"vocabulary size must be larger than {RESERVED}, got {size}")
        counts = Counter(word for doc in docs for word in doc)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return cls(tuple(word for word, _ in ranked[: size - RESERVED]))

    def __len__(self) -> int:
        """Number of ids, reserved ones included: the row count of the embedding matrix."""
        return RESERVED + len(self.words)

    def encode(self, doc: list[str]) -> list[int]:
        return [self.index.get(word, UNK) for word in doc]

    def encode_all(self, docs: Iterable[list[str]]) -> list[list[int]]:
        return [self.encode(doc) for doc in docs]

    def coverage(self, docs: Iterable[list[str]]) -> float:
        """Share of tokens that are in the vocabulary (the rest become UNK)."""
        total = known = 0
        for doc in docs:
            total += len(doc)
            known += sum(word in self.index for word in doc)
        return known / total if total else 0.0
