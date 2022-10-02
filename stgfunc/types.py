from __future__ import annotations

from typing import Any, NamedTuple, Protocol, SupportsFloat

from vstools import SingleOrArr, vs

__all__ = [
    'DebanderFN',
    'MaskCredit'
]


class DebanderFN(Protocol):
    def __call__(
        self, clip: vs.VideoNode, threshold: SingleOrArr[SupportsFloat], *args: Any, **kwargs: Any
    ) -> vs.VideoNode:
        ...


class MaskCredit(NamedTuple):
    mask: vs.VideoNode
    start_frame: int
    end_frame: int
