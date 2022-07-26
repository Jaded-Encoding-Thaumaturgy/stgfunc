from __future__ import annotations

from enum import IntEnum, auto
from typing import Any, NamedTuple, Protocol, SupportsFloat, Tuple, TypeAlias, TypeVar

import vapoursynth as vs
from vsexprtools.types import SingleOrArr

T = TypeVar('T')
R = TypeVar('R')


Range: TypeAlias = int | Tuple[int | None, int | None] | None


class DebanderFN(Protocol):
    def __call__(
        self, clip: vs.VideoNode, threshold: SingleOrArr[SupportsFloat], *args: Any, **kwargs: Any
    ) -> vs.VideoNode:
        ...


class MaskCredit(NamedTuple):
    mask: vs.VideoNode
    start_frame: int
    end_frame: int


class Grainer(IntEnum):
    AddGrain = auto()
    AddNoise = auto()
