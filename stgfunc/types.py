from __future__ import annotations

from abc import abstractmethod
from enum import IntEnum, auto
from typing import Any, Callable, List, NamedTuple, Protocol, SupportsFloat, TypeVar, Union, runtime_checkable

import vapoursynth as vs

T = TypeVar('T')
R = TypeVar('R')
F = TypeVar('F', bound=Callable[..., Any])

SingleOrArr = Union[T, List[T]]
SingleOrArrOpt = Union[SingleOrArr[T], None]


class DebanderFN(Protocol):
    def __call__(
        self, clip: vs.VideoNode, threshold: SingleOrArr[SupportsFloat], *args: Any, **kwargs: Any
    ) -> vs.VideoNode:
        ...


@runtime_checkable
class SupportsString(Protocol):
    """An ABC with one abstract method __str__."""
    @abstractmethod
    def __str__(self) -> str:
        pass


class StrList(List[SupportsString]):
    @property
    def string(self) -> str:
        pass

    @string.getter
    def string(self) -> str:
        return self.to_str()

    def to_str(self) -> str:
        return str(self)

    def __str__(self) -> str:
        from .utils import flatten

        return ' '.join(map(str, flatten(self)))


class MaskCredit(NamedTuple):
    mask: vs.VideoNode
    start_frame: int
    end_frame: int


class Grainer(IntEnum):
    AddGrain = auto()
    AddNoise = auto()
