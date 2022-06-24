from __future__ import annotations

from functools import partial
from math import ceil, floor
from typing import Iterable, List, Sequence

import vapoursynth as vs

from ..types import T

core = vs.core


def cround(x: float) -> int:
    return floor(x + 0.5) if x > 0 else ceil(x - 0.5)


def clamp(val: float, min_val: float, max_val: float) -> float:
    return min_val if val < min_val else max_val if val > max_val else val


def mod_x(val: int | float, x: int) -> int:
    return max(x * x, cround(val / x) * x)


mod2 = partial(mod_x, x=2)

mod4 = partial(mod_x, x=4)


def to_arr(array: Sequence[T] | T) -> List[T]:
    return list(
        array if (type(array) in {list, tuple, range, zip, set, map, enumerate}) else [array]  # type: ignore
    )


def flatten(items: Iterable[T]) -> Iterable[T]:
    for val in items:
        if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
            for sub_x in flatten(val):
                yield sub_x
        else:
            yield val  # type: ignore


def remove_chars(string: str, chars: str = '') -> str:
    return string.translate({ord(char): None for char in chars})
