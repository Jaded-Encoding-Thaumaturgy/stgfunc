from __future__ import annotations

from math import floor
from functools import partial, wraps
from typing import Any, Callable, Iterable, List, Sequence, cast, overload

import vapoursynth as vs

from ..types import F, T

core = vs.core


def mod4(x: float) -> int:
    return 16 if x < 16 else floor(x / 4 + 0.5) * 4



@overload
def disallow_variable_format(*, only_first: bool = False) -> Callable[[F], F]:
    ...


@overload
def disallow_variable_format(function: F | None = None, /) -> F:
    ...


def disallow_variable_format(function: F | None = None, /, *, only_first: bool = False) -> Callable[[F], F] | F:
    """Function decorator that raises an exception if input clips have variable format.
        :param function:    Function to wrap.
        :param only_first:  Whether to check only the first argument or not.

        :return:  Wrapped function.
    """

    if function is None:
        return cast(Callable[[F], F], partial(disallow_variable_format, only_first=only_first))

    @wraps(function)
    def _check(*args: Any, **kwargs: Any) -> Any:
        assert function
        if (only_first and args[0].format is None) or any([
            a.format is None for a in args if isinstance(a, vs.VideoNode)
        ]):
            raise ValueError('Variable-format clips not supported.')

        return function(*args, **kwargs)

    return cast(F, _check)


@overload
def disallow_variable_resolution(*, only_first: bool = False) -> Callable[[F], F]:
    ...


@overload
def disallow_variable_resolution(function: F | None = None, /) -> F:
    ...


def disallow_variable_resolution(function: F | None = None, /, *, only_first: bool = False) -> Callable[[F], F] | F:
    """Function decorator that raises an exception if input clips have variable resolution.
        :param function:    Function to wrap.
        :param only_first:  Whether to check only the first argument or not.

        :return:  Wrapped function.
    """

    if function is None:
        return cast(Callable[[F], F], partial(disallow_variable_resolution, only_first=only_first))

    @wraps(function)
    def _check(*args: Any, **kwargs: Any) -> Any:
        assert function
        if ((only_first and not all([args[0].width, args[0].height])) or not all([
            c for s in [[a.width, a.height] for a in args if isinstance(a, vs.VideoNode)] for c in s
        ])):
            raise ValueError('Variable-resolution clips not supported.')

        return function(*args, **kwargs)

    return cast(F, _check)


def to_arr(array: Sequence[T] | T) -> List[T]:
    return list(array if (type(array) in {list, tuple, range, zip, set, map, enumerate}) else [array])  # type: ignore


def flatten(items: Iterable[T]) -> Iterable[T]:
    for val in items:
        if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
            for sub_x in flatten(val):
                yield sub_x
        else:
            yield val  # type: ignore


def remove_chars(string: str, chars: str = '') -> str:
    return string.translate({ord(char): None for char in chars})
