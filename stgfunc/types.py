from __future__ import annotations

import vapoursynth as vs
from abc import abstractmethod
from functools import partial, wraps
from typing import runtime_checkable, Protocol, List, Union, Any, TypeVar, Callable, cast

T = TypeVar('T')
R = TypeVar('R')
F = TypeVar('F', bound=Callable)

SingleOrArr = Union[T, List[T]]
SingleOrArrOpt = Union[SingleOrArr[T], None]


@runtime_checkable
class SupportsString(Protocol):
    """An ABC with one abstract method __str__."""
    @abstractmethod
    def __str__(self) -> str:
        pass


def disallow_variable_format(function: F, /, *, only_first: bool = False) -> F:
    """Function decorator that raises an exception if input clips have variable format.
        :param function:    Function to wrap.
        :param only_first:  Whether to check only the first argument or not.

        :return:  Wrapped function.
    """

    if function is None:
        return partial(disallow_variable_format, only_first=only_first)

    @wraps(function)
    def _check(*args: Any, **kwargs: Any) -> Any:
        if (only_first and args[0].format is None) or any([
            a.format is None for a in args if isinstance(a, vs.VideoNode)
        ]):
            raise ValueError('Variable-format clips not supported.')

        return function(*args, **kwargs)

    return cast(F, _check)


def disallow_variable_resolution(function: F, only_first: bool = False) -> F:
    """Function decorator that raises an exception if input clips have variable resolution.
        :param function:    Function to wrap.
        :param only_first:  Whether to check only the first argument or not.

        :return:  Wrapped function.
    """

    @wraps(function)
    def _check(*args: Any, **kwargs: Any) -> Any:
        if ((only_first and not all(args[0].width, args[0].height)) or any([
            not all(a.width, a.height) for a in args if isinstance(a, vs.VideoNode)
        ])):
            raise ValueError('Variable-resolution clips not supported.')

        return function(*args, **kwargs)

    return cast(F, _check)
