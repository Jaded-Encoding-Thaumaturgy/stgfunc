from __future__ import annotations

import vapoursynth as vs
from abc import abstractmethod
from functools import partial, wraps
from typing import runtime_checkable, Protocol, List, Union, Any, TypeVar, Callable, cast, overload

F = TypeVar('F', bound=Callable[..., Any])
T = TypeVar('T')
R = TypeVar('R')

SingleOrArr = Union[T, List[T]]
SingleOrArrOpt = Union[SingleOrArr[T], None]


@runtime_checkable
class SupportsString(Protocol):
    """An ABC with one abstract method __str__."""
    @abstractmethod
    def __str__(self) -> str:
        pass


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
