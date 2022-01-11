from typing import Tuple


class EasingBase:
    limit: Tuple[float, float]
    start: int
    end: int
    duration: int

    def __init__(self, start: int = ..., end: int = ..., duration: int = ...) -> None: ...
    def func(self, t: float) -> float: ...
    def ease(self, alpha: float) -> float: ...
    def __call__(self, alpha: float) -> float: ...


class LinearInOut(EasingBase):
    def func(self, t: float) -> float: ...


class QuadEaseInOut(EasingBase):
    def func(self, t: float) -> float: ...


class QuadEaseIn(EasingBase):
    def func(self, t: float) -> float: ...


class QuadEaseOut(EasingBase):
    def func(self, t: float) -> float: ...


class CubicEaseIn(EasingBase):
    def func(self, t: float) -> float: ...


class CubicEaseOut(EasingBase):
    def func(self, t: float) -> float: ...


class CubicEaseInOut(EasingBase):
    def func(self, t: float) -> float: ...


class QuarticEaseIn(EasingBase):
    def func(self, t: float) -> float: ...


class QuarticEaseOut(EasingBase):
    def func(self, t: float) -> float: ...


class QuarticEaseInOut(EasingBase):
    def func(self, t: float) -> float: ...


class QuinticEaseIn(EasingBase):
    def func(self, t: float) -> float: ...


class QuinticEaseOut(EasingBase):
    def func(self, t: float) -> float: ...


class QuinticEaseInOut(EasingBase):
    def func(self, t: float) -> float: ...


class SineEaseIn(EasingBase):
    def func(self, t: float) -> float: ...


class SineEaseOut(EasingBase):
    def func(self, t: float) -> float: ...


class SineEaseInOut(EasingBase):
    def func(self, t: float) -> float: ...


class CircularEaseIn(EasingBase):
    def func(self, t: float) -> float: ...


class CircularEaseOut(EasingBase):
    def func(self, t: float) -> float: ...


class CircularEaseInOut(EasingBase):
    def func(self, t: float) -> float: ...


class ExponentialEaseIn(EasingBase):
    def func(self, t: float) -> float: ...


class ExponentialEaseOut(EasingBase):
    def func(self, t: float) -> float: ...


class ExponentialEaseInOut(EasingBase):
    def func(self, t: float) -> float: ...


class ElasticEaseIn(EasingBase):
    def func(self, t: float) -> float: ...


class ElasticEaseOut(EasingBase):
    def func(self, t: float) -> float: ...


class ElasticEaseInOut(EasingBase):
    def func(self, t: float) -> float: ...


class BackEaseIn(EasingBase):
    def func(self, t: float) -> float: ...


class BackEaseOut(EasingBase):
    def func(self, t: float) -> float: ...


class BackEaseInOut(EasingBase):
    def func(self, t: float) -> float: ...


class BounceEaseIn(EasingBase):
    def func(self, t: float) -> float: ...


class BounceEaseOut(EasingBase):
    def func(self, t: float) -> float: ...


class BounceEaseInOut(EasingBase):
    def func(self, t: float) -> float: ...
