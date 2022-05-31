from __future__ import annotations

import string
from enum import Enum
from itertools import cycle
from math import ceil
from typing import Any, Dict, Iterator, List, Sequence, SupportsFloat, Tuple, Callable

import vapoursynth as vs

from .types import SingleOrArr, SingleOrArrOpt, SupportsString, StrList
from .utils import flatten, get_planes, to_arr

core = vs.core

StrArr = SingleOrArr[SupportsString]
StrArrOpt = SingleOrArrOpt[SupportsString]

vs_alph = (alph := list(string.ascii_lowercase))[(idx := alph.index('x')):] + alph[:idx]
akarin_available = hasattr(core, 'akarin')

expr_func: Callable[
    [vs.VideoNode | Sequence[vs.VideoNode], str | Sequence[str]], vs.VideoNode
] = getattr(core, 'akarin' if akarin_available else 'std').Expr


class ExprOp(str, Enum):
    # 1 Argument
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    SIN = "sin"
    COS = "cos"
    ABS = "abs"
    NOT = "not"
    DUP = "dup"
    DUPN = "dupN"
    # 2 Arguments
    MAX = "max"
    MIN = "min"
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    POW = "pow"
    GT = ">"
    LT = "<"
    EQ = "="
    GTE = ">="
    LTE = "<="
    AND = "and"
    OR = "or"
    XOR = "xor"
    SWAP = "swap"
    SWAPN = "swapN"
    # 3 Arguments
    TERN = "?"

    @classmethod
    def clamp(cls, min: float, max: float) -> StrList:
        return StrList([min, ExprOp.MAX, max, ExprOp.MIN])

    @classmethod
    def convolution(cls, var: str, convolution: Sequence[SupportsFloat]) -> StrList:
        expr_conv = [
            x.format(c=var, w=w) for x, w in zip([
                '{c}[-1,-1] {w} *', '{c}[0,-1] {w} *', '{c}[1,-1] {w} *',
                '{c}[-1,0] {w} *', '{c} {w} *', '{c}[1,0] {w} *',
                '{c}[-1,1] {w} *', '{c}[0,1] {w} *', '{c}[1,1] {w} *'
            ], convolution)
        ]

        return StrList([*expr_conv, cls.ADD * 8, sum(map(float, convolution)), cls.DIV])

    @classmethod
    def matrix(cls, var: str, full: bool = True) -> StrList:
        return StrList([
            f'{var}[-1,-1]', f'{var}[0,-1]', f'{var}[1,-1]',
            f'{var}[-1,0]', *([f'{var}'] if full else []), f'{var}[1,0]',
            f'{var}[-1,1]', f'{var}[0,1]', f'{var}[1,1]'
        ])

    @classmethod
    def boxblur(cls, var: str) -> StrList:
        return StrList([*cls.matrix(var), cls.ADD * 8, 9, cls.DIV])

    def __str__(self) -> str:
        return self.value

    def __next__(self) -> ExprOp:
        return self

    def __iter__(self) -> Iterator[ExprOp]:
        return cycle([self])

    def __mul__(self, n: int) -> List[ExprOp]:  # type: ignore[override]
        return [self] * n


def _combine_norm__ix(ffix: StrArrOpt, n_clips: int) -> List[SupportsString]:
    if ffix is None:
        return [''] * n_clips

    ffix = [ffix] if (type(ffix) in {str, tuple}) else list(ffix)  # type: ignore

    return ffix * max(1, ceil(n_clips / len(ffix)))


def combine(
    clips: Sequence[vs.VideoNode], operator: ExprOp = ExprOp.MAX, suffix: StrArrOpt = None, prefix: StrArrOpt = None,
    expr_suffix: StrArrOpt = None, expr_prefix: StrArrOpt = None, planes: SingleOrArrOpt[int] = None,
    **expr_kwargs: Dict[str, Any]
) -> vs.VideoNode:
    n_clips = len(clips)

    prefixes, suffixes = (_combine_norm__ix(x, n_clips) for x in (prefix, suffix))

    normalized_args = [to_arr(x)[:n_clips + 1] for x in (prefixes, vs_alph, suffixes)]

    args = zip(*normalized_args)

    operators = operator * (n_clips - 1)

    return expr(clips, [expr_prefix, args, operators, expr_suffix], planes, **expr_kwargs)


def expr(
    clips: Sequence[vs.VideoNode], expr: StrArr, planes: SingleOrArrOpt[int], **expr_kwargs: Dict[str, Any]
) -> vs.VideoNode:
    firstclip = clips[0]
    assert firstclip.format

    n_planes = firstclip.format.num_planes

    expr_array: List[SupportsString] = flatten(expr)  # type: ignore

    expr_array_filtered = filter(lambda x: x is not None and x != '', expr_array)

    expr_string = ' '.join([str(x).strip() for x in expr_array_filtered])

    planesl = get_planes(planes, firstclip)

    return expr_func(clips, [
        expr_string if x in planesl else ''
        for x in range(n_planes)
    ], **expr_kwargs)


def weighted_merge(*weighted_clips: Tuple[vs.VideoNode, float]) -> vs.VideoNode:
    assert len(weighted_clips) <= len(vs_alph), ValueError("weighted_merge: Too many clips!")

    clips, weights = zip(*weighted_clips)

    return combine(clips, ExprOp.ADD, zip(weights, ExprOp.MUL), expr_suffix=[sum(weights), ExprOp.DIV])
