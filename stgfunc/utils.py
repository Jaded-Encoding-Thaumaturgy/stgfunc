from __future__ import annotations

import re
import ast
import string
import inspect
import lvsfunc as lvf
from enum import Enum
import vapoursynth as vs
from itertools import cycle
from math import ceil, floor
from fractions import Fraction
from lvsfunc.types import Range
from lvsfunc.util import get_prop
from vskernels import Point
from vsutil import depth as vdepth, get_depth
from typing import Tuple, Union, List, Sequence, Dict, Any, TypeVar, Iterator, Iterable, SupportsFloat

from .types import SingleOrArr, SingleOrArrOpt, SupportsString, disallow_variable_format


core = vs.core

T = TypeVar('T')
StrArr = SingleOrArr[SupportsString]
StrArrOpt = SingleOrArrOpt[SupportsString]

vs_alph = (alph := list(string.ascii_lowercase))[(idx := alph.index('x')):] + alph[:idx]
akarin_available = hasattr(core, 'akarin')

expr_func = core.__getattr__('akarin' if akarin_available else 'std').Expr


@disallow_variable_format
def get_planes(_planes: SingleOrArrOpt[int], clip: vs.VideoNode) -> List[int]:
    assert clip.format
    n_planes = clip.format.num_planes

    planes = to_arr(range(n_planes) if _planes is None else _planes)

    return [p for p in planes if p < n_planes]


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
        return ' '.join(map(str, flatten(self)))


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

    def __mul__(self, n: int) -> List[ExprOp]:  # type: ignore
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

    expr_array = flatten(expr)  # type: ignore

    expr_array = filter(lambda x: x is not None and x != '', expr_array)

    expr_string = ' '.join([str(x).strip() for x in expr_array])

    planesl = get_planes(planes, firstclip)

    return expr_func(clips, [
        expr_string if x in planesl else ''
        for x in range(n_planes)
    ], **expr_kwargs)


@disallow_variable_format
def squaremask(clip: vs.VideoNode, width: int, height: int, offset_x: int, offset_y: int) -> vs.VideoNode:
    assert clip.format

    bits = get_depth(clip)
    src_w, src_h = clip.width, clip.height

    mask_format = clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0)

    white = 1 if mask_format.sample_type == vs.FLOAT else (1 << bits) - 1

    white_clip = core.std.BlankClip(clip, width, height, mask_format.id, 1, color=white, keep=True)

    padded = white_clip.std.AddBorders(
        offset_x, src_w - width - offset_x,
        offset_y, src_h - height - offset_y
    )

    return padded * clip.num_frames


@disallow_variable_format
def replace_squaremask(
    clipa: vs.VideoNode, clipb: vs.VideoNode, mask_params: Tuple[int, int, int, int],
    ranges: Union[Range, List[Range], None] = None,
    blur_sigma: int | None = None, invert: bool = False
) -> vs.VideoNode:
    assert clipa.format
    assert clipb.format

    mask = squaremask(clipb, *mask_params)

    if invert:
        mask = mask.std.InvertMask()

    if blur_sigma is not None:
        mask = mask.bilateralgpu.Bilateral(
            blur_sigma) if clipa.format.bits_per_sample == 32 else mask.bilateral.Gaussian(blur_sigma)

    merge = clipa.std.MaskedMerge(clipb, mask)

    return lvf.rfs(clipa, merge, ranges) if ranges else merge


def freeze_replace_mask(
    mask: vs.VideoNode, insert: vs.VideoNode,
    mask_params: Tuple[int, int, int, int], frame: int, frame_range: Tuple[int, int]
) -> vs.VideoNode:
    masked_insert = replace_squaremask(mask[frame], insert[frame], mask_params)
    return lvf.rfs(mask, masked_insert * mask.num_frames, frame_range)


def depth(*clips_depth: vs.VideoNode, **kwargs: Dict[str, Any]) -> Iterable[vs.VideoNode]:
    assert isinstance(clips_depth[-1], int)  # type: ignore

    clips = [vdepth(clip, clips_depth[-1], **kwargs) for clip in clips_depth[:-1]]  # type: ignore

    return clips[0] if len(clips) == 1 else clips


@disallow_variable_format
def get_bits(clip: vs.VideoNode, expected_depth: int = 16) -> Tuple[int, vs.VideoNode]:
    return (bits := get_depth(clip)), vdepth(clip, expected_depth) if bits != expected_depth else clip


@disallow_variable_format
def isGray(clip: vs.VideoNode) -> bool:
    assert clip.format
    return clip.format.color_family == vs.GRAY


def checkValue(condition: bool, error_message: str) -> None:
    if condition:
        raise ValueError(error_message)


@disallow_variable_format
def checkSimilarClips(clipa: vs.VideoNode, clipb: vs.VideoNode) -> bool:
    assert clipa.format and clipb.format
    return clipa.height == clipb.height and clipa.width == clipb.width and clipa.format.id == clipb.format.id


def change_fps(clip: vs.VideoNode, fps: Fraction) -> vs.VideoNode:
    src_num, src_den = clip.fps_num, clip.fps_den
    dest_num, dest_den = fps.as_integer_ratio()

    if (dest_num, dest_den) == (src_num, src_den):
        return clip

    factor = (dest_num / dest_den) * (src_den / src_num)

    def _frame_adjuster(n: int) -> vs.VideoNode:
        original = floor(n / factor)
        return clip[original] * (clip.num_frames + 100)

    new_fps_clip = clip.std.BlankClip(
        length=floor(clip.num_frames * factor), fpsnum=dest_num, fpsden=dest_den
    )

    return new_fps_clip.std.FrameEval(_frame_adjuster)


def weighted_merge(*weighted_clips: Tuple[vs.VideoNode, float]) -> vs.VideoNode:
    assert len(weighted_clips) <= len(vs_alph), ValueError("weighted_merge: Too many clips!")

    clips, weights = zip(*weighted_clips)

    return combine(clips, ExprOp.ADD, zip(weights, ExprOp.MUL), expr_suffix=[sum(weights), ExprOp.DIV])


def to_arr(array: Sequence[T] | T) -> List[T]:
    return list(array if (type(array) in {
        list, tuple, range, zip, set, map, enumerate
    }) else [array])  # type: ignore


def flatten(items: Iterable[T]) -> Iterable[T]:
    for val in items:
        if isinstance(val, Iterable) and not isinstance(val, (str, bytes)):
            for sub_x in flatten(val):
                yield sub_x
        else:
            yield val  # type: ignore


def pad_reflect(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    d_width, d_height = clip.width + left + right, clip.height + top + bottom

    return Point(src_width=d_width, src_height=d_height).scale(clip, d_width, d_height, (-top, -left))


def remove_chars(string: str, chars: str = '') -> str:
    return string.translate({ord(char): None for char in chars})


def destructure(dict_: Dict[str, T], ignore_no_key: bool = False) -> T | Iterator[T]:
    if not isinstance(dict_, object):
        raise TypeError(f"destructure: {dict_} is not an object!")

    parent_frame = inspect.currentframe()

    assert parent_frame and parent_frame.f_back

    previous_frame = parent_frame.f_back

    frame_info = inspect.getframeinfo(previous_frame)

    assert frame_info.code_context

    re_flags = re.I + re.M + re.U

    with open(previous_frame.f_code.co_filename, 'r') as f:
        source = f.read()

    end_index = previous_frame.f_lineno

    ast_root = ast.parse(source, previous_frame.f_code.co_filename)

    caller_node: ast.AST | None = None

    def _asttreewalk(parent: ast.AST, lineno: int) -> None:
        nonlocal caller_node
        for child in ast.iter_child_nodes(parent):
            if hasattr(child, 'lineno'):
                lineno = child.lineno
            if lineno > end_index:
                break

            is_call = isinstance(child, ast.Name) and isinstance(parent, ast.Call)

            if (is_call and child.id == parent_frame.f_code.co_name):  # type: ignore
                caller_node = parent
                break

            _asttreewalk(child, lineno)

    _asttreewalk(ast_root, 0)

    assert caller_node, RuntimeError('destructure: Code not properly formatted!')

    start_index = caller_node.lineno - 1

    source_split = source.splitlines()

    source_lines = source_split[start_index:end_index]

    curr_line = ' '.join(source_lines)

    nospaces = re.sub(r"\s+", r'', curr_line, 0, re_flags)

    if (cr_Tk := ')=' in nospaces) or (br_Tk := ']=' in nospaces):
        tmp_idx = start_index
        while ('(' if cr_Tk else '[' if br_Tk else '{') != curr_line[0] or curr_line[0] not in {'(', '['}:
            tmp_idx -= 1
            curr_line = '\n'.join([source_split[tmp_idx], curr_line])

    curr_line = re.sub(r"\n\n+", r'\n', curr_line, 0, re_flags)
    curr_line = re.sub(
        r"([\[\(])[\n\s]*(.*)[\n\s]*([\]\)]).*(=)", r'\1\2\3\4', curr_line, 0, re_flags
    )

    curr_line = re.sub(r"\s+", r'', curr_line, 0, re_flags)

    (lvalues, *_) = curr_line.strip().partition('=')

    lvalues = remove_chars(lvalues, ')(][ }{')

    keys = [ss for s in lvalues.split(',') if (ss := s.strip())]

    def _generator() -> Iterator[T]:
        nonlocal keys, dict_

        for key in keys:
            value: T

            try:
                value = dict_.__getattribute__(key)
            except BaseException:
                try:
                    value = dict_.__dict__.__getitem__(key)
                except BaseException:
                    try:
                        value = dict_.__class__.__dict__.__getitem__(key)
                    except BaseException:
                        try:
                            value = dict_[key]
                        except BaseException:
                            if not ignore_no_key:
                                raise KeyError(key)

            yield value

    gen_result = _generator()

    if len(keys) == 1:
        return next(gen_result)
    else:
        return gen_result


def get_color_range(clip: vs.VideoNode) -> vs.ColorRange:
    f = clip.get_frame(0)

    return vs.ColorRange(get_prop(f, '_ColorRange', int) if '_ColorRange' in f.props else 1)
