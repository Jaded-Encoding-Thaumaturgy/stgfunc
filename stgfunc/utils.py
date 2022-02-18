from __future__ import annotations

import string
import lvsfunc as lvf
from enum import Enum
import vapoursynth as vs
from math import ceil, floor
from fractions import Fraction
from lvsfunc.types import Range
from vsutil import depth as vdepth, get_depth
from typing import Tuple, Union, List, Sequence, Dict, Any, TypeVar

from .types import SingleOrArr, SingleOrArrOpt, SupportsString, disallow_variable_format


core = vs.core

T = TypeVar('T')
StrArr = SingleOrArr[SupportsString]

vs_alph = (alph := list(string.ascii_lowercase))[(idx := alph.index('x')):] + alph[:idx]


@disallow_variable_format
def get_planes(_planes: SingleOrArrOpt[int], clip: vs.VideoNode) -> List[int]:
    assert clip.format
    n_planes = clip.format.num_planes

    return [p for p in to_arr(range(n_planes) if _planes is None else _planes) if p < n_planes]  # type: ignore


class ExprOp(str, Enum):
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

    def __str__(self) -> str:
        return self.value


def combine(
    clips: Sequence[vs.VideoNode], operator: ExprOp = ExprOp.MAX, planes: List[int] | None = None,
    prefix: StrArr = '', suffix: StrArr = '', expr_prefix: StrArr = '', expr_suffix: StrArr = ''
) -> vs.VideoNode:
    n_clips = len(clips)

    prefixes = ((p := to_arr(prefix)) * max(1, ceil(n_clips / len(p))))
    suffixes = ((s := to_arr(suffix)) * max(1, ceil(n_clips / len(s))))
    expr_arr = [c for s[:n_clips + 1] in zip(prefixes, vs_alph, suffixes) for c in s] + [operator] * (n_clips - 1)

    return expr(clips, [*to_arr(expr_prefix), *expr_arr, *to_arr(expr_suffix)], planes)


def expr(clips: Sequence[vs.VideoNode], expr: StrArr, planes: List[int] | None) -> vs.VideoNode:
    firstclip = clips[0]
    assert firstclip.format

    expr_string = ' '.join([x for x in map(lambda x: str(x).strip(), expr) if x])  # type: ignore

    planesl = get_planes(planes, firstclip)

    return core.std.Expr(clips, [expr_string if x in planesl else '' for x in range(firstclip.format.num_planes)])


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


def depth(*clips_depth: vs.VideoNode, **kwargs: Dict[str, Any]) -> Sequence[vs.VideoNode]:
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
    assert isinstance(clipa, vs.VideoNode) and clipa.format
    assert isinstance(clipb, vs.VideoNode) and clipb.format
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

    return combine(clips, ExprOp.ADD, None, weights, ExprOp.MUL, None, [sum(weights), ExprOp.DIV])


def to_arr(array: SingleOrArr[T]) -> List[T]:
    return list(array) if (type(array) in [list, tuple, range]) else [array]  # type: ignore
