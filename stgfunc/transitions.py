from __future__ import annotations

from enum import Enum, IntEnum
from fractions import Fraction
from functools import partial
from math import ceil
from typing import NamedTuple, cast

from vskernels import Catrom, Kernel
from vstools import FrameRangeN, FrameRangesN, change_fps, clamp, disallow_variable_format, insert_clip, vs

from .easing import F_Easing, Linear, OnAxis

__all__ = [
    'fade', 'fade_freeze', 'fade_in', 'fade_out',
    'fade_in_freeze', 'fade_out_freeze',
    'crossfade', 'fade_ranges',
    'PanDirection', 'PanFunction', 'PanFunctions',
    'panner'
]


def fade(
    clipa: vs.VideoNode, clipb: vs.VideoNode, invert: bool, start: int,
    end: int, function: F_Easing = Linear
) -> vs.VideoNode:
    clipa_cut = clipa[start:end]
    clipb_cut = clipb[start:end]

    if invert:
        fade = crossfade(clipa_cut, clipb_cut, function)
    else:
        fade = crossfade(clipb_cut, clipa_cut, function)

    return insert_clip(clipa, fade, start)


def fade_freeze(
    clipa: vs.VideoNode, clipb: vs.VideoNode, invert: bool, start: int,
    end: int, function: F_Easing = Linear
) -> vs.VideoNode:
    start_f, end_f = (start, end) if invert else (end, start)

    length = end - start + 1

    return fade(
        insert_clip(clipa, clipa[start_f] * length, start),
        insert_clip(clipb, clipb[end_f] * length, start),
        invert, start, end, function
    )


def fade_in(clip: vs.VideoNode, start: int, end: int, function: F_Easing = Linear) -> vs.VideoNode:
    return fade(clip, clip.std.BlankClip(), False, start, end, function)


def fade_out(clip: vs.VideoNode, start: int, end: int, function: F_Easing = Linear) -> vs.VideoNode:
    return fade(clip, clip.std.BlankClip(), True, start, end, function)


def fade_in_freeze(clip: vs.VideoNode, start: int, end: int, function: F_Easing = Linear) -> vs.VideoNode:
    return fade_in(insert_clip(clip, clip[end] * (end - start + 1), start), start, end, function)


def fade_out_freeze(clip: vs.VideoNode, start: int, end: int, function: F_Easing = Linear) -> vs.VideoNode:
    return fade_out(insert_clip(clip, clip[start] * (end - start + 1), start), start, end, function)


def crossfade(
        clipa: vs.VideoNode, clipb: vs.VideoNode, function: F_Easing,
        debug: bool | int | tuple[int, int] = False
) -> vs.VideoNode:
    assert clipa.format and clipb.format

    if not clipa.height == clipb.height and clipa.width == clipb.width and clipa.format.id == clipb.format.id:
        raise ValueError('crossfade: Both clips must have the same length, dimensions and format.')

    ease_function = function(0, 1, clipa.num_frames)

    def _fading(n: int) -> vs.VideoNode:
        weight = ease_function.ease(n)
        merge = clipa.std.Merge(clipb, weight)
        return merge.text.Text(str(weight), 9, 2) if debug else merge

    return clipa.std.FrameEval(_fading)


def fade_ranges(
    clip_a: vs.VideoNode, clip_b: vs.VideoNode, ranges: FrameRangeN | FrameRangesN,
    fade_length: int = 5, ease_func: F_Easing = Linear
) -> vs.VideoNode:
    from lvsfunc.util import normalize_ranges

    nranges = normalize_ranges(clip_b, ranges)
    nranges = [(s - fade_length, e + fade_length) for s, e in nranges]
    nranges = normalize_ranges(clip_b, nranges)  # type: ignore

    franges = [range(s, e + 1) for s, e in nranges]

    ease_function = ease_func(0, 1, fade_length)

    def _fading(n: int) -> vs.VideoNode:
        frange: range | None = next((x for x in franges if n in x), None)

        if frange is None:
            return clip_a

        if frange.start + fade_length >= n >= frange.start:
            weight = ease_function.ease(n - frange.start)

            return clip_a.std.Merge(clip_b, weight)
        elif frange.stop - fade_length <= n <= frange.stop:
            weight = ease_function.ease(frange.stop - n)

            return clip_b.std.Merge(clip_a, 1 - weight)

        return clip_b

    return clip_a.std.FrameEval(_fading)


class PanDirection(IntEnum):
    NORMAL = 0
    INVERTED = 1


class PanFunction(NamedTuple):
    direction: PanDirection = PanDirection.NORMAL
    function_x: F_Easing = OnAxis
    function_y: F_Easing = OnAxis


class PanFunctions(PanFunction, Enum):
    VERTICAL_TTB = PanFunction(function_y=Linear)
    HORIZONTAL_LTR = PanFunction(function_x=Linear)
    VERTICAL_BTT = PanFunction(PanDirection.INVERTED, function_y=Linear)
    HORIZONTAL_RTL = PanFunction(PanDirection.INVERTED, function_x=Linear)


@disallow_variable_format
def panner(
    clip: vs.VideoNode, stitched: vs.VideoNode,
    pan_func: PanFunction | PanFunctions = PanFunctions.VERTICAL_TTB,
    fps: Fraction = Fraction(24000, 1001), kernel: Kernel = Catrom()
) -> vs.VideoNode:
    assert clip.format
    assert stitched.format

    if (stitched.format.subsampling_h, stitched.format.subsampling_w) != (0, 0):
        raise ValueError("stgfunc.panner: stitched can't be subsampled!")

    clip_cfps = change_fps(clip, fps)

    offset_x, offset_y = (stitched.width - clip.width), (stitched.height - clip.height)

    ease_x = pan_func.function_x(0, offset_x, clip_cfps.num_frames).ease
    ease_y = pan_func.function_y(0, offset_y, clip_cfps.num_frames).ease

    clamp_x = partial(lambda x: int(clamp(x, min_val=0, max_val=offset_x)))
    clamp_y = partial(lambda x: int(clamp(x, min_val=0, max_val=offset_y)))

    def _pan(n: int) -> vs.VideoNode:
        x_e, x_v = divmod(clamp_x(ease_x(n)), 1)
        y_e, y_v = divmod(clamp_y(ease_y(n)), 1)

        if n == clip_cfps.num_frames - 1:
            x_e, y_e = clamp_x(offset_x - 1), clamp_y(offset_y - 1)
            x_v, y_v = int(x_e == offset_x - 1), int(y_e == offset_y - 1)

        x_c, y_c = ceil(x_v), ceil(y_v)

        cropped = stitched.std.CropAbs(
            clip.width + x_c, clip.height + y_c, int(x_e), int(y_e)
        )

        shifted = kernel.shift(cropped, (y_v, x_v))

        cropped = shifted.std.Crop(bottom=y_c, right=x_c)

        return kernel.resample(cropped, cast(vs.VideoFormat, clip.format))

    newpan = clip_cfps.std.FrameEval(_pan)

    return newpan[::-1] if pan_func.direction == PanDirection.INVERTED else newpan
