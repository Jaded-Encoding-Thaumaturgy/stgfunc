import lvsfunc as lvf
import vapoursynth as vs
from enum import Enum, IntEnum
from fractions import Fraction
from lvsfunc.kernels import Kernel, Catrom
from typing import NamedTuple, Type, Union, Tuple
from vsutil import insert_clip, disallow_variable_format

from .easing import EasingBase, Linear
from .utils import checkSimilarClips, change_fps

core = vs.core


def fade(
    clipa: vs.VideoNode, clipb: vs.VideoNode, invert: bool, start: int,
    end: int, function: Type[EasingBase] = Linear
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
    end: int, function: Type[EasingBase] = Linear
) -> vs.VideoNode:
    return fade(
        lvf.rfs(clipa, clipa[start if invert else end] * clipa.num_frames, (start, end)),
        lvf.rfs(clipb, clipb[end if invert else start] * clipb.num_frames, (start, end)),
        invert, start, end, function
    )


def fade_in(clip: vs.VideoNode, start: int, end: int, function: Type[EasingBase] = Linear) -> vs.VideoNode:
    return fade(clip, clip.std.BlankClip(), False, start, end, function)


def fade_out(clip: vs.VideoNode, start: int, end: int, function: Type[EasingBase] = Linear) -> vs.VideoNode:
    return fade(clip, clip.std.BlankClip(), True, start, end, function)


def fade_in_freeze(clip: vs.VideoNode, start: int, end: int, function: Type[EasingBase] = Linear) -> vs.VideoNode:
    return fade_in(lvf.rfs(clip, clip[end] * clip.num_frames, (start, end)), start, end, function)


def fade_out_freeze(clip: vs.VideoNode, start: int, end: int, function: Type[EasingBase] = Linear) -> vs.VideoNode:
    return fade_out(lvf.rfs(clip, clip[start] * clip.num_frames, (start, end)), start, end, function)


def crossfade(
        clipa: vs.VideoNode, clipb: vs.VideoNode, function: Type[EasingBase],
        debug: Union[bool, int, Tuple[int, int]] = False) -> vs.VideoNode:
    if not checkSimilarClips(clipa, clipb):
        raise ValueError('crossfade: Both clips must have the same length, dimensions and format.')

    ease_function = function(0, 1, clipa.num_frames)

    def _fading(n: int) -> vs.VideoNode:
        weight = ease_function.ease(n)
        merge = clipa.std.Merge(clipb, weight)
        return merge.text.Text(str(weight), 9, 2) if debug else merge

    return clipa.std.FrameEval(_fading)


