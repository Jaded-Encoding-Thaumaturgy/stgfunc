from __future__ import annotations

from typing import List
from vskernels import Point
from fractions import Fraction
from math import floor

import vapoursynth as vs

from .func import to_arr, disallow_variable_format
from ..types import SingleOrArrOpt

core = vs.core


@disallow_variable_format
def get_planes(_planes: SingleOrArrOpt[int], clip: vs.VideoNode) -> List[int]:
    assert clip.format
    n_planes = clip.format.num_planes

    planes = to_arr(range(n_planes) if _planes is None else _planes)

    return [p for p in planes if p < n_planes]


def pad_reflect(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    d_width, d_height = clip.width + left + right, clip.height + top + bottom

    return Point(src_width=d_width, src_height=d_height).scale(clip, d_width, d_height, (-top, -left))


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
