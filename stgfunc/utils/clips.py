from __future__ import annotations

from fractions import Fraction
from math import floor

import vapoursynth as vs

__all__ = [
    'change_fps'
]

core = vs.core


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
