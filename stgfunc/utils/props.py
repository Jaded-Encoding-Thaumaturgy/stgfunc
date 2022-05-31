from __future__ import annotations


from typing import Type

import vapoursynth as vs

core = vs.core
def get_color_range(clip: vs.VideoNode) -> vs.ColorRange:
    return vs.ColorRange(get_prop(clip, '_ColorRange', int, 1))
