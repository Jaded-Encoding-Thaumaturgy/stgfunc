from __future__ import annotations


from typing import Tuple

import vapoursynth as vs
from vsutil import depth, get_depth

from .func import disallow_variable_format


@disallow_variable_format
def expect_bits(clip: vs.VideoNode, expected_depth: int = 16) -> Tuple[int, vs.VideoNode]:
    return (bits := get_depth(clip)), depth(clip, expected_depth) if bits != expected_depth else clip


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
