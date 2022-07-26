from __future__ import annotations

from typing import Tuple

import vapoursynth as vs
from vsutil import depth, disallow_variable_format, get_depth


@disallow_variable_format
def expect_bits(clip: vs.VideoNode, expected_depth: int = 16) -> Tuple[int, vs.VideoNode]:
    return (bits := get_depth(clip)), depth(clip, expected_depth) if bits != expected_depth else clip


def checkValue(condition: bool, error_message: str) -> None:
    if condition:
        raise ValueError(error_message)


@disallow_variable_format
def checkSimilarClips(clipa: vs.VideoNode, clipb: vs.VideoNode) -> bool:
    assert clipa.format and clipb.format
    return clipa.height == clipb.height and clipa.width == clipb.width and clipa.format.id == clipb.format.id
