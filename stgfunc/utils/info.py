from __future__ import annotations

import vapoursynth as vs
from vsutil import depth, disallow_variable_format, get_depth


@disallow_variable_format
def expect_bits(clip: vs.VideoNode, expected_depth: int = 16) -> tuple[int, vs.VideoNode]:
    return (bits := get_depth(clip)), depth(clip, expected_depth) if bits != expected_depth else clip


def checkValue(condition: bool, error_message: str) -> None:
    if condition:
        raise ValueError(error_message)
