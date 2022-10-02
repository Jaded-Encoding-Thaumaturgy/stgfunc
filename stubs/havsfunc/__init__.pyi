from __future__ import annotations


def GrainFactory3(
    clp: vs.VideoNode,
    g1str: float | None, g2str: float | None, g3str: float | None,
    g1shrp: int | None, g2shrp: int | None, g3shrp: int | None,
    g1size: float | None, g2size: float | None, g3size: float | None,
    temp_avg: int | None, ontop_grain: float | None, seed: int | None,
    th1: int | None, th2: int | None, th3: int | None, th4: int | None
) -> vs.VideoNode: ...
