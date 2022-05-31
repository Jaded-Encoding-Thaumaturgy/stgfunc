from __future__ import annotations

from typing import Callable, Iterable, List, Protocol, Sequence, Tuple, TypeVar

import vapoursynth as vs
from lvsfunc.util import get_prop

from .types import T

core = vs.core


class _CompFunction(Protocol):
    def __call__(self, __iterable: Iterable[T], *, key: Callable[[T], float]) -> T:
        ...


def bestframeselect(
    clips: Sequence[vs.VideoNode], ref: vs.VideoNode,
    stat_func: Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode] = core.std.PlaneStats,
    prop: str = 'PlaneStatsDiff', comp_func: _CompFunction = max, debug: bool | Tuple[bool, int] = False
) -> vs.VideoNode:
    """
    Rewritten from https://github.com/po5/notvlc/blob/master/notvlc.py#L23.

    Picks the 'best' clip for any given frame using stat functions.
    clips: list of clips
    ref: reference clip, e.g. core.average.Mean(clips) / core.median.Median(clips)
    stat_func: function that adds frame properties
    prop: property added by stat_func to compare
    comp_func: function to decide which clip to pick, e.g. min, max
    debug: display values of prop for each clip, and which clip was picked, optionally specify alignment
    """
    diffs = [stat_func(clip, ref) for clip in clips]
    indices = list(range(len(diffs)))
    do_debug, alignment = debug if isinstance(debug, tuple) else (debug, 7)

    def _select(n: int, f: List[vs.VideoFrame]) -> vs.VideoNode:
        scores = [
            get_prop(diff, prop, float) for diff in f
        ]

        best = comp_func(indices, key=lambda i: scores[i])

        if do_debug:
            return clips[best].text.Text(
                "\n".join([f"Prop: {prop}", *[f"{i}: {s}"for i, s in enumerate(scores)], f"Best: {best}"]), alignment
            )

        return clips[best]

    return core.std.FrameEval(clips[0], _select, diffs)
