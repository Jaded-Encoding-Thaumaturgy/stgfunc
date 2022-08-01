from __future__ import annotations

from typing import Any, Callable, Sequence

import vapoursynth as vs
from vsexprtools import EXPR_VARS, ComparatorFunc, ExprOp, PlanesT, combine, normalise_planes
from vskernels import get_prop
from vsutil import get_neutral_value

__all__ = [
    'bestframeselect',
    'median_plane_value', 'mean_plane_value',
    'weighted_merge'
]

core = vs.core


def bestframeselect(
    clips: Sequence[vs.VideoNode], ref: vs.VideoNode,
    stat_func: Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode] = core.std.PlaneStats,
    prop: str = 'PlaneStatsDiff', comp_func: ComparatorFunc = max, debug: bool | tuple[bool, int] = False
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

    def _select(n: int, f: list[vs.VideoFrame]) -> vs.VideoNode:
        scores = [
            get_prop(diff.props, prop, float) for diff in f
        ]

        best = comp_func(indices, key=lambda i: scores[i])

        if do_debug:
            return clips[best].text.Text(
                "\n".join([f"Prop: {prop}", *[f"{i}: {s}"for i, s in enumerate(scores)], f"Best: {best}"]), alignment
            )

        return clips[best]

    return core.std.FrameEval(clips[0], _select, diffs)


def median_plane_value(
    clip: vs.VideoNode, planes: PlanesT = None, single_out: bool = False, cuda: bool | None = None
) -> vs.VideoNode:
    import numpy as np

    try:
        import cupy  # type: ignore
        cuda_available = True
    except ImportError:
        cupy = None
        cuda_available = False

    assert clip.format

    do_cuda = cuda_available if cuda is None else cuda

    npp = cupy if do_cuda and cuda_available and clip.height > 720 and clip.width > 1024 else np

    norm_planes = normalise_planes(clip, planes)

    if single_out:
        def _median_pvalue_modify_frame(f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
            fdst = f[1].copy()

            max_val = 0

            for plane in norm_planes:
                farr = npp.asarray(f[0][plane])
                farr = npp.reshape(farr, farr.shape[0] * farr.shape[1])

                max_val = max(max_val, int(npp.bincount(farr).argmax()))

            np.asarray(fdst[0])[0] = max_val

            return fdst
    else:
        def _median_pvalue_modify_frame(f: list[vs.VideoFrame], n: int) -> vs.VideoFrame:
            fdst = f[1].copy()

            for plane in norm_planes:
                farr = npp.asarray(f[0][plane])
                farr = npp.reshape(farr, farr.shape[0] * farr.shape[1])

                val = int(npp.bincount(farr).argmax())

                np.asarray(fdst[plane])[0] = val

            return fdst

    if single_out:
        out_format = vs.core.query_video_format(
            vs.GRAY, clip.format.sample_type, clip.format.bits_per_sample
        )
    elif clip.format.color_family == vs.YUV:
        out_format = clip.format.replace(subsampling_h=0, subsampling_w=0)
    else:
        out_format = clip.format

    nluma, nchroma = get_neutral_value(clip), get_neutral_value(clip, True)

    blankclip = clip.std.BlankClip(
        1, 1, int(out_format), keep=True, color=[nluma, nchroma, nchroma][:out_format.num_planes]
    )

    outclip = blankclip.std.ModifyFrame([clip, blankclip], _median_pvalue_modify_frame)

    return outclip.resize.Point(clip.width, clip.height)


def mean_plane_value(
    clip: vs.VideoNode, excl_values: list[float] | list[list[float]] | None = None, planes: PlanesT = None,
    single_out: bool = False, cuda: bool | None = None, prop: str = '{plane}Mean'
) -> vs.VideoNode:
    import numpy as np

    try:
        import cupy
        cuda_available = True
    except ImportError:
        cupy = None
        cuda_available = False

    assert clip.format

    do_cuda = cuda_available if cuda is None else cuda

    npp = cupy if do_cuda and cuda_available and clip.height > 720 and clip.width > 1024 else np

    norm_planes = normalise_planes(clip, planes)

    n_planes = len(norm_planes)

    color_fam_str = 'YUV' if clip.format.color_family == vs.YUV else 'RGB'

    prop_name = prop.format(plane=color_fam_str)
    plane_kwords = [prop.format(plane=plane) for plane in color_fam_str]

    if not excl_values:
        def _get_mean(farr: Any, plane: int) -> Any:
            return float(npp.mean(farr))
    elif isinstance(excl_values[0], (int, float)):
        excl_cut = excl_values[1:]

        def _get_mean(farr: Any, plane: int) -> Any:
            cond = farr != excl_values[0]  # type: ignore

            for val in excl_cut:
                cond = cond & (farr != val)

            selected = farr[cond]

            mean = npp.mean(selected)

            return float(mean)
    else:
        excl_cut = [excl[1:] for excl in excl_values]  # type: ignore

        def _get_mean(farr: Any, plane: int) -> Any:
            cond = farr != excl_values[plane][0]  # type: ignore

            for val in excl_cut[plane]:  # type: ignore
                cond = cond & (farr != val)

            selected = farr[cond]

            mean = npp.mean(selected)

            return float(mean)

    if single_out:
        def _mean_excl_modify_frame(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
            fdst = f.copy()

            means = []

            for plane in norm_planes:
                farr = npp.asarray(f[plane])
                farr = npp.reshape(farr, farr.shape[0] * farr.shape[1])

                means.append(_get_mean(farr, plane))

            fdst.props.update(**{prop_name: sum(means) / n_planes})

            return fdst
    else:
        def _mean_excl_modify_frame(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
            fdst = f.copy()

            for plane in norm_planes:
                farr = npp.asarray(f[plane])
                farr = npp.reshape(farr, farr.shape[0] * farr.shape[1])

                fdst.props.update(**{plane_kwords[plane]: _get_mean(farr, plane)})

            return fdst

    return clip.std.ModifyFrame(clip, _mean_excl_modify_frame)


def weighted_merge(*weighted_clips: tuple[vs.VideoNode, float]) -> vs.VideoNode:
    assert len(weighted_clips) <= len(EXPR_VARS), ValueError("weighted_merge: Too many clips!")

    clips, weights = zip(*weighted_clips)

    return combine(clips, ExprOp.ADD, zip(weights, ExprOp.MUL), expr_suffix=[sum(weights), ExprOp.DIV])
