from __future__ import annotations

from functools import reduce
from typing import Any, Callable, Sequence, cast

from vsexprtools import EXPR_VARS, ExprOp, combine
from vstools import ComparatorFunc, PlanesT, core, get_neutral_value, get_prop, normalize_planes, split, vs

__all__ = [
    'bestframeselect',
    'median_plane_value', 'mean_plane_value',
    'weighted_merge'
]


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

    norm_planes = normalize_planes(clip, planes)

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
        1, 1, out_format.id, keep=True, color=[nluma, nchroma, nchroma][:out_format.num_planes]
    )

    outclip = blankclip.std.ModifyFrame([clip, blankclip], _median_pvalue_modify_frame)

    return outclip.resize.Point(clip.width, clip.height)


def mean_plane_value(
    clip: vs.VideoNode, excl_values: Sequence[int | float] | Sequence[Sequence[int | float]] | None = None,
    single_out: bool = False, prop: str | list[str] = '{plane}Mean', cuda: bool | None = None, planes: PlanesT = None
) -> vs.VideoNode:
    import numpy as np
    from numpy.typing import NDArray

    assert clip.format

    try:
        import cupy
        cuda_available = True
    except ImportError:
        cupy = None
        cuda_available = False

    assert clip.format

    do_cuda = cuda_available if cuda is None else cuda

    npp = cupy if do_cuda and cuda_available and clip.height > 720 and clip.width > 1024 else np

    norm_planes = normalize_planes(clip, planes)
    n_planes = len(norm_planes)

    color_fam_str = 'YUV' if clip.format.color_family == vs.YUV else 'RGB'

    plane_sizes = [plane.width * plane.height for plane in split(clip)]

    def to_float(farr: NDArray[Any]) -> float:
        return float(np.nan_to_num(npp.mean(farr, dtype=npp.float64)))

    if excl_values is not None:
        excl_values = list(excl_values)  # type: ignore

    if not excl_values:
        def _get_arr_mean(farr: NDArray[Any], plane: int) -> float:
            return to_float(farr)
    else:
        is_single = isinstance(excl_values[0], (int, float))

        if is_single:
            excl_sfirst, excl_scut = cast(float, excl_values[0]), cast(list[float], excl_values[1:])
        else:
            if len(excl_values) < clip.format.num_planes:
                raise ValueError('mean_plane_value: you must specify an array of excluded values per each plane!')

            excl_mfirst = cast(list[float], [arr[0] for arr in excl_values])  # type: ignore
            excl_mcut = cast(list[list[float]], [arr[1:] for arr in excl_values])  # type: ignore

        def _get_cond(farr: NDArray[Any], excl_cut: list[float], excl_first: float) -> Any:
            return reduce(lambda cond, val: cond & (farr != val), excl_cut, farr != excl_first)

        if is_single:
            val_len = len(excl_values)

            if val_len == 1:
                def _get_arr_mean(farr: NDArray[Any], plane: int) -> float:
                    return to_float(farr[farr != excl_sfirst])
            elif val_len == 2:
                excl_ssecond = excl_values[1]

                def _get_arr_mean(farr: NDArray[Any], plane: int) -> float:
                    return to_float(farr[(farr != excl_sfirst) & (farr != excl_ssecond)])
            else:
                def _get_arr_mean(farr: NDArray[Any], plane: int) -> float:
                    return to_float(farr[_get_cond(farr, excl_scut, excl_sfirst)])
        else:
            def _get_arr_mean(farr: NDArray[Any], plane: int) -> float:
                return to_float(farr[_get_cond(farr, excl_mcut[plane], excl_mfirst[plane])])

    def _get_mean(f: vs.VideoFrame, plane: int) -> float:
        return _get_arr_mean(
            npp.reshape(
                npp.asarray(f[plane]),
                plane_sizes[plane]
            ), plane
        )

    if single_out:
        if isinstance(prop, list):
            raise ValueError('mean_plane_value: with single_out=True, prop must be a string!')

        prop_name = prop.format(plane=color_fam_str)

        def _mean_excl_modify_frame(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
            fdst = f.copy()
            fdst.props.update([(prop_name, sum(_get_mean(f, plane) for plane in norm_planes) / n_planes)])
            return fdst
    else:
        if isinstance(prop, list):
            prop_planes_name = prop
        else:
            prop_planes_name = [prop.format(plane=plane) for plane in color_fam_str]

        plane_kwords = list(zip(norm_planes, prop_planes_name))

        def _mean_excl_modify_frame(f: vs.VideoFrame, n: int) -> vs.VideoFrame:
            fdst = f.copy()
            fdst.props.update(**{kword: _get_mean(f, plane) for plane, kword in plane_kwords})
            return fdst

    return clip.std.ModifyFrame(clip, _mean_excl_modify_frame)


def weighted_merge(*weighted_clips: tuple[vs.VideoNode, float]) -> vs.VideoNode:
    assert len(weighted_clips) <= len(EXPR_VARS), ValueError("weighted_merge: Too many clips!")

    clips, weights = zip(*weighted_clips)

    return combine(clips, ExprOp.ADD, zip(weights, ExprOp.MUL), expr_suffix=[sum(weights), ExprOp.DIV])
