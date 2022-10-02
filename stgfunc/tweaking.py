from __future__ import annotations

from enum import IntEnum
from itertools import cycle
from math import cos, degrees, floor, pi, sin
from typing import Any, NamedTuple, Sequence, SupportsFloat

from vsexprtools import ExprOp, norm_expr
from vskernels import BSpline, Catrom, Point
from vstools import (
    FrameRangeN, StrList, VSFunction, core, disallow_variable_format, disallow_variable_resolution, fallback, get_depth,
    get_neutral_value, get_peak_value, get_prop, get_subsampling, insert_clip, scale_value, vs
)

from .easing import ExponentialEaseIn, F_Easing
from .transitions import crossfade

__all__ = [
    'Tweak',
    'tweak_clip', 'multi_tweak',
    'BalanceMode', 'WeightMode', 'Override',
    'auto_balance', 'bbmod_fast'
]


@disallow_variable_format()
def tweak_clip(
        clip: vs.VideoNode, cont: float = 1.0, sat: float = 1.0,
        bright: float = 0.0, hue: float = 0.0, relative_sat: float | None = None,
        range_in: vs.ColorRange | None = None, range_out: vs.ColorRange | None = None,
        clamp: bool = True, pre: vs.VideoNode | VSFunction | None = None, post: VSFunction | None = None
) -> vs.VideoNode:
    assert clip.format

    bits = get_depth(clip)

    range_in = fallback(range_in, get_prop(clip, '_ColorRange', int, vs.ColorRange, vs.RANGE_LIMITED))
    range_out = fallback(range_out, range_in)

    sv_args_out = dict[str, Any](
        input_depth=8, output_depth=bits, range_in=1 - range_in, range=1 - range_out, scale_offsets=True
    )

    luma_min = scale_value(16, **sv_args_out)
    chroma_min = scale_value(16, **sv_args_out, chroma=True)

    luma_max = scale_value(235, **sv_args_out)
    chroma_max = scale_value(240, **sv_args_out, chroma=True)

    chroma_center = get_neutral_value(clip, True)

    if relative_sat is not None:
        if cont == 1.0 or relative_sat == 1.0:
            sat = cont
        else:
            sat = (cont - 1.0) * relative_sat + 1.0

    cont = max(cont, 0.0)
    sat = max(sat, 0.0)

    if (hue == bright == 0.0) and (sat == cont == 1.0):
        return clip

    pre_clip = pre(clip) if callable(pre) else fallback(pre, clip)

    clips = [pre_clip]

    yexpr = list[Any](['x'])
    cexpr = list[Any](['x'])

    if (hue != 0.0 or sat != 1.0) and clip.format.color_family != vs.GRAY:
        hue *= pi / degrees(pi)

        hue_sin, hue_cos = sin(hue), cos(hue)

        normalize = [chroma_center, ExprOp.SUB]

        cexpr.extend([normalize, hue_cos, sat, ExprOp.MUL * 2])

        if hue != 0:
            clips += [pre_clip.std.ShufflePlanes([0, 2, 1], vs.YUV)]
            cexpr.extend(['y', normalize, hue_sin, sat, ExprOp.MUL * 2, ExprOp.ADD])

        cexpr.extend([chroma_center, ExprOp.ADD])

        if clamp and range_out:
            cexpr.extend(StrList([chroma_min, ExprOp.MAX, chroma_max, ExprOp.MIN]))

    if bright != 0 or cont != 1:
        if luma_min > 0:
            yexpr.extend([luma_min, ExprOp.SUB])

        if cont != 1:
            yexpr.extend([cont, ExprOp.MUL])

        if (luma_min + bright) != 0:
            yexpr.extend([luma_min, bright, ExprOp.ADD * 2])

        if clamp and range_out:
            yexpr.extend(StrList([luma_min, ExprOp.MAX, luma_max, ExprOp.MIN]))

    tclip = norm_expr(clips, (yexpr, cexpr))

    return post(tclip) if callable(post) else tclip


class Tweak(NamedTuple):
    frame: int
    cont: SupportsFloat | None = None
    sat: SupportsFloat | None = None
    bright: SupportsFloat | None = None
    hue: SupportsFloat | None = None
    ease_func: F_Easing = ExponentialEaseIn


@disallow_variable_format(only_first=True)
def multi_tweak(clip: vs.VideoNode, tweaks: list[Tweak], debug: bool = False, **tkargs: dict[str, Any]) -> vs.VideoNode:
    if len(tweaks) < 2:
        raise ValueError("multi_tweak: 'At least two tweaks need to be passed!'")

    for i, tmp_tweaks in enumerate(zip([tweaks[0]] + tweaks, tweaks, cycle(tweaks[1:]))):
        tprev, tweak, tnext = [list(filter(None, x)) for x in tmp_tweaks]

        if len(tweak) == 1 and len(tprev) > 1 and i > 0:
            tweak = tweak[:1] + tprev[1:]

        cefunc, _ = tweak.pop(), tnext.pop()
        start, stop = tweak.pop(0), tnext.pop(0)

        if start == stop:
            continue

        assert isinstance(start, int) and isinstance(stop, int)

        spliced_clip = clip[start:stop]

        if tweak == tnext:
            tweaked_clip = tweak_clip(spliced_clip, *tweak, **tkargs)  # type: ignore
        else:
            clipa, clipb = (tweak_clip(spliced_clip, *args, **tkargs) for args in (tweak, tnext))  # type: ignore

            tweaked_clip = crossfade(clipa, clipb, cefunc, debug)  # type: ignore

        clip = insert_clip(clip, tweaked_clip, start)

    return clip


class BalanceMode(IntEnum):
    AUTO = 0
    UNDIMMING = 1
    DIMMING = 2


class WeightMode(IntEnum):
    INTERPOLATE = 0
    MEDIAN = 1
    MEAN = 2
    MAX = 3
    MIN = 4
    NONE = 5


class Override(NamedTuple):
    frame_range: FrameRangeN
    cont: SupportsFloat
    override_mode: WeightMode = WeightMode.INTERPOLATE


@disallow_variable_format
def auto_balance(
    clip: vs.VideoNode, target_max: SupportsFloat | None = None, relative_sat: float = 1.0,
    range_in: vs.ColorRange = vs.RANGE_LIMITED, frame_overrides: Override | Sequence[Override] = [],
    ref: vs.VideoNode | None = None, radius: int = 1, delta_thr: float = 0.4,
    min_thr: int | float | None = None, max_thr: SupportsFloat | None = None,
    balance_mode: BalanceMode = BalanceMode.UNDIMMING, weight_mode: WeightMode = WeightMode.MEAN,
    debug: bool = False, **range_kwargs: Any
) -> vs.VideoNode:
    import numpy as np
    from lvsfunc.util import normalize_ranges

    ref_clip = fallback(ref, clip)

    bits = get_depth(ref_clip)

    zero = scale_value(16, 8, bits, 1 - range_in, scale_offsets=True)

    target = float(fallback(
        target_max,
        scale_value(
            235, input_depth=8, output_depth=bits,
            range_in=1 - range_in, scale_offsets=True
        )
    )) - float(zero)

    min_thr = fallback(min_thr, 0)
    max_thr = fallback(max_thr, get_peak_value(clip))

    if weight_mode == WeightMode.NONE:
        raise ValueError("auto_balance: Global weight mode can't be NONE!")

    ref_stats = ref_clip.std.PlaneStats()

    range_kwargs.update({'range_in': range_in})

    over_mapped = list[tuple[range, float, WeightMode]]()

    if frame_overrides:
        frame_overrides = [frame_overrides] if isinstance(frame_overrides, Override) else list(frame_overrides)

        over_frames, over_conts, over_int_modes = list(zip(*frame_overrides))

        oframes_ranges = [
            range(start, stop + 1)
            for start, stop in normalize_ranges(clip, list(over_frames))
        ]

        over_mapped = list(zip(oframes_ranges, over_conts, over_int_modes))

    clipfrange = range(0, clip.num_frames)

    def _weighted(x: float, y: float, z: float) -> float:
        return max(1, x - z) / max(1, y - z)

    def _autobalance(n: int, f: Sequence[vs.VideoFrame]) -> vs.VideoNode:
        override: tuple[range, float, WeightMode] | None = next((x for x in over_mapped if n in x[0]), None)

        psvalues: Any = np.asarray([
            _weighted(target, get_prop(frame.props, 'PlaneStatsMax', float), zero) for frame in f
        ])

        middle_idx = psvalues.size // 2

        curr_value = psvalues[middle_idx]

        if not min_thr <= curr_value <= max_thr:
            return clip

        if balance_mode == BalanceMode.UNDIMMING:
            psvalues[psvalues < 1.0] = 1.0
        elif balance_mode == BalanceMode.DIMMING:
            psvalues[psvalues > 1.0] = 1.0

        psvalues[(abs(psvalues - curr_value) > delta_thr)] = curr_value

        def _get_cont(mode: WeightMode, frange: range) -> Any:
            if mode == WeightMode.INTERPOLATE:
                if radius < 1:
                    raise ValueError("auto_balance: 'Radius has to be >= 1 with WeightMode.INTERPOLATE!'")

                weight = (n - (frange.start - 1)) / (frange.stop - (frange.start - 1))

                weighted_prev = psvalues[middle_idx - 1] * (1 - weight)
                weighted_next = psvalues[middle_idx + 1] * weight

                return weighted_prev + weighted_next
            elif mode == WeightMode.MEDIAN:
                return np.median(psvalues)
            elif mode == WeightMode.MEAN:
                return psvalues.mean()
            elif mode == WeightMode.MAX:
                return psvalues.max()
            elif mode == WeightMode.MIN:
                return psvalues.min()

            return psvalues[middle_idx]

        if override:
            frange, cont, override_mode = override

            if override_mode == WeightMode.NONE:
                return clip

            if cont is not None:
                psvalues[
                    max(0, middle_idx - (n - frange.start)):
                    min(len(psvalues), middle_idx + (frange.stop - n))
                ] = cont

            if (override_mode != weight_mode):
                cont = _get_cont(override_mode, frange)
        else:
            cont = _get_cont(weight_mode, clipfrange)

        sat = (cont - 1) * relative_sat + 1

        return tweak_clip(clip, cont, sat, **range_kwargs)

    stats_clips = [
        *(ref_stats[0] * i + ref_stats[:-i] for i in range(1, radius + 1)),
        ref_stats,
        *(ref_stats[i:] + ref_stats[-1] * i for i in range(1, radius + 1)),
    ]

    return clip.std.FrameEval(_autobalance, stats_clips, clip)


@disallow_variable_format
@disallow_variable_resolution
def bbmod_fast(
    clip: vs.VideoNode, top: int = 0, bottom: int = 0,
    left: int = 0, right: int = 0, thresh: int = 128,
    blur: int = 1000, scale: int = 1
) -> vs.VideoNode:
    assert clip.format

    if clip.format.color_family not in {vs.YUV, vs.GRAY}:
        raise ValueError("bbmod_fast: Only YUV and GRAY color family supported!")

    if thresh <= 0 or thresh > 255:
        raise ValueError("bbmod_fast: Threshold has to be in range (1, 255) !")

    if scale < 1:
        raise ValueError("bbmod_fast: Scale has to be > 1!")

    if any(x < 0 for x in {top, bottom, left, right}):
        raise ValueError("bbmod_fast: All sides have to be >= 0!")

    if get_subsampling(clip):
        clip444 = Catrom.resample(clip, clip.format.replace(subsampling_h=0, subsampling_w=0))
    else:
        clip444 = clip

    bits = get_depth(clip444)

    neutral_luma = get_neutral_value(clip)
    neutral_chroma = get_neutral_value(clip, chroma=True)

    blur_width = max(8, floor(clip.width / blur)) * scale * 2

    tv_clamp = scale_value(16, 8, bits, scale_offsets=True)
    thr_scaled = scale_value(thresh, 8, bits)

    euler_clamp = '1 exp sin 8 clamp'
    clamp_luma = f'{neutral_luma - thr_scaled} {neutral_luma + thr_scaled} clamp'

    expressions = [
        f'{tv_clamp} i!   z i@ - y i@ - / {euler_clamp} x i@ - * {clamp_luma} i@ +',
        f'{neutral_chroma} n! z y - z y / {euler_clamp} x n@ - * n@ + + x - x +'
    ][:clip.format.num_planes]

    def _brescale(ref: vs.VideoNode) -> vs.VideoNode:
        return BSpline.scale(BSpline.scale(ref, blur_width, ref.height), ref.width, ref.height)

    def _bbmod(clip: vs.VideoNode, top: int, bottom: int) -> vs.VideoNode:
        originalRows = core.std.StackVertical([
            clip.std.CropAbs(clip.width, top),
            clip.std.CropAbs(clip.width, bottom, top=clip.height - bottom)
        ])

        upsampleRows = core.std.StackVertical([
            clip.std.CropAbs(clip.width, 1, top=top),
            clip.std.CropAbs(clip.width, 1, top=clip.height - bottom)
        ]).resize.Point(height=top + bottom)

        balanced = core.akarin.Expr([
            originalRows, _brescale(originalRows), _brescale(upsampleRows)
        ], expressions)

        return core.std.StackVertical([
            balanced.std.CropAbs(clip.width, top),
            clip.std.CropAbs(clip.width, clip.height - top - bottom, top=top),
            balanced.std.CropAbs(clip.width, bottom, top=balanced.height - bottom),
        ])

    upsample = Point().scale(
        clip444, clip.width * scale, clip.height * scale
    ) if scale > 1 else clip444

    fixed = _bbmod(upsample, top * scale, bottom * scale)

    if any({left, right}):
        fixed = fixed.std.Transpose()
        fixed = _bbmod(fixed, right * scale, left * scale)
        fixed = fixed.std.Transpose()

    if scale > 1:
        fixed = Point().scale(fixed, clip.width, clip.height)

    return Catrom.resample(fixed, clip.format)
