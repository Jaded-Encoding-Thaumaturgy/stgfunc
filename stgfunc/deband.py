from __future__ import annotations

from functools import partial
from itertools import cycle
from typing import Any, SupportsFloat, cast

from vsexprtools import ExprOp, combine
from vskernels import Catrom, Lanczos
from vstools import (
    DitherType, SingleOrArr, core, depth, disallow_variable_format, disallow_variable_resolution, expect_bits,
    get_depth, get_prop, get_w, get_y, iterate, join, vs
)

from .mask import detail_mask
from .misc import set_output
from .noise import adaptive_grain
from .types import DebanderFN

__all__ = [
    'masked_f3kdb',
    'auto_deband'
]


def masked_f3kdb(
    clip: vs.VideoNode, rad: int = 16, threshold: SingleOrArr[int] = 24,
    grain: SingleOrArr[int] = [12, 0], mask_args: dict[str, Any] = {}
) -> vs.VideoNode:
    try:
        from debandshit import dumb3kdb
    except ModuleNotFoundError:
        raise ModuleNotFoundError('masked_f3kdb: missing dependency `debandshit`')

    clip, bits = expect_bits(clip, 16)

    mask_kwargs = dict[str, Any](brz=(1000, 2750)) | mask_args

    deband_mask = detail_mask(clip, **mask_kwargs)

    deband = dumb3kdb(clip, radius=rad, threshold=threshold, grain=grain, seed=69420)
    deband_masked = deband.std.MaskedMerge(clip, deband_mask)

    return deband_masked if bits == 16 else depth(deband_masked, bits)


__auto_deband_cache = dict[str, tuple[vs.VideoNode, list[vs.VideoNode], vs.VideoNode, vs.VideoNode]]()


@disallow_variable_format(only_first=True)
@disallow_variable_resolution(only_first=True)
def auto_deband(
    clip: vs.VideoNode, cambi_thr: float = 12.0, cambi_scale: float = 1.2,
    min_thr: int | float = 24, max_thr: int | float = 48, steps: int = 4,
    grain_thrs: tuple[int, int, int] | None = None,
    debander: DebanderFN | None = None,
    ref: vs.VideoNode | None = None, downsample_h: None | int = None,
    debug: tuple[bool, bool] = (False, False),
    debander_args: dict[str, Any] = {}, adptvgr_args: dict[str, Any] = {},
    **cambi_kwargs: Any
) -> vs.VideoNode:
    """
        Automated banding detection and filtering via the use of CAMBI.
        A range of potential debanding functions are spawned, of which
        an approximated value is chosen based off the score returned by CAMBI.

        Please see:
            https://github.com/AkarinVS/vapoursynth-plugin/wiki/CAMBI

        Function is extensible, allowing for custom functions for
        debanding and grain applied in place of defaults.
        For anime, consider either disabling the graining function, or
        or using adptvgr_args={"static"=True}

        Initial concept from:
            https://git.concertos.live/AHD/awsmfunc/src/branch/autodeband/awsmfunc/detect.py#L463-L645

        Requirements:
            Plugins:
                https://github.com/AkarinVS/vapoursynth-plugin

            Modules:
                https://gitlab.com/Ututu/adptvgrnmod
                https://github.com/HomeOfVapourSynthEvolution/havsfunc
                https://github.com/Irrational-Encoding-Wizardry/vs-debandshit

        :param clip:            Clip to be processed.
        :param cambi_thr:       CAMBI threshold for processing.
                                Defaults to 12.0.
        :param cambi_scale:     Multiplication of CAMBI score passed to function.
                                Higher values will result in a stronger median strength.
                                Defaults to 1.2.
        :param min_thr:         Lower deband threshold.
                                Defaults to 24 (fk3db).
        :param max_thr:         Upper deband threshold.
                                Defaults to 48 (fk3db).
        :param steps:           Number of spawned filters.
                                Defaults to 4.
        :param grain_thrs:      Grain coefficients that will be passed to GrainFactory3.
                                Higher means less grain will be applied. None to disable grain.
                                Defaults to None
        :param debander:        Call a custom debandshit debanding function.
                                Function should take `clip` and `threshold` parameters.
                                Threshold is dynamically generated as per usual. Use your own mask.
                                Defaults to None.
        :param ref:             Ref clips which gets used to compute CAMBI calculations.
                                Defaults to None.
        :param downsample_h:    Decrease CAMBI CPU usage by downsampling input to desired resolution.
                                Defaults to None.
        :param debug:           A tuple of booleans.
                                Set first value to True to show relevant frame properties.
                                Set second value to True to ouput CAMBI's masks.
                                Defaults to (False, False).
        :param debander_args:   Args passed to the debandshit debander.
        :param adptvgr_args:    Adaptive grain args, dict.
                                Can pass parameters such as `static` (bool).
        :param cambi_kwargs:    Kwargs values passed to core.akarin.Cambi.
                                Can pass parameteres such as:
                                    `topk` (default: 0.1)
                                    `tvi_threshold` (default: 0.012)
    """
    try:
        from havsfunc import GrainFactory3
    except ModuleNotFoundError:
        raise ModuleNotFoundError('auto_deband: missing dependency `havsfunc`')

    if debander:
        debander_func = debander
    else:
        try:
            from debandshit import f3kbilateral
        except ModuleNotFoundError:
            raise ModuleNotFoundError('auto_deband: missing dependency `debandshit`')
        debander_func = cast(DebanderFN, f3kbilateral)

    global __auto_deband_cache
    assert clip.format

    cfamily = clip.format.color_family

    if cfamily not in (vs.GRAY, vs.YUV):
        raise ValueError("auto_deband: only YUV and GRAY clips are supported")

    is_gray = cfamily is vs.GRAY

    cambi_args = dict(topk=0.1, tvi_threshold=0.012) | cambi_kwargs | dict(scores=True)
    adptvgr_args = dict(lo=18, hi=240) | adptvgr_args

    catrom = Catrom(dither_type=DitherType.ERROR_DIFFUSION)

    clip16 = depth(clip, 16, dither_type=DitherType.ERROR_DIFFUSION)

    ref = ref or clip

    cache_key = '_'.join(map(str, map(hash, {
        ref, frozenset(cambi_args.items()), frozenset(cambi_args.values()), downsample_h
    })))

    if cache_key in __auto_deband_cache:
        cambi, cambi_masks, banding_mask, graining_mask = __auto_deband_cache[cache_key]
    else:
        ref16 = depth(ref, 16, dither_type=DitherType.ERROR_DIFFUSION)

        ref16 = get_y(ref16).std.Limiter(16 << 8, 235 << 8)

        if downsample_h:
            ref16 = Lanczos(0, dither_type=DitherType.ERROR_DIFFUSION).scale(
                ref16, get_w(downsample_h, ref16.width / ref16.height), downsample_h
            )

        ref10 = depth(ref16, 10, dither_type=DitherType.ORDERED)

        cambi = ref10.akarin.Cambi(**cambi_args)  # type: ignore

        cambi_masks = [
            catrom.scale(
                cambi.std.PropToClip('CAMBI_SCALE%d' % i), clip16.width, clip16.height
            ) for i in range(5)
        ]

        banding_mask = combine(
            cambi_masks, ExprOp.ADD, zip(range(1, 6), ExprOp.LOG, ExprOp.MUL),
            expr_suffix=[ExprOp.SQRT, 2, ExprOp.LOG, ExprOp.MUL]
        ).std.Convolution([1, 2, 1, 2, 4, 2, 1, 2, 1])

        graining_mask = combine(
            cambi_masks, ExprOp.ADD, zip(
                range(1, 6), cycle({2}), ExprOp.LOG, ExprOp.MUL, ExprOp.MUL
            ), expr_suffix=[ExprOp.SQRT, 2, ExprOp.LOG, ExprOp.MUL]
        )

        banding_mask, graining_mask = [
            depth(clip, 16, dither_type=DitherType.NONE) for clip in [banding_mask, graining_mask]
        ]

        n_d = round(clip.height / 1080 * 10)

        graining_mask = iterate(graining_mask, core.std.Minimum, round(n_d / 3))
        graining_mask = iterate(graining_mask, core.std.Maximum, n_d)
        graining_mask = graining_mask.bilateral.Gaussian(5)
        graining_mask = combine([graining_mask, banding_mask], ExprOp.ADD)
        graining_mask = graining_mask.bilateral.Gaussian(5)

        __auto_deband_cache[cache_key] = (cambi, cambi_masks, banding_mask, graining_mask)

    props_clip = clip16.std.CopyFrameProps(cambi)

    def _perform_graining(deband: vs.VideoNode, threshold: SupportsFloat) -> vs.VideoNode:
        assert grain_thrs

        gkwargs = {f"g{i}str": float(threshold) * thr / 100 for i, thr in enumerate(grain_thrs, 1)}

        yuv = join([deband] * 3) if is_gray else deband

        grained = adaptive_grain(
            yuv, grainer=lambda str_luma, str_chr, static, seed: partial(GrainFactory3, **gkwargs), **adptvgr_args
        )

        grain = get_y(grained) if is_gray else grained

        return deband.std.MaskedMerge(grain, graining_mask)

    def _perform_deband(threshold: SupportsFloat) -> vs.VideoNode:
        deband = debander_func(clip=clip16, threshold=threshold, **debander_args)

        deband = clip16.std.MaskedMerge(deband, banding_mask)

        return deband if grain_thrs is None else _perform_graining(deband, threshold)

    thr_types = {type(min_thr), type(max_thr)}

    thr_type = int if int in thr_types else float if float in thr_types else int

    delta = (max_thr - min_thr) / (steps - 1)

    thresholds = list(map(thr_type, [min_thr + i * delta for i in range(steps)]))

    deband_clips = list(map(_perform_deband, thresholds))

    thresholds, deband_clips = [0, *thresholds], [clip16, *deband_clips]

    debug_props = [
        'CAMBI', 'resolution', 'deband_str', 'approx_val',
        *({} if grain_thrs is None else [
            f"g{i+1}str" for i in range(len(grain_thrs))
        ])
    ]

    def _select_deband(n: int, f: vs.VideoFrame) -> vs.VideoNode:
        nonlocal thresholds, props_clip, deband_clips

        cambi_val = get_prop(f.props, 'CAMBI', float, None, 0.0)

        score = cambi_val * cambi_scale if cambi_val >= cambi_thr else 0

        approx_val, approx_idx = min(
            (abs(thr - score), i) for (i, thr) in enumerate(thresholds)
        )

        vals = [
            cambi_val, downsample_h or clip.height, score, approx_val,
            *({} if grain_thrs is None else [score / thr for thr in grain_thrs])
        ]

        deb_clip = deband_clips[approx_idx]

        for prop, val in zip(debug_props, vals):
            deb_clip = deb_clip.std.SetFrameProp(prop, **{type(val).__name__ + "val": val})

        return deb_clip

    process = props_clip.std.FrameEval(_select_deband, props_clip, props_clip)

    if any(debug):
        if debug[0]:
            process = process.text.FrameProps(debug_props)

        if debug[1]:
            set_output(banding_mask)
            set_output(graining_mask)

            for i, cmask in enumerate(cambi_masks):
                set_output(cmask, 'Cambi Mask - %s' % i)

    return depth(process, get_depth(clip))
