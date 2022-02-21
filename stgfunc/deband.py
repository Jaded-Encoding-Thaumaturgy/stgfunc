import vapoursynth as vs
from functools import partial
from debandshit.debanders import f3kbilateral, dumb3kdb
from typing import Tuple, Any, Dict, Sequence, Union, List
from vsutil import get_y, get_w, join, depth, get_depth, Dither

from .utils import get_bits
from .mask import detail_mask


core = vs.core


def masked_f3kdb(
    clip: vs.VideoNode, rad: int = 16, threshold: Union[int, List[int]] = 24,
    grain: Union[int, List[int]] = [12, 0], mask_args: Dict[str, Any] = {}
) -> vs.VideoNode:
    bits, clip = get_bits(clip)
    clip = depth(clip, 16)

    mask_kwargs: Dict[str, Any] = dict(brz=(1000, 2750)) | mask_args

    deband_mask = detail_mask(clip, **mask_kwargs)

    deband = dumb3kdb(clip, radius=rad, threshold=threshold, grain=grain, seed=69420)
    deband_masked = deband.std.MaskedMerge(clip, deband_mask)

    return deband_masked if bits == 16 else depth(deband_masked, bits)


class DebanderFN:
    def __call__(self, clip: vs.VideoNode, threshold: int | Sequence[int], *args: Any, **kwargs: Any) -> vs.VideoNode:
        ...


def auto_deband(
    clip: vs.VideoNode, cambi_thr: float = 12.0, cambi_scale: float = 1.2,
    min_thr: int | float = 24, max_thr: int | float = 48, steps: int = 4,
    grain_thrs: Tuple[int, int, int] | None = None,
    debander: DebanderFN = partial(f3kbilateral, limflt_args={"thr": 0.3}),
    downsample_h: None | int = None, chroma: bool = False, debug: bool = False,
    adptvgr_args: Dict[str, Any] = {}, **cambi_kwargs: Dict[str, Any]
) -> vs.VideoNode:
    """
    Automated banding detection and filtering via the use of CAMBI.
    A range of potential debanding functions are spawned, of which
    an approximated value is chosen based off the score returned by CAMBI.

    Please see: https://github.com/AkarinVS/vapoursynth-plugin/wiki/CAMBI

    Function is extensible, allowing for custom functions for
    debanding and grain applied in place of defaults.
    For anime, consider either disabling the graining function, or
    or using adptvgr_kwargs={"static"=True}

    Initial concept from: https://git.concertos.live/AHD/awsmfunc/src/branch/autodeband/awsmfunc/detect.py#L463-L645

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
    :param debander:        Call a custom debanding function.
                            Function should take `clip` and `threshold` parameters.
                            Threshold is dynamically generated as per usual. Use your own mask.
                            Defaults to None.
    :param downsample_h:    Decrease CAMBI CPU usage by downsampling input to desired resolution.
                            Defaults to None.
    :param chroma:          Whether to process chroma or not.
                            Defaults to False.
    :param debug:           Show relevant frame properties.
                            Defaults to False.
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
        raise ModuleNotFoundError("auto_deband: 'missing dependency `havsfunc`'")

    try:
        from adptvgrnMod import adptvgrnMod
    except ModuleNotFoundError:
        raise ModuleNotFoundError("auto_deband: 'missing dependency `adptvgrnMod`'")

    if clip.format.color_family not in (vs.GRAY, vs.YUV):
        raise ValueError("auto_deband: only YUV and GRAY clips are supported")

    is_gray = clip.format.color_family is vs.GRAY

    cambi_args = dict(topk=0.1, tvi_threshold=0.012) | cambi_kwargs
    adptvgr_args = dict(lo=18, hi=240, grain_chroma=not is_gray or chroma) | adptvgr_args

    clip16 = depth(clip, 16)

    ref = get_y(clip16).std.Limiter(16 << 8, 235 << 8)

    if downsample_h:
        ref = ref.resize.Lanczos(
            get_w(downsample_h, ref.width / ref.height), downsample_h,
            filter_param_a=0, dither_type="error_diffusion"
        )

    ref = depth(ref, 10, dither_type=Dither.ORDERED)

    cambi = ref.akarin.Cambi(**cambi_args)

    props_clip = clip16.std.CopyFrameProps(cambi)

    def _perform_graining(clip: vs.VideoNode, threshold: float = None) -> vs.VideoNode:
        gkwargs = {f"g{i}str": threshold * thr / 100 for i, thr in enumerate(grain_thrs, 1)}

        grained = adptvgrnMod(
            join([clip] * 3) if is_gray else clip,
            grainer=lambda g: GrainFactory3(g, **gkwargs), **adptvgr_args
        )

        return get_y(grained) if is_gray else grained

    def _perform_deband(threshold: float) -> vs.VideoNode:
        deband = debander(clip=clip16, threshold=threshold)

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

        cambi_val = float(f.props['CAMBI'])  # type: ignore

        score = cambi_val * cambi_scale if cambi_val >= cambi_thr else 0

        approx_val, approx_idx = min(
            (abs(thr - score), i) for (i, thr) in enumerate(thresholds)
        )

        vals = [
            cambi_val, ref.height, score, approx_val,
            *({} if grain_thrs is None else [score / thr for thr in grain_thrs])
        ]

        deb_clip = deband_clips[approx_idx]

        for prop, val in zip(debug_props, vals):
            deb_clip = deb_clip.std.SetFrameProp(prop, **{type(val).__name__ + "val": val})

        return deb_clip

    process = props_clip.std.FrameEval(_select_deband, props_clip)

    if debug:
        process = process.text.FrameProps(debug_props)

    return depth(process, get_depth(clip))
