from __future__ import annotations

import os
from pathlib import Path

import vapoursynth as vs
from vsexprtools import ExprOp, combine
from vsexprtools.types import VSFunction
from vsmask.edge import Kirsch, PrewittTCanny
from vsrgtools import removegrain
from vsutil import depth, disallow_variable_format, get_depth, get_peak_value, get_y, insert_clip, iterate

from .misc import source as stgsource
from .types import MaskCredit, Range
from .utils import expect_bits

core = vs.core


@disallow_variable_format
def adg_mask(clip: vs.VideoNode, luma_scaling: float = 8.0, relative: bool = False) -> vs.VideoNode:
    """
    Re-reimplementation of kageru's adaptive_grain mask as expr, with added `relative` param.
    Really just the same speed and everything (though, *slightly* faster in float32),
    it's just a plugin dep less in your *func if already make use of Akarin's plugin.
    For a description of the math and the general idea, see his article. \n
    https://blog.kageru.moe/legacy/adaptivegrain.html \n
    https://git.kageru.moe/kageru/adaptivegrain
    """
    y = get_y(clip).std.PlaneStats(prop='P')

    assert y.format

    peak = get_peak_value(y)

    is_integer = y.format.sample_type == vs.INTEGER

    x_string, aft_int = (f'x {peak} / ', f' {peak} *') if is_integer else ('x ', '')

    if relative:
        x_string += 'Y! Y@ 0.5 < x.PMin 0 max 0.5 / log Y@ * x.PMax 1.0 min 0.5 / log Y@ * ? '

    x_string += '0 0.999 clamp X!'

    return y.akarin.Expr(
        f'{x_string} 1 X@ X@ X@ X@ X@ '
        '18.188 * 45.47 - * 36.624 + * 9.466 - * 1.124 + * - '
        f'x.PAverage 2 pow {luma_scaling} * pow {aft_int}',
    )


def perform_masks_credit(path: Path) -> list[MaskCredit]:
    if not os.path.isdir(path):
        raise ValueError("perform_mask_credit: 'path' must be an existing path!")

    files = [file.stem for file in path.glob('*')]

    clips = [stgsource(file) for file in files]
    files_ranges = [list(map(int, name.split('_'))) for name in files]

    return [
        MaskCredit(
            clip,
            ranges[-2] if len(ranges) > 2 else (end := ranges[-1]), end
        ) for clip, ranges in zip(clips, files_ranges)
    ]


def to_gray(clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    return core.std.AssumeFPS(clip, ref).resize.Point(format=vs.GRAY16)


def manual_masking(
    clip: vs.VideoNode, src: vs.VideoNode, path: str,
    mapfunc: VSFunction | None = None
) -> vs.VideoNode:
    manual_masks = perform_masks_credit(Path(path))

    for mask in manual_masks:
        maskclip = to_gray(mask.mask, src)
        maskclip = mapfunc(maskclip) if mapfunc else maskclip.std.Binarize()
        insert = clip.std.MaskedMerge(src, maskclip)
        insert = insert[mask.start_frame:mask.end_frame + 1]
        clip = insert_clip(clip, insert, mask.start_frame)

    return clip


def get_manual_mask(clip: vs.VideoNode, path: str, mapfunc: VSFunction | None = None) -> vs.VideoNode:
    mask = MaskCredit(stgsource(path), 0, 0)

    maskclip = to_gray(mask.mask, clip)

    return mapfunc(maskclip) if mapfunc else maskclip.std.Binarize()


def simple_detail_mask(
    clip: vs.VideoNode, sigma: float | None = None, rad: int = 3, brz_a: float = 0.025, brz_b: float = 0.045
) -> vs.VideoNode:
    from lvsfunc import range_mask, scale_thresh

    brz_a = scale_thresh(brz_a, clip)
    brz_b = scale_thresh(brz_b, clip)

    y = get_y(clip)

    blur = y.bilateral.Gaussian(sigma) if sigma else y

    mask_a = range_mask(blur, rad=rad).std.Binarize(brz_a)

    mask_b = PrewittTCanny().edgemask(blur).std.Binarize(brz_b)

    mask = core.akarin.Expr([mask_a, mask_b], 'x y max')

    return removegrain(removegrain(mask, 22), 11).std.Limiter()


def multi_detail_mask(clip: vs.VideoNode, thr: float = 0.015) -> vs.VideoNode:
    general_mask = simple_detail_mask(clip, rad=1, brz_a=1, brz_b=24.3 * thr)

    return combine([
        combine([
            simple_detail_mask(clip, brz_a=1, brz_b=2 * thr),
            iterate(general_mask, core.std.Maximum, 3).std.Maximum().std.Inflate()
        ], ExprOp.MIN), general_mask.std.Maximum()
    ], ExprOp.MIN)


def tcanny(clip: vs.VideoNode, thr: float) -> vs.VideoNode:
    msrcp = clip.bilateral.Gaussian(1).retinex.MSRCP([50, 200, 350], None, thr)

    tcunnied = msrcp.tcanny.TCanny(mode=1, sigma=1)

    return tcunnied.std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])


def linemask(clip_y: vs.VideoNode) -> vs.VideoNode:
    return combine([
        Kirsch().edgemask(clip_y),
        tcanny(clip_y, 0.000125),
        tcanny(clip_y, 0.0025),
        tcanny(clip_y, 0.0055),
        multi_detail_mask(clip_y, 0.011),
        multi_detail_mask(clip_y, 0.013)
    ], ExprOp.ADD)


def credit_mask(
    clip: vs.VideoNode, ref: vs.VideoNode, thr: int,
    blur: float | None = 1.65, prefilter: bool = True,
    expand: int = 8
) -> vs.VideoNode:
    from vardefunc.mask import Difference

    if blur is None or blur <= 0:
        blur_src, blur_ref = clip, ref
    else:
        blur_src = clip.bilateral.Gaussian(blur)
        blur_ref = ref.bilateral.Gaussian(blur)

    ed_mask = Difference().creditless(
        blur_src[0] + blur_src, blur_src, blur_ref,
        start_frame=0, thr=thr, prefilter=prefilter
    )

    bits, credit_mask = expect_bits(ed_mask)
    credit_mask = iterate(credit_mask, core.std.Minimum, 6)
    credit_mask = iterate(credit_mask, lambda x: core.std.Minimum(x).std.Maximum(), 8)
    if expand:
        credit_mask = iterate(credit_mask, core.std.Maximum, expand)
    credit_mask = credit_mask.std.Inflate().std.Inflate().std.Inflate()

    return credit_mask if bits == 16 else depth(credit_mask, bits)


# Stolen from Light by yours truly <3
def detail_mask(
    clip: vs.VideoNode,
    sigma: float = 1.0, rxsigma: list[int] = [50, 200, 350],
    pf_sigma: float | None = 1.0, brz: tuple[int, int] = (2500, 4500),
    rg_mode: int = 17
) -> vs.VideoNode:
    bits, clip = expect_bits(clip)

    clip_y = get_y(clip)

    pf = core.bilateral.Gaussian(clip_y, pf_sigma) if pf_sigma else clip_y
    ret = pf.retinex.MSRCP(rxsigma, None, 0.005)

    blur_ret = core.bilateral.Gaussian(ret, sigma)
    blur_ret_diff = combine([blur_ret, ret], ExprOp.SUB).std.Deflate()
    blur_ret_brz = iterate(blur_ret_diff, core.std.Inflate, 4)
    blur_ret_brz = blur_ret_brz.std.Binarize(brz[0]).morpho.Close(size=8)

    prewitt_mask = clip_y.std.Prewitt().std.Binarize(brz[1]).std.Deflate().std.Inflate()
    prewitt_brz = prewitt_mask.std.Binarize(brz[1]).morpho.Close(size=4)

    merged = combine([blur_ret_brz, prewitt_brz], ExprOp.ADD)
    rm_grain = core.rgvs.RemoveGrain(merged, rg_mode)

    return rm_grain if bits == 16 else depth(rm_grain, bits)


@disallow_variable_format
def squaremask(clip: vs.VideoNode, width: int, height: int, offset_x: int, offset_y: int) -> vs.VideoNode:
    assert clip.format

    bits = get_depth(clip)
    src_w, src_h = clip.width, clip.height

    mask_format = clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0)

    white = 1 if mask_format.sample_type == vs.FLOAT else (1 << bits) - 1

    white_clip = core.std.BlankClip(clip, width, height, mask_format.id, 1, color=white, keep=True)

    padded = white_clip.std.AddBorders(
        offset_x, src_w - width - offset_x,
        offset_y, src_h - height - offset_y
    )

    return padded * clip.num_frames


@disallow_variable_format
def replace_squaremask(
    clipa: vs.VideoNode, clipb: vs.VideoNode, mask_params: tuple[int, int, int, int],
    ranges: Range | list[Range] | None = None,
    blur_sigma: int | None = None, invert: bool = False
) -> vs.VideoNode:
    from lvsfunc import replace_ranges

    assert clipa.format
    assert clipb.format

    mask = squaremask(clipb, *mask_params)

    if invert:
        mask = mask.std.InvertMask()

    if blur_sigma is not None:
        if clipa.format.bits_per_sample == 32:
            mask = mask.bilateralgpu.Bilateral(blur_sigma)
        else:
            mask = mask.bilateral.Gaussian(blur_sigma)

    merge = clipa.std.MaskedMerge(clipb, mask)

    return replace_ranges(clipa, merge, ranges) if ranges else merge


def freeze_replace_mask(
    mask: vs.VideoNode, insert: vs.VideoNode,
    mask_params: tuple[int, int, int, int], frame: int, frame_range: tuple[int, int]
) -> vs.VideoNode:
    start, end = frame_range

    masked_insert = replace_squaremask(mask[frame], insert[frame], mask_params)

    return insert_clip(mask, masked_insert * (end - start + 1), start)
