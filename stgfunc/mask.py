from __future__ import annotations

import os
from pathlib import Path

from vsexprtools import ExprOp, combine
from vstools import (
    FrameRangeN, FrameRangesN, VSFunction, core, depth, disallow_variable_format, expect_bits, get_depth,
    get_peak_value, get_y, insert_clip, iterate, vs
)

from .misc import source as stgsource
from .types import MaskCredit

__all__ = [
    'adg_mask',
    'perform_masks_credit',
    'to_gray', 'manual_masking', 'get_manual_mask',
    'tcanny', 'linemask',
    'detail_mask',
    'squaremask', 'replace_squaremask',
    'freeze_replace_mask'
]


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

    x_string, aft_int = (f'x {peak} / ', f' {peak} * 0.5 +') if is_integer else ('x ', '')

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
    return clip.std.AssumeFPS(ref).resize.Point(format=vs.GRAY16)


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


def tcanny(clip: vs.VideoNode, thr: float) -> vs.VideoNode:
    msrcp = clip.bilateral.Gaussian(1).retinex.MSRCP([50, 200, 350], None, thr)

    tcunnied = msrcp.tcanny.TCanny(mode=1, sigma=1)

    return tcunnied.std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])


def linemask(clip_y: vs.VideoNode) -> vs.VideoNode:
    try:
        from vsscale import multi_detail_mask
    except ModuleNotFoundError:
        raise ModuleNotFoundError('linemask: missing dependency `vsrgtools`')

    try:
        from vsmask.edge import Kirsch
    except ModuleNotFoundError:
        raise ModuleNotFoundError('linemask: missing dependency `vsmask`')

    return combine([
        Kirsch().edgemask(clip_y),
        tcanny(clip_y, 0.000125),
        tcanny(clip_y, 0.0025),
        tcanny(clip_y, 0.0055),
        multi_detail_mask(clip_y, 0.011),
        multi_detail_mask(clip_y, 0.013)
    ], ExprOp.ADD)


# Stolen from Light by yours truly <3
def detail_mask(
    clip: vs.VideoNode,
    sigma: float = 1.0, rxsigma: list[int] = [50, 200, 350],
    pf_sigma: float | None = 1.0, brz: tuple[int, int] = (2500, 4500),
    rg_mode: int = 17
) -> vs.VideoNode:
    clip, bits = expect_bits(clip)

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

    white_clip = clip.std.BlankClip(width, height, mask_format.id, 1, color=white, keep=True)

    padded = white_clip.std.AddBorders(
        offset_x, src_w - width - offset_x,
        offset_y, src_h - height - offset_y
    )

    return padded * clip.num_frames


@disallow_variable_format
def replace_squaremask(
    clipa: vs.VideoNode, clipb: vs.VideoNode, mask_params: tuple[int, int, int, int],
    ranges: FrameRangeN | FrameRangesN | None = None,
    blur_sigma: float | None = None, invert: bool = False
) -> vs.VideoNode:
    try:
        from lvsfunc import replace_ranges
    except ModuleNotFoundError:
        raise ModuleNotFoundError('replace_squaremask: missing dependency `lvsfunc`')

    try:
        from vsrgtools import box_blur, gauss_blur
    except ModuleNotFoundError:
        raise ModuleNotFoundError('replace_squaremask: missing dependency `vsrgtools`')

    assert clipa.format and clipb.format

    mask = squaremask(clipb, *mask_params)

    if invert:
        mask = mask.std.InvertMask()

    if isinstance(blur_sigma, int):
        mask = box_blur(mask, blur_sigma)
    elif isinstance(blur_sigma, float):
        mask = gauss_blur(mask, blur_sigma)

    merge = clipa.std.MaskedMerge(clipb, mask)

    return replace_ranges(clipa, merge, ranges) if ranges else merge


def freeze_replace_mask(
    mask: vs.VideoNode, insert: vs.VideoNode,
    mask_params: tuple[int, int, int, int], frame: int, frame_range: tuple[int, int]
) -> vs.VideoNode:
    start, end = frame_range

    masked_insert = replace_squaremask(mask[frame], insert[frame], mask_params)

    return insert_clip(mask, masked_insert * (end - start + 1), start)
