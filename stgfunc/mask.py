from __future__ import annotations

import os
from pathlib import Path
from typing import NamedTuple

from vsexprtools import ExprOp, combine
from vstools import (
    FrameRangeN, FrameRangesN, VSFunction, disallow_variable_format, get_depth, insert_clip, replace_ranges, vs
)

from .misc import source as stgsource

__all__ = [
    'perform_masks_credit',
    'to_gray', 'manual_masking', 'get_manual_mask',
    'tcanny', 'linemask',
    'squaremask', 'replace_squaremask',
    'freeze_replace_mask'
]


class MaskCredit(NamedTuple):
    mask: vs.VideoNode
    start_frame: int
    end_frame: int


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
