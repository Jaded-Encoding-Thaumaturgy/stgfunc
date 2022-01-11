import os
import lvsfunc as lvf
import kagefunc as kgf
import vapoursynth as vs
from pathlib import Path
from vsutil import iterate
from lvsfunc.types import VSFunction
from typing import NamedTuple, List, Optional

from .misc import source as stgsource

core = vs.core


class MaskCredit(NamedTuple):
    mask: vs.VideoNode
    start_frame: int
    end_frame: int


def perform_masks_credit(path: Path) -> List[MaskCredit]:
    if not os.path.isdir(path):
        raise ValueError("stgfunc.mask.perform_mask_credit: 'path' must be an existing path!")

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
    mapfunc: Optional[VSFunction] = None
) -> vs.VideoNode:
    manual_masks = perform_masks_credit(Path(path))

    for mask in manual_masks:
        maskclip = to_gray(mask.mask, src)
        maskclip = mapfunc(maskclip) if mapfunc else maskclip.std.Binarize()
        clip = lvf.rfs(clip, core.std.MaskedMerge(clip, src, maskclip), [(mask.start_frame, mask.end_frame)])

    return clip


def get_manual_mask(clip: vs.VideoNode, path: str, mapfunc: Optional[VSFunction] = None) -> vs.VideoNode:
    mask = MaskCredit(stgsource(path), 0, 0)

    maskclip = to_gray(mask.mask, clip)

    return mapfunc(maskclip) if mapfunc else maskclip.std.Binarize()


def generate_detail_mask(clip: vs.VideoNode, thr: float = 0.015) -> vs.VideoNode:
    general_mask = lvf.mask.detail_mask(clip, rad=1, brz_a=1, brz_b=24.3 * thr)

    return core.std.Expr([
        core.std.Expr([
            lvf.mask.detail_mask(clip, brz_a=1, brz_b=2 * thr),
            iterate(general_mask, core.std.Maximum, 3).std.Maximum().std.Inflate()
        ], "x y -"), general_mask.std.Maximum()
    ], "x y -")


def tcanny(clip: vs.VideoNode, thr: float, openCL: bool = False) -> vs.VideoNode:
    gaussian = clip.bilateral.Gaussian(1)
    msrcp = core.retinex.MSRCP(gaussian, sigma=[50, 200, 350], upper_thr=thr)

    params = dict(mode=1, sigma=1)

    tcunnied = msrcp.tcanny.TCannyCL(**params) if openCL else msrcp.tcanny.TCanny(**params)

    return tcunnied.std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])


def linemask(clip_y: vs.VideoNode) -> vs.VideoNode:
    return core.std.Expr([
        kgf.kirsch(clip_y),
        tcanny(clip_y, 0.000125),
        tcanny(clip_y, 0.0025),
        tcanny(clip_y, 0.0055),
        generate_detail_mask(clip_y, 0.011),
        generate_detail_mask(clip_y, 0.013)
    ], "x y + z a + + b c + +")
