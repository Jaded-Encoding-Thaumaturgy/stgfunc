import os
import lvsfunc as lvf
import vapoursynth as vs
from pathlib import Path
from vsmask.edge import Kirsch
from lvsfunc.types import VSFunction
from vsutil import depth, iterate, get_y
from typing import NamedTuple, List, Optional, Tuple

from .misc import source as stgsource
from .utils import get_bits, combine, ExprOp

core = vs.core


class MaskCredit(NamedTuple):
    mask: vs.VideoNode
    start_frame: int
    end_frame: int


def perform_masks_credit(path: Path) -> List[MaskCredit]:
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


def _to_gray(clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    return core.std.AssumeFPS(clip, ref).resize.Point(format=vs.GRAY16)


def manual_masking(
    clip: vs.VideoNode, src: vs.VideoNode, path: str,
    mapfunc: Optional[VSFunction] = None
) -> vs.VideoNode:
    manual_masks = perform_masks_credit(Path(path))

    for mask in manual_masks:
        maskclip = _to_gray(mask.mask, src)
        maskclip = mapfunc(maskclip) if mapfunc else maskclip.std.Binarize()
        clip = lvf.rfs(clip, core.std.MaskedMerge(clip, src, maskclip), [(mask.start_frame, mask.end_frame)])

    return clip


def get_manual_mask(clip: vs.VideoNode, path: str, mapfunc: Optional[VSFunction] = None) -> vs.VideoNode:
    mask = MaskCredit(stgsource(path), 0, 0)

    maskclip = _to_gray(mask.mask, clip)

    return mapfunc(maskclip) if mapfunc else maskclip.std.Binarize()


def generate_detail_mask(clip: vs.VideoNode, thr: float = 0.015) -> vs.VideoNode:
    general_mask = lvf.mask.detail_mask(clip, rad=1, brz_a=1, brz_b=24.3 * thr)

    return combine([
        combine([
            lvf.mask.detail_mask(clip, brz_a=1, brz_b=2 * thr),
            iterate(general_mask, core.std.Maximum, 3).std.Maximum().std.Inflate()
        ], ExprOp.MIN), general_mask.std.Maximum()
    ], ExprOp.MIN)


def tcanny(clip: vs.VideoNode, thr: float, openCL: bool = False) -> vs.VideoNode:
    msrcp = clip.bilateral.Gaussian(1).retinex.MSRCP([50, 200, 350], None, thr)

    params = dict(mode=1, sigma=1)

    tcunnied = msrcp.tcanny.TCannyCL(**params) if openCL else msrcp.tcanny.TCanny(**params)

    return tcunnied.std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])


def linemask(clip_y: vs.VideoNode) -> vs.VideoNode:
    return combine([
        Kirsch().get_mask(clip_y),
        tcanny(clip_y, 0.000125),
        tcanny(clip_y, 0.0025),
        tcanny(clip_y, 0.0055),
        generate_detail_mask(clip_y, 0.011),
        generate_detail_mask(clip_y, 0.013)
    ], ExprOp.ADD)


def getCreditMask(
    clip: vs.VideoNode, ref: vs.VideoNode, thr: int,
    blur: Optional[float] = 1.65, prefilter: bool = True,
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

    bits, credit_mask = get_bits(ed_mask)
    credit_mask = iterate(credit_mask, core.std.Minimum, 6)
    credit_mask = iterate(credit_mask, lambda x: x.std.Minimum().std.Maximum(), 8)
    if expand:
        credit_mask = iterate(credit_mask, core.std.Maximum, expand)
    credit_mask = credit_mask.std.Inflate().std.Inflate().std.Inflate()

    return credit_mask if bits == 16 else depth(credit_mask, bits)


# Stolen from Light by yours truly <3
def detail_mask(
    clip: vs.VideoNode,
    sigma: float = 1.0, rxsigma: List[int] = [50, 200, 350],
    pf_sigma: Optional[float] = 1.0, brz: Tuple[int, int] = (2500, 4500),
    rg_mode: int = 17
) -> vs.VideoNode:
    bits, clip = get_bits(clip)

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
