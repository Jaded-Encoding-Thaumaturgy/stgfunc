import os
import vapoursynth as vs
from pathlib import Path
from vsutil import iterate
from typing import NamedTuple, List
from .misc import source as stgsource

core = vs.core


class MaskCredit(NamedTuple):  # pylint: disable=inherit-non-class
  mask: vs.VideoNode
  start_frame: int
  end_frame: int


def perform_masks_credit(path: Path) -> List:
  if not os.path.isdir(path):
    raise "stgfunc.mask.perform_mask_credit: 'path' must be an existing path!"

  masks = []

  for mask in path.glob('*'):
    ranges = str(mask.stem).split('_')
    end = int(ranges[-1])
    start = int(ranges[-2]) if len(ranges) > 2 else end

    masks.append(MaskCredit(stgsource(str(mask)), start, end))
  return masks


def to_gray(clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
  import mvsfunc as mvf

  clip = core.std.AssumeFPS(clip, ref)
  return core.resize.Point(clip, format=vs.GRAY16, matrix_s=mvf.GetMatrix(ref))


def manual_masking(clip: vs.VideoNode, src: vs.VideoNode, path: str, mapfunc=None):
  import lvsfunc as lvf

  manual_masks = perform_masks_credit(Path(path))

  for mask in manual_masks:
    maskclip = to_gray(mask.mask, src)
    maskclip = mapfunc(maskclip) if mapfunc else maskclip.std.Binarize()
    clip = lvf.rfs(clip, core.std.MaskedMerge(clip, src, maskclip), [(mask.start_frame, mask.end_frame)])

  return clip


def get_manual_mask(clip: vs.VideoNode, path: str, mapfunc=None):
  mask = MaskCredit(stgsource(str(Path(path))))

  maskclip = to_gray(mask.mask, clip)

  return mapfunc(maskclip) if mapfunc else maskclip.std.Binarize()


def generate_detail_mask(clip: vs.VideoNode, thr: float = 0.015) -> vs.VideoNode:
  import lvsfunc as lvf

  general_mask = lvf.mask.detail_mask(clip, rad=1, radc=1, brz_a=1, brz_b=24.3 * thr)

  return core.std.Expr([
      core.std.Expr([
          lvf.mask.detail_mask(clip, brz_a=1, brz_b=2 * thr),
          iterate(general_mask, core.std.Maximum, 3).std.Maximum().std.Inflate()
      ], "x y -"), general_mask.std.Maximum()
  ], "x y -")


def tcanny(clip: vs.VideoNode, thr: float):
  gaussian = clip.bilateral.Gaussian(1)
  msrcp = core.retinex.MSRCP(gaussian, sigma=[50, 200, 350], upper_thr=thr)
  return msrcp.tcanny.TCanny(mode=1, sigma=1).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])


def linemask(clip_y: vs.VideoNode):
  import kagefunc as kgf

  return core.std.Expr([
      kgf.kirsch(clip_y),
      tcanny(clip_y, 0.000125),
      tcanny(clip_y, 0.0025),
      tcanny(clip_y, 0.0055),
      generate_detail_mask(clip_y, 0.011),
      generate_detail_mask(clip_y, 0.013)
  ], "x y + z a + + b c + +")
