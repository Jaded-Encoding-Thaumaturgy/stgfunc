import lvsfunc as lvf
import vapoursynth as vs
from lvsfunc.types import Range
from vsutil import depth as vdepth
from typing import Tuple, Union, List, Sequence, Optional


def replace_squaremask(
    clipa: vs.VideoNode, clipb: vs.VideoNode, mask_params: Tuple[int, int, int, int],
    ranges: Union[Range, List[Range], None] = None,
    blur_sigma: Optional[int] = None, invert: bool = False
) -> vs.VideoNode:
  import kagefunc as kgf

  mask = kgf.squaremask(clipb, *mask_params)

  if invert:
    mask = mask.std.InvertMask()

  if blur_sigma is not None:
    mask = mask.bilateralgpu.Bilateral(blur_sigma) if clipa.format.bits_per_sample == 32 else mask.bilateral.Gaussian(blur_sigma)

  merge = clipa.std.MaskedMerge(clipb, mask)

  return lvf.rfs(clipa, merge, ranges) if ranges else merge


def depth(*clips_depth: vs.VideoNode) -> Sequence[vs.VideoNode]:
  assert isinstance(clips_depth[-1], int)

  clips = [vdepth(clip, clips_depth[-1]) for clip in clips_depth[:-1]]

  return clips[0] if len(clips) == 1 else clips
