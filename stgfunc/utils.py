import lvsfunc as lvf
import vapoursynth as vs
from lvsfunc.types import Range
from vsutil import depth as vdepth
from typing import Tuple, Union, List, Sequence


def replace_squaremask(clipa: vs.VideoNode, clipb: vs.VideoNode, mask_params: Tuple[int, int, int, int], ranges: Union[Range, List[Range], None]) -> vs.VideoNode:
  import kagefunc as kgf

  return lvf.rfs(
      clipa, clipa.std.MaskedMerge(
          clipb, kgf.squaremask(clipb, *mask_params)
      ), ranges
  )


def depth(*clips_depth: vs.VideoNode) -> Sequence[vs.VideoNode]:
  assert type(clips_depth[-1]) is int

  clips = [vdepth(clip, clips_depth[-1]) for clip in clips_depth[:-1]]

  return clips[0] if len(clips) == 1 else clips
