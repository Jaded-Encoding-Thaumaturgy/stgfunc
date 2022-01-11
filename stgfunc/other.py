import vapoursynth as vs
from vsutil import fallback
from lvsfunc.util import get_prop
from typing import List, Callable, Iterable, Protocol, Sequence, TypeVar, Union, Tuple

core = vs.core


# Zastin
def mt_xxpand_multi(clip, sw=1, sh=None, mode='square', planes=None, start=0, M__imum=core.std.Maximum, **params):
  sh = fallback(sh, sw)
  planes = list(range(clip.format.num_planes)) if planes is None else [planes] if isinstance(planes, int) else planes

  if mode == 'ellipse':
    coordinates = [[1] * 8, [0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0]]
  elif mode == 'losange':
    coordinates = [[0, 1, 0, 1, 1, 0, 1, 0]] * 3
  else:
    coordinates = [[1] * 8] * 3

  clips = [clip]

  end = min(sw, sh) + start

  for x in range(start, end):
    clips += [M__imum(clips[-1], coordinates=coordinates[x % 3], planes=planes, **params)]

  for x in range(end, end + sw - sh):
    clips += [M__imum(clips[-1], coordinates=[0, 0, 0, 1, 1, 0, 0, 0], planes=planes, **params)]

  for x in range(end, end + sh - sw):
    clips += [M__imum(clips[-1], coordinates=[0, 1, 0, 0, 0, 0, 1, 0], planes=planes, **params)]

  return clips


T = TypeVar('T')


class _CompFunction(Protocol):
  def __call__(self, __iterable: Iterable[T], *, key: Callable[[T], float]) -> T:
    ...


def bestframeselect(
    clips: Sequence[vs.VideoNode], ref: vs.VideoNode,
    stat_func: Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode] = core.std.PlaneStats,
    prop: str = 'PlaneStatsDiff', comp_func: _CompFunction = max, debug: Union[bool, Tuple[bool, int]] = False
) -> vs.VideoNode:
  """
  Rewritten from https://github.com/po5/notvlc/blob/master/notvlc.py#L23.

  Picks the 'best' clip for any given frame using stat functions.
  clips: list of clips
  ref: reference clip, e.g. core.average.Mean(clips) / core.median.Median(clips)
  stat_func: function that adds frame properties
  prop: property added by stat_func to compare
  comp_func: function to decide which clip to pick, e.g. min, max
  debug: display values of prop for each clip, and which clip was picked, optionally specify alignment
  """
  diffs = [stat_func(clip, ref) for clip in clips]
  indices = list(range(len(diffs)))
  do_debug, alignment = debug if isinstance(debug, tuple) else (debug, 7)

  def _select(n: int, f: List[vs.VideoFrame]) -> vs.VideoNode:
    scores = [
        get_prop(diff, prop, float) for diff in f
    ]

    best = comp_func(indices, key=lambda i: scores[i])

    if do_debug:
      return clips[best].text.Text(
          "\n".join([
              f"Prop: {prop}",
              *[f"{i}: {s}"for i, s in enumerate(scores)],
              f"Best: {best}"
          ]), alignment)

    return clips[best]

  return core.std.FrameEval(clips[0], _select, diffs)
