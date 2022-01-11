import vapoursynth as vs
from typing import Union, Tuple

IntUnionChannelsType = Union[int, Tuple[int, int], Tuple[int, int, int]]
FloatUnionChannelsType = Union[float, Tuple[float, float], Tuple[float, float, float]]


def isGray(clip: vs.VideoNode) -> bool:
  return clip.format.color_family == vs.GRAY


def checkValue(condition: bool, error_message: str):
  if condition:
    raise ValueError(error_message)


def getThreeChannelsTuple(param):
  param = [param] if not isinstance(param, list) else param

  if (length := len(param)) < 3:
    param += [param[-1]] * (3 - length)

  return param


def checkLastTwo(param: Union[IntUnionChannelsType, FloatUnionChannelsType]):
  return param[-2] == param[-1]


def checkSimilarClips(clipa: vs.VideoNode, clipb: vs.VideoNode):
  return isinstance(clipa, vs.VideoNode) and isinstance(clipb, vs.VideoNode) and clipa.height == clipb.height and clipa.width == clipb.width and clipa.format.id == clipb.format.id
