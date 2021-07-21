import EoEfunc as eoe
import vapoursynth as vs
from typing import Union, Tuple

IntUnionChannelsType = Union[int, Tuple[int, int], Tuple[int, int, int]]  # pylint: disable=unsubscriptable-object
FloatUnionChannelsType = Union[float, Tuple[float, float], Tuple[float, float, float]]  # pylint: disable=unsubscriptable-object


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


def checkLastTwo(param: Union[IntUnionChannelsType, FloatUnionChannelsType]):  # pylint: disable=unsubscriptable-object
  return param[-2] == param[-1]


def checkSimilarClips(clipa: vs.VideoNode, clipb: vs.VideoNode):
  return clipa.height != clipb.height or clipa.width != clipb.width or eoe.format.get_format(clipa, "8") != eoe.format.get_format(clipb, "8")
