import EoEfunc as eoe
import vapoursynth as vs
from typing import Literal
from vsutil import split, join, depth
from .helpers import checkSimilarClips, checkValue, isGray, IntUnionChannelsType, FloatUnionChannelsType, getThreeChannelsTuple, checkLastTwo

core = vs.core


def KNLMeansCL(
    clip: vs.VideoNode, trange: IntUnionChannelsType = 2, search_radius: IntUnionChannelsType = 2,
    similarity_radius: IntUnionChannelsType = 4, sigma: FloatUnionChannelsType = 1,  # pylint: disable=unsubscriptable-object
    contraSharpening: bool = False, ref_clip: vs.VideoNode = None,  # pylint: disable=unsubscriptable-object
    device_type: Literal["cpu", "gpu", "accelerator"] = "gpu", device_id: int = 0, **kwargs  # pylint: disable=unsubscriptable-object
) -> vs.VideoNode:
  """KNLMeansCL Wrapper

  :param clip: Input clip to process
  :param trange: Temporal range, (2 * d + 1) frames; before, after and current. [d](Defaults to 2)
  :param search_radius: Radius of the search window, (2 * a + 1)^2 pixels of radius. [a](Defaults to 2)
  :param similarity_radius: Radius of the similarity neighbourhood window, (2 * s + 1)^2. [s](Defaults to 4)
  :param sigma: Strength of filtering, the higher, the more noise (and potentially details) you *steamroll*. [h](Defaults to 1)
  :param ref_clip: Reference clip to do contrasharpening with. (Defaults to None)
  :param device_id: The 'device_id' + 1º device of type 'device_type' in the system. (Defaults to 0)

  :returns: Denoised clip.
  """
  trange = getThreeChannelsTuple(trange)
  search_radius = getThreeChannelsTuple(search_radius)
  similarity_radius = getThreeChannelsTuple(similarity_radius)
  sigma = getThreeChannelsTuple(sigma)

  checkValue(not isinstance(clip, vs.VideoNode), "stg.KNLMeansCL: 'clip' must be a VideoNode")
  checkValue(ref_clip is not None and not isinstance(clip, vs.VideoNode), "stg.KNLMeansCL: 'refclip' must be a VideoNode")
  checkValue(checkSimilarClips(clip, ref_clip), "stg.KNLMeansCL: 'clip' and 'refclip' must have the same dimensions, color family and subsampling")
  checkValue(any(tr < 0 for tr in trange), "stg.KNLMeansCL: 'range' must be int > 0")
  checkValue(any(sr < 0 for sr in search_radius), "stg.KNLMeansCL: 'search_radius' must be int > 0")
  checkValue(any(sr < 0 for sr in similarity_radius), "stg.KNLMeansCL: 'similarity_radius' must be int > 0")
  checkValue(any(s < 0 for s in sigma), "stg.KNLMeansCL: 'similarity_radius' must be int > 0")

  src = depth(clip, 16)

  def knlCall(clip: vs.VideoNode, index: int, planes: Literal["Y", "UV"]):  # pylint: disable=unsubscriptable-object
    knl = clip.knlm.KNLMeansCL(
        d=trange[index], a=search_radius[index], s=similarity_radius[index],
        h=sigma[index], channels=planes, device_type=device_type,
        device_id=device_id, **kwargs
    )

    ref_clip_plane = (ref_clip_planes[index] if planes == "Y" else ref_clip) if ref_clip is not None else clip

    return eoe.misc.ContraSharpening(knl, ref_clip_plane, 2) if contraSharpening else knl

  if ref_clip:
    ref_clip = depth(ref_clip, 16)
    ref_clip_planes = split(ref_clip)
  else:
    ref_clip_planes = [None, None, None]

  if isGray(src):
    return depth(knlCall(src, 0, "Y"), clip.format.bits_per_sample)

  src_planes = split(src)

  if checkLastTwo(trange) and checkLastTwo(search_radius) and checkLastTwo(similarity_radius) and checkLastTwo(sigma):
    y = knlCall(src_planes[0], 0, "Y")
    _, u, v = split(knlCall(src, 1, "UV"))
    src_planes = [y, u, v]
  else:
    src_planes = [knlCall(src_planes[i], i, "Y") for i in range(3)]

  return depth(join(src_planes), clip.format.bits_per_sample)
