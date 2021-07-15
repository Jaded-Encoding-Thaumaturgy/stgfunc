import os
import fleep
import vsutil
import inspect
import mimetypes
import numpy as np
from cv2 import cv2
import lvsfunc as lvf
import EoEfunc as eoe
import vapoursynth as vs
from vsutil import iterate
from typing import Sequence
from typing import Any, Optional
from videoprops import get_video_properties, get_audio_properties

core = vs.core

annoying_formats_exts = ['iso', 'vob']


def set_output(clip: vs.VideoNode, text: bool = True):
  if text:
    ref_id = str(id(clip))
    ref_name = "Clip A"
    for x in inspect.currentframe().f_back.f_locals.items():
      if (str(id(x[1])) == ref_id):
        ref_name = x[0]
        break

    clip = clip.text.Text(ref_name.title(), 7, 2)

  index = len(vs.get_outputs()) + 1
  clip.set_output(index)


def source(file: str, depth: Optional[int] = None, ref: Optional[vs.VideoNode] = None, force_lsmas: bool = False,
           mpls: bool = False, mpls_playlist: int = 0, mpls_angle: int = 0, **index_args: Any) -> vs.VideoNode:
  """
  Generic clip import function.
  Automatically determines if ffms2 or L-SMASH should be used to import a clip, but L-SMASH can be forced.
  It also automatically determines if an image has been imported.
  You can set its fps using 'fpsnum' and 'fpsden', or using a reference clip with 'ref'.

  Alias for this function is `stgfunc.src`.

  Originally by LightArrowEXE @ lvsfunc, modified to be more accurate.

  Dependencies:
  * ffms2
  * L-SMASH-Works (optional: m2ts sources)
  * d2vsource (optional: d2v sources)
  * dgdecodenv (optional: dgi sources)
  * VapourSynth-ReadMpls (optional: mpls sources)

  :param file:              Input file
  :param ref:               Use another clip as reference for the clip's format,
                              resolution, and framerate (Default: None)
  :param force_lsmas:       Force files to be imported with L-SMASH (Default: False)
  :param mpls:              Load in a mpls file (Default: False)
  :param mpls_playlist:     Playlist number, which is the number in mpls file name (Default: 0)
  :param mpls_angle:        Angle number to select in the mpls playlist (Default: 0)
  :param kwargs:            Arguments passed to the indexing filter

  :return:                  Vapoursynth clip representing input file
  """

  if file.startswith('file:///'):
    file = file[8::]

  mimeType, mimeName = getMimeType(file)
  extention = os.path.splitext(file)[1][1:].lower()

  def checkMimeExt(mType, ext):
    return mType == mimeType and extention == ext

  if mimeType is None:
    raise ValueError("source: 'The source file format is not supported'")
  elif mimeType == "audio":
    raise ValueError("source: 'Audio files are not supported'")

  # Error handling for some file types
  if isMPLS(file) and mpls == False:
    raise ValueError("source: 'Please set \"mpls = True\" and give a path to the base Blu-ray directory when trying to load in mpls files'")

  if extention in annoying_formats_exts:
    raise ValueError("source: 'Please use an external indexer like d2vwitch or DGIndexNV for this file and import that'")

  if ref and ref.format is None:
    raise ValueError("source: 'Variable-format clips not supported.'")

  if mimeType == "video":
    if force_lsmas:
      clip = core.lsmas.LWLibavSource(file, **index_args)
    elif mpls:
      mpls_in = core.mpls.Read(file, mpls_playlist, mpls_angle)
      clip = core.std.Splice([
          core.lsmas.LWLibavSource(mpls_in['clip'][i], **index_args) for i in range(mpls_in['count'])
      ])
    elif extention == 'd2v':
      clip = core.d2v.Source(file, **index_args)
    elif extention == 'dgi':
      clip = core.dgdecodenv.DGSource(file, **index_args)
    else:
      if (extention == 'm2ts') or (mimeName in ['mpeg-tts', 'mpeg2', 'vc1']) or checkMimeExt('hevc', 'mp4'):
        clip = core.lsmas.LWLibavSource(file, **index_args)
      elif mimeName in ['h264', 'h263', 'hevc', 'vp8', 'mpeg1', 'mpeg4'] or checkMimeExt('av1', 'ivf') or checkMimeExt('vp9', 'mkv'):
        clip = core.ffms2.Source(file, **index_args)
      elif mimeName == 'mpeg1':
        clip = core.ffms2.Source(file, seekmode=0, **index_args)
      else:
        if extention == 'ts':
          raise ValueError("source: 'Please use another indexer for this .ts file'")  # noqa: E501
        clip = core.lsmas.LWLibavSource(file, **index_args)
  elif mimeType == "image":
    clip = core.imwri.Read(file, **index_args)
    if ref:
      clip = clip * (ref.num_frames - 1)

  if ref:
    clip = core.std.AssumeFPS(clip, fpsnum=ref.fps.numerator, fpsden=ref.fps.denominator)

    clip = core.resize.Bicubic(clip, width=ref.width, height=ref.height, format=ref.format.id, matrix=str(lvf.misc.get_matrix(ref)))

  if (depth is not None):
    clip = vsutil.depth(clip, depth)

  return clip


def getMimeType(filename: str, /):
  info = getInfoFleep(filename)

  if info is None:
    info = getInfoFFProbe(filename)

  if info is None:
    info = getInfoFFProbe(filename, True)

  if info is None:
    info = mimetypes.guess_type(filename)
    if info[0]:
      info = tuple(info[0].split("/"))

  if info and info[1] is not None:
    info = tuple([info[0], info[1].split(".")[-1].rstrip("video")])

  return tuple(x.lower() if x else x for x in info)


def getInfoFFProbe(filename: str, audio: bool = False, /):
  try:
    info = get_video_properties(filename) if not audio else get_audio_properties(filename)
    return (info["codec_type"], info["codec_name"])
  except Exception:
    return None


def getInfoFleep(filename: str, /):
  try:
    with open(filename, "rb") as file:
      info = fleep.get(file.read(128))
      file.close()

    if not len(info.type) or "video" in info.type:
      return None

    return tuple(info.mime[0].split("/"))
  except Exception:
    return None


def isMPLS(filename: str, /):
  try:
    with open(filename, "rb") as file:
      head = file.read(8).decode("UTF-8")
      file.close()
      return "MPLS" in head
  except Exception:
    return False


# from here https://forum.doom9.org/showthread.php?p=1801023

def tonemapping_table(clip: vs.VideoNode, source_peak: int = 1000):
  c = clip

  LDR_nits = 100
  exposure_bias = source_peak / LDR_nits
  o = c
  c = core.std.Limiter(c, 0)
  r = core.std.ShufflePlanes(c, planes=[0], colorfamily=vs.GRAY)
  g = core.std.ShufflePlanes(c, planes=[1], colorfamily=vs.GRAY)
  b = core.std.ShufflePlanes(c, planes=[2], colorfamily=vs.GRAY)
  lum = core.std.Expr(clips=[r, g, b], expr="0.216 x * 0.715 y * + 0.0722 z * +")
  lum = core.std.Limiter(lum, 0)
  lum = core.std.ShufflePlanes(lum, planes=[0], colorfamily=vs.RGB)

  rr = "x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.05 + * 0.004 + x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / - 1 {exposure_bias} 0.15 {exposure_bias} * 0.05 + * 0.004 + {exposure_bias} 0.15 {exposure_bias} * 0.50 + * 0.06 + / 0.02 0.30 / - / * ".format(
      exposure_bias=exposure_bias)
  gg = "y  {exposure_bias} * 0.15 y  {exposure_bias} * * 0.05 + * 0.004 + y  {exposure_bias} * 0.15 y  {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / - 1 {exposure_bias} 0.15 {exposure_bias} * 0.05 + * 0.004 + {exposure_bias} 0.15 {exposure_bias} * 0.50 + * 0.06 + / 0.02 0.30 / - / * ".format(
      exposure_bias=exposure_bias)
  bb = "z {exposure_bias} * 0.15 z  {exposure_bias} * * 0.05 + * 0.004 + z  {exposure_bias} * 0.15 z  {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / - 1 {exposure_bias} 0.15 {exposure_bias} * 0.05 + * 0.004 + {exposure_bias} 0.15 {exposure_bias} * 0.50 + * 0.06 + / 0.02 0.30 / - / * ".format(
      exposure_bias=exposure_bias)

  r1 = core.std.Expr(
      clips=[
          r,
          g,
          b],
      expr="x y >=   y z > {rr}   z x > x y - z y - / {bb} {gg} - * {gg} +     z y > {rr} {rr}   ?  ?     ?    x z >=   x z - y z - / {gg} {bb} - * {bb} + z y >  {rr}   {rr}            ? ? ?     ".format(
          exposure_bias=exposure_bias,
          rr=rr,
          gg=gg,
          bb=bb))
  g1 = core.std.Expr(
      clips=[
          r,
          g,
          b],
      expr="x y >=   y z > y z - x z - / {rr} {bb} - * {bb} +  z x > {gg}      z y > {gg}  {gg}   ?  ?     ?    x z >=   {gg}  z y >  y x - z x - / {bb} {rr} - * {rr} +   {gg}            ? ? ?        ".format(
          exposure_bias=exposure_bias,
          rr=rr,
          gg=gg,
          bb=bb))
  b1 = core.std.Expr(
      clips=[
          r,
          g,
          b],
      expr="x y >=   y z > {bb}   z x > {bb}      z y > z y -  x y - / {rr} {gg} - * {gg} +  {bb}   ?  ?     ?    x z >=   {bb}  z y >  {bb}   z x - y x - / {gg} {rr} - * {rr} +            ? ? ?    ".format(
          exposure_bias=exposure_bias,
          rr=rr,
          gg=gg,
          bb=bb))

  crgb = core.std.ShufflePlanes(clips=[r1, g1, b1], planes=[0, 0, 0], colorfamily=vs.RGB)

  lumtm = core.std.Expr(clips=[r1, g1, b1], expr="0.216 x * 0.715 y * + 0.0722 z * +")
  lumtm = core.std.Limiter(lumtm, 0)
  lumtm = core.std.ShufflePlanes(lumtm, planes=[0], colorfamily=vs.RGB)
  clum = core.std.Expr(clips=[o, lum, lumtm], expr=" x {exposure_bias} *  y {exposure_bias} *  / z *  ".format(exposure_bias=exposure_bias))
  clum = core.std.Limiter(clum, 0)

  mask = core.std.Expr(clips=[lum, lumtm], expr=" x {exposure_bias} * y - abs {exposure_bias}  1 - /  ".format(exposure_bias=exposure_bias))
  mask = core.std.Limiter(mask, 0, 1)

  ctemp = core.std.MaskedMerge(crgb, lumtm, mask)
  c = core.std.Merge(clum, ctemp, 0.5)

  return c


def do_general_tonemapping(src: vs.VideoNode, source_peak=1000, matrix_in_s="2020ncl", transfer_in_s="st2084"):

  c = core.resize.Bicubic(
      clip=src,
      format=vs.RGBS,
      filter_param_a=0,
      filter_param_b=0.75,
      matrix_in_s=matrix_in_s,
      chromaloc_in_s="center",
      chromaloc_s="center",
      range_in_s="limited",
      dither_type="none")

  c = core.resize.Bilinear(clip=c, format=vs.RGBS, transfer_in_s=transfer_in_s, transfer_s="linear", dither_type="none", nominal_luminance=source_peak)

  c = tonemapping_table(c, source_peak=source_peak)

  c = core.resize.Bilinear(clip=c, format=vs.RGBS, primaries_in_s="2020", primaries_s="709", dither_type="none")
  c = core.resize.Bilinear(clip=c, format=vs.RGBS, transfer_in_s="linear", transfer_s="709", dither_type="none")

  c = core.resize.Bicubic(
      clip=c,
      format=vs.YUV420P8,
      matrix_s="709",
      filter_param_a=0,
      filter_param_b=0.75,
      range_in_s="full",
      range_s="limited",
      chromaloc_in_s="center",
      chromaloc_s="center",
      dither_type="none")

  return c


# By End-Of-Eternity
# https://discord.com/channels/856381934052704266/856383287672438824/859069185929248778

def Maximum(clip: vs.VideoNode, iterations: int = 1, coordinates: Sequence[int] = [1, 1, 1, 1, 1, 1, 1, 1]) -> vs.VideoNode:
  if clip.format.color_family != vs.GRAY:
    raise ValueError("This proof of concept isn't cool enough to handle multiple planes")

  if iterations < 1:
    raise ValueError("🤔")

  if len(coordinates) != 8:
    raise ValueError("coordinates must have a length of 8")

  # why don't either the stdlib or cv2 do this?
  if all(c == 0 for c in coordinates):
    return clip

  # standard library is faster for less than 5 iterations
  # cv2 seems less well optimised for non full kernels
  if iterations < 5 or (iterations < 10 and any(c != 1 for c in coordinates)):
    return iterate(clip, core.std.Maximum, iterations)

  coordinates = list(coordinates)
  element = np.array(coordinates[:4] + [1] + coordinates[4:], dtype=np.uint8).reshape(3, 3)

  def dilate_func(n, array: np.ndarray) -> np.ndarray:
    return cv2.dilate(array[:, :, 0], element, iterations=iterations)[:, :, np.newaxis]

  return eoe.vsnp.array_eval(clip, dilate_func)
