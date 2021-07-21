import numpy as np
import cv2
import EoEfunc as eoe
import vapoursynth as vs
from vsutil import iterate
from typing import Sequence

core = vs.core

#################################################################
### Code I haven't written or modified in any significant way ###
#################################################################

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
  lum = limitRGB(r, g, b)

  rr = "x {exposure_bias} * 0.15 x {exposure_bias} * * 0.05 + * 0.004 + x {exposure_bias} * 0.15 x {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / - 1 {exposure_bias} 0.15 {exposure_bias} * 0.05 + * 0.004 + {exposure_bias} 0.15 {exposure_bias} * 0.50 + * 0.06 + / 0.02 0.30 / - / * ".format(exposure_bias=exposure_bias)
  gg = "y {exposure_bias} * 0.15 y {exposure_bias} * * 0.05 + * 0.004 + y {exposure_bias} * 0.15 y {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / - 1 {exposure_bias} 0.15 {exposure_bias} * 0.05 + * 0.004 + {exposure_bias} 0.15 {exposure_bias} * 0.50 + * 0.06 + / 0.02 0.30 / - / * ".format(exposure_bias=exposure_bias)
  bb = "z {exposure_bias} * 0.15 z {exposure_bias} * * 0.05 + * 0.004 + z {exposure_bias} * 0.15 z {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / - 1 {exposure_bias} 0.15 {exposure_bias} * 0.05 + * 0.004 + {exposure_bias} 0.15 {exposure_bias} * 0.50 + * 0.06 + / 0.02 0.30 / - / * ".format(exposure_bias=exposure_bias)

  r1 = core.std.Expr(clips=[r,g,b],expr="x y >= y z > {rr} z x > x y - z y - / {bb} {gg} - * {gg} + z y > {rr} {rr} ? ? ? x z >= x z - y z - / {gg} {bb} - * {bb} + z y > {rr} {rr} ? ? ?".format(exposure_bias=exposure_bias,rr=rr,gg=gg,bb=bb))
  g1 = core.std.Expr(clips=[r, g, b],expr="x y >= y z > y z - x z - / {rr} {bb} - * {bb} + z x > {gg} z y > {gg} {gg} ? ? ? x z >= {gg} z y > y x - z x - / {bb} {rr} - * {rr} + {gg} ? ? ? ".format(exposure_bias=exposure_bias, rr=rr, gg=gg, bb=bb))
  b1 = core.std.Expr(clips=[r,g,b], expr="x y >= y z > {bb} z x > {bb} z y > z y - x y - / {rr} {gg} - * {gg} + {bb} ? ? ? x z >= {bb} z y > {bb} z x - y x - / {gg} {rr} - * {rr} + ? ? ? ".format(exposure_bias=exposure_bias,rr=rr,gg=gg, bb=bb))


  crgb = core.std.ShufflePlanes(clips=[r1, g1, b1], planes=[0, 0, 0], colorfamily=vs.RGB)

  lumtm = limitRGB(r1, g1, b1)
  clum = core.std.Expr(clips=[o, lum, lumtm], expr=" x {exposure_bias} * y {exposure_bias} * / z * ".format(exposure_bias=exposure_bias))
  clum = core.std.Limiter(clum, 0)

  mask = core.std.Expr(clips=[lum, lumtm], expr=" x {exposure_bias} * y - abs {exposure_bias} 1 - / ".format(exposure_bias=exposure_bias))
  mask = core.std.Limiter(mask, 0, 1)

  ctemp = core.std.MaskedMerge(crgb, lumtm, mask)

  return core.std.Merge(clum, ctemp, 0.5)


def limitRGB(r, g, b):
  result = core.std.Expr(clips=[r, g, b], expr="0.216 x * 0.715 y * + 0.0722 z * +")
  result = core.std.Limiter(result, 0)
  result = core.std.ShufflePlanes(result, planes=[0], colorfamily=vs.RGB)

  return result


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

  return core.resize.Bicubic(
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

  return eoe.vsnp.array_eval(clip, lambda n, array: cv2.dilate(array[:, :, 0], element, iterations=iterations)[:, :, np.newaxis])
