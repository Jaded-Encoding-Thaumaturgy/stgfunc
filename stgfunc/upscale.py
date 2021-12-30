import lvsfunc as lvf
from typing import Dict
import vapoursynth as vs
from functools import cache
from os import PathLike, path

core = vs.core


@cache
def _get_shader(name: str):
  return path.join(path.dirname(__file__), rf".\.shaders\{name}")


_SHADERS = _get_shader('FSRCNNX_x2_56-16-4-1.glsl')


@lvf.misc.chroma_injector
@lvf.misc.allow_variable(width=1920, height=1080)
def upscale_rescale(clip: vs.VideoNode, width: int = 1920, height: int = 1080) -> vs.VideoNode:
  return upscale(clip, width, height)


def upscale(
    clip: vs.VideoNode, width: int = 1920, height: int = 1080, SSIM: bool = False,
    weight: float = 1 / 2, SHADERS: PathLike = _SHADERS, ssim_kwargs: Dict[str, any] = {}
) -> vs.VideoNode:
  import muvsfunc as mvf
  from nnedi3_rpow2 import nnedi3_rpow2

  nnedi3 = nnedi3_rpow2(clip, rfactor=2, kernel="spline64", nns=3, nsize=4, qual=2)
  nnedi3 = core.resize.Bicubic(nnedi3, clip.width * 2, clip.height * 2)

  fsrcnnx = fsrcnnx_upscale(clip, nnedi3, SHADERS=SHADERS)

  merge = core.std.Merge(nnedi3, fsrcnnx, weight)

  if not SSIM:
    return merge.resize.Bicubic(width, height, format=clip.format)

  ssim_down_args = dict(smooth=0, format=clip.format)
  ssim_down_args |= ssim_kwargs

  return mvf.SSIM_downsample(merge, width, height, **ssim_down_args)


def fsrcnnx_upscale(clip: vs.VideoNode, ref: vs.VideoNode = None, width: int = 1920, height: int = 1080, SHADERS: PathLike = _SHADERS) -> vs.VideoNode:
  fsrcnnx = clip.resize.Point(format=vs.YUV444P16, dither_type=None)
  fsrcnnx = core.placebo.Shader(fsrcnnx, width=clip.width * 2, height=clip.height * 2, shader=SHADERS, filter='box')

  if ref is not None:
    return fsrcnnx.resize.Bicubic(ref.width, ref.height, format=ref.format)
  else:
    return fsrcnnx.resize.Bicubic(width, height, format=clip.format)
