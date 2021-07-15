from os import PathLike
import lvsfunc as lvf
import vapoursynth as vs
from nnedi3_rpow2 import nnedi3_rpow2

core = vs.core

_SHADERS = r".\.shaders\FSRCNNX_x2_56-16-4-1.glsl"


@lvf.misc.chroma_injector
@lvf.misc.allow_variable(width=1920, height=1080)
def upscale_rescale(clip, width: int = 1920, height: int = 1080) -> vs.VideoNode:
  return upscale(clip, width, height)

# mostly stolen code from Light and Vardë


def upscale(clip, width: int = 1920, height: int = 1080, SHADERS: PathLike = _SHADERS) -> vs.VideoNode:
  nn3 = nnedi3_rpow2(clip, rfactor=2, kernel="spline64", nns=3, nsize=4, qual=2)
  nn3 = core.resize.Bicubic(nn3, clip.width * 2, clip.height * 2)

  fsrcnnx = clip.resize.Point(format=vs.YUV444P16, dither_type=None)
  fsrcnnx = core.placebo.Shader(fsrcnnx, width=clip.width * 2, height=clip.height * 2, shader=SHADERS, filter='box')\
      .resize.Bicubic(nn3.width, nn3.height, format=nn3.format)

  merge = core.std.Merge(nn3, fsrcnnx, weight=1 / 2)

  return merge.resize.Bicubic(width, height, format=clip.format)


def fsrcnnx_upscale(clip, width: int = 1920, height: int = 1080, SHADERS: PathLike = _SHADERS) -> vs.VideoNode:
  fsrcnnx = clip.resize.Point(format=vs.YUV444P16, dither_type=None)
  fsrcnnx = core.placebo.Shader(fsrcnnx, width=clip.width * 2, height=clip.height * 2, shader=SHADERS, filter='box')

  return fsrcnnx.resize.Bicubic(width, height, format=clip.format)
