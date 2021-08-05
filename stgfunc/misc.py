import json
import vsutil
import inspect
import mimetypes
import lvsfunc as lvf
import vapoursynth as vs
from enum import IntEnum
from shutil import which
from .helpers import checkValue
from os import access, path, R_OK
from subprocess import check_output
from typing import Tuple, Any, Optional, Union

core = vs.core

file_headers_data = None
annoying_formats_exts = ['iso', 'vob']
index_formats_mimes = ['video/d2v']
file_headers_filename = path.join(path.dirname(path.abspath(__file__)), "__file_headers.json")


def set_output(clip: vs.VideoNode, text: bool = True):
  index = len(vs.get_outputs()) + 1

  if text:
    ref_id = str(id(clip))
    ref_name = f"Clip {index}"
    for x in inspect.currentframe().f_back.f_locals.items():
      if (str(id(x[1])) == ref_id):
        ref_name = x[0]
        break

    clip = clip.text.Text(ref_name.title(), 7, 2)

  clip.set_output(index)


def source(file: str, depth: Optional[int] = None, ref: Optional[vs.VideoNode] = None, force_lsmas: bool = False,
           mpls: bool = False, mpls_playlist: int = 0, mpls_angle: int = 0, matrix_prop: Union[bool, int] = None, **index_args: Any) -> vs.VideoNode:
  """
  Generic clip import function.
  Automatically determines if ffms2 or L-SMASH should be used to import a clip, but L-SMASH can be forced.
  It also automatically determines if an image has been imported.
  You can set its fps using 'fpsnum' and 'fpsden', or using a reference clip with 'ref'.

  Alias for this function is `stgfunc.src`.

  Originally by LightArrowEXE @ lvsfunc, modified to be more accurate.

  Dependencies:
  * ffmpeg & ffprobe
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
  :param matrix_prop:       Can either be bool or int, True to automatically detect it.

  :return:                  Vapoursynth clip representing input file
  """

  if file.startswith('file:///'):
    file = file[8::]

  mimeType, mimeName = getMimeType(file)
  extention = path.splitext(file)[1][1:].lower()

  def checkMimeExt(mType, ext):
    return mType == mimeType and extention == ext

  checkValue(mimeType is None, "source: 'The source file format is not supported'")
  checkValue(mimeType == "audio", "source: 'Audio files are not supported'")
  checkValue(isMPLS(file) and not mpls, "source: 'Please set \"mpls = True\" and give a path to the base Blu-ray directory when trying to load in mpls files'")
  checkValue(extention in annoying_formats_exts, "source: 'Please use an external indexer like d2vwitch or DGIndexNV for this file and import that'")
  checkValue(ref and ref.format is None, "source: 'Variable-format clips not supported.'")

  if mimeType == "video":
    if force_lsmas:
      clip = core.lsmas.LWLibavSource(file, **index_args)
    elif mpls:
      mpls_in = core.mpls.Read(file, mpls_playlist, mpls_angle)
      clip = core.std.Splice([core.lsmas.LWLibavSource(mpls_in['clip'][i], **index_args) for i in range(mpls_in['count'])])
    elif extention == 'd2v':
      clip = core.d2v.Source(file, **index_args)
    elif extention == 'dgi':
      clip = core.dgdecodenv.DGSource(file, **index_args)
    else:
      if (extention == 'm2ts') or (mimeName in ['mpeg-tts', 'mpeg2', 'vc1']) or checkMimeExt('hevc', 'mp4'):
        clip = core.lsmas.LWLibavSource(file, **index_args)
      elif mimeName in ['h264', 'h263', 'hevc', 'vp8', 'mpeg1', 'mpeg4', 'ffv1'] or checkMimeExt('av1', 'ivf') or checkMimeExt('vp9', 'mkv'):
        clip = core.ffms2.Source(file, **index_args)
      elif mimeName == 'mpeg1':
        clip = core.ffms2.Source(file, seekmode=0, **index_args)
      else:
        checkValue(extention == 'ts', "source: 'Please use another indexer for this .ts file'")
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

  if isinstance(matrix_prop, int) or matrix_prop is True:
    clip = clip.std.SetFrameProp('_Matrix', intval=lvf.misc.get_matrix(clip) if matrix_prop is True else matrix_prop)

  return clip


def getMimeType(filename: str, /):
  info = getInfoFromFileHeaders(filename)

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
    info = get_ffprobe_stream_properties(filename, "audio" if audio else "video")
    return (info["codec_type"], info["codec_name"])
  except Exception:
    return None


def getInfoFromFileHeaders(filename: str, /):
  try:
    with open(filename, "rb") as file:
      ftype, fmime = get_mime_from_file_header(file.read(128))
      file.close()

    if not ftype or (fmime not in index_formats_mimes and ftype == "video"):
      return None

    return tuple(fmime.split("/"))
  except Exception:
    return None


def get_mime_from_file_header(fbytes) -> Tuple[Union[str, None], Union[str, None]]:  # pylint: disable=unsubscriptable-object
  global file_headers_data

  if file_headers_data is None:
    with open(file_headers_filename) as file_headers:
      file_headers_data = json.loads(file_headers.read())

  info, max_slen = None, 0

  stream = " ".join('{:02X}'.format(byte) for byte in fbytes)

  for mimetype in file_headers_data:
    offset = mimetype["offset"] * 2 + mimetype["offset"]
    for signature in mimetype["signature"]:
      slen = len(signature)
      if (slen > max_slen) and (signature == stream[offset:slen + offset]):
        info = mimetype
        max_slen = slen

  if info is None:
    return (None, None)

  return (info["type"], info["mime"])


def get_ffprobe_stream_properties(filename: str, stream_type: str = "video", ffprobe_path="ffprobe"):
  checkValue(stream_type not in {'video', 'audio'}, "Stream type not supported.")

  if not which(ffprobe_path):
    raise RuntimeError('FFmpeg/FFprobe is not installed')

  if not filename or not path.isfile(filename) or not access(filename, R_OK):
    raise RuntimeError(f'File not found or inaccessible: {filename}')

  ffprobe_args = ('-loglevel', 'panic', '-select_streams', f'{stream_type[0]}:0', '-show_streams', '-print_format', 'json')

  ffprobe_output = check_output([ffprobe_path, *ffprobe_args, filename], encoding='utf-8')
  ffprobe_data = json.loads(ffprobe_output)

  if 'streams' not in ffprobe_data or not ffprobe_data['streams']:
    raise RuntimeError(f'No usable {stream_type} stream found in {filename}')

  return ffprobe_data['streams'][0]


def isMPLS(filename: str, /):
  try:
    with open(filename, "rb") as file:
      head = file.read(8).decode("UTF-8")
      file.close()
      return "MPLS" in head
  except Exception:
    return False
