from __future__ import annotations

import inspect
import json
import mimetypes
from functools import cache
from os import R_OK, access, path
from pathlib import Path
from shutil import which
from subprocess import check_output
from typing import Any, cast

from vskernels import Bicubic
from vstools import core, depth_func, to_arr, vs

__all__ = [
    'SetsuCubic',
    'source', 'src',
    'set_output', 'output',
    'x8_SHADERS', 'x16_SHADERS', 'x56_SHADERS',
    'SSIM_DOWNSCALER_SHADERS', 'SSIM_SUPERSAMPLER_SHADERS'

]

file_headers_data = None
annoying_formats_exts = ['iso', 'vob']
index_formats_mimes = ['video/d2v', 'video/dgi']
file_headers_filename = path.join(path.dirname(path.abspath(__file__)), "__file_headers.json")


class SetsuCubic(Bicubic):
    """Bicubic b=-0.26470935063297507, c=0.7358829780174403"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(b=-0.26470935063297507, c=0.7358829780174403, **kwargs)


def checkValue(condition: bool, error_message: str) -> None:
    if condition:
        raise ValueError(error_message)


def set_output(clip: vs.VideoNode, text: bool | int | str | tuple[int, int] | tuple[int, int, str] = True) -> None:
    index = len(vs.get_outputs())

    ref_id = str(id(clip))
    arr = to_arr(text)

    if any([isinstance(x, str) for x in arr]):
        ref_name = arr[-1]
    else:
        ref_name = f"Clip {index}"

        current_frame = inspect.currentframe()

        assert current_frame
        assert current_frame.f_back

        for x in current_frame.f_back.f_locals.items():
            if (str(id(x[1])) == ref_id):
                ref_name = x[0]
                break

            ref_name = ref_name.title()
        ref_name = ref_name.title()

    if isinstance(text, tuple):
        pos, scale, title = (*text, ref_name)[:3]
    elif isinstance(text, int) and text is not True:
        pos, scale, title = (text, 2, ref_name)
    else:
        pos, scale, title = (7, 2, ref_name)

    if text:
        clip = clip.text.Text(title, pos, scale)

    clip = clip.std.SetFrameProp('Name', data=title)

    clip.set_output(index)


def source(
    file: str | Path, depth: int | None = None, ref: vs.VideoNode | None = None,
    force_lsmas: bool = False, matrix_prop: int | None = None, **index_args: Any
) -> vs.VideoNode:
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

    :param file:              Input file
    :param ref:               Use another clip as reference for the clip's format,
                                resolution, and framerate (Default: None)
    :param force_lsmas:       Force files to be imported with L-SMASH (Default: False)
    :param kwargs:            Arguments passed to the indexing filter
    :param matrix_prop:       Can either be bool or int, True to automatically detect it.

    :return:                  Vapoursynth clip representing input file
    """
    file = str(file)

    if file.startswith('file:///'):
        file = file[8::]

    mimeType, mimeName = getMimeType(file)
    extention = path.splitext(file)[1][1:].lower()

    def checkMimeExt(mType: str, ext: str) -> bool:
        return mType == mimeType and extention == ext

    checkValue(mimeType is None, "source: 'The source file format is not supported'")
    checkValue(mimeType == "audio", "source: 'Audio files are not supported'")
    checkValue(
        extention in annoying_formats_exts,
        "source: 'Please use an external indexer like d2vwitch or DGIndexNV for this file and import that'"
    )
    checkValue(not not ref and ref.format is None, "source: 'Variable-format clips not supported.'")

    if mimeType == "video":
        if force_lsmas:
            clip = core.lsmas.LWLibavSource(file, **index_args)
        elif extention == 'd2v':
            clip = core.d2v.Source(file, **index_args)
        elif extention == 'dgi':
            clip = core.dgdecodenv.DGSource(file, **index_args)
        else:
            if (extention == 'm2ts') or (mimeName in ['mpeg-tts', 'hevc', 'mpeg2', 'vc1']):
                clip = core.lsmas.LWLibavSource(file, **index_args)
            elif (
                mimeName in [
                    'h264', 'h263', 'vp8', 'mpeg1', 'mpeg4', 'ffv1'
                ] or checkMimeExt('av1', 'ivf') or checkMimeExt('vp9', 'mkv')
            ):
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
        assert ref.format

        clip = clip.std.AssumeFPS(None, ref.fps.numerator, ref.fps.denominator)
        clip = clip.resize.Bicubic(
            ref.width, ref.height, format=ref.format.id, matrix=matrix_prop
        )
    elif isinstance(matrix_prop, int):
        clip = clip.std.SetFrameProp('_Matrix', intval=matrix_prop)

    if depth:
        clip = depth_func(clip, depth)

    return clip


def getMimeType(filename: str, /) -> tuple[str | None, ...]:
    info = getInfoFromFileHeaders(filename)

    if info is None:
        info = getInfoFFProbe(filename)

    if info is None:
        info = getInfoFFProbe(filename, True)

    if info is None:
        infogt = mimetypes.guess_type(filename)
        if infogt[0]:
            info = tuple(infogt[0].split("/"))

    if info and info[1] is not None:
        info = tuple([info[0], info[1].split(".")[-1].rstrip("video")])

    return tuple([
        x.lower() if x else x for x in info if isinstance(x, str) or x is None  # type: ignore
    ] + [None, None])[:2]


def getInfoFFProbe(filename: str, audio: bool = False, /) -> Any:
    try:
        info = get_ffprobe_stream_properties(filename, "audio" if audio else "video")
        return (info["codec_type"], info["codec_name"])
    except Exception:
        return None


def getInfoFromFileHeaders(filename: str, /) -> tuple[str, ...] | None:
    try:
        with open(filename, "rb") as file:
            ftype, fmime = get_mime_from_file_header(cast(bytearray, file.read(128)))
            file.close()

        if not ftype or (fmime not in index_formats_mimes and ftype == "video"):
            return None

        return tuple(fmime.split("/")) if fmime else None
    except Exception:
        return None


def get_mime_from_file_header(fbytes: bytearray) -> tuple[str | None, str | None]:
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


def get_ffprobe_stream_properties(filename: str, stream_type: str = "video", ffprobe_path: str = "ffprobe") -> Any:
    checkValue(stream_type not in {'video', 'audio'}, "Stream type not supported.")

    if not which(ffprobe_path):
        raise RuntimeError('FFmpeg/FFprobe is not installed')

    if not filename or not path.isfile(filename) or not access(filename, R_OK):
        raise RuntimeError(f'File not found or inaccessible: {filename}')

    ffprobe_args = (
        '-loglevel',
        'panic',
        '-select_streams',
        f'{stream_type[0]}:0',
        '-show_streams',
        '-print_format',
        'json')

    ffprobe_output = check_output([ffprobe_path, *ffprobe_args, filename], encoding='utf-8')
    ffprobe_data = json.loads(ffprobe_output)

    if 'streams' not in ffprobe_data or not ffprobe_data['streams']:
        raise RuntimeError(f'No usable {stream_type} stream found in {filename}')

    return ffprobe_data['streams'][0]


def isMPLS(filename: str, /) -> bool:
    try:
        with open(filename, "rb") as file:
            head = file.read(8).decode("UTF-8")
            file.close()
            return "MPLS" in head
    except Exception:
        return False


@cache
def _get_shader(name: str) -> str:
    return path.join(path.dirname(__file__), f"./shaders/{name}")


x8_SHADERS = _get_shader('FSRCNNX_x2_8-0-4-1.glsl')
x16_SHADERS = _get_shader('FSRCNNX_x2_16-0-4-1.glsl')
x56_SHADERS = _get_shader('FSRCNNX_x2_56-16-4-1.glsl')
SSIM_DOWNSCALER_SHADERS = _get_shader('SSimDownscaler.glsl')
SSIM_SUPERSAMPLER_SHADERS = _get_shader('SSimSuperRes.glsl')

src = source
output = set_output
