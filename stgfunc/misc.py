from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from vstools import core, depth_func, to_arr, vs, FileType, check_variable, check_perms, CustomRuntimeError

__all__ = [
    'source', 'src',
    'set_output', 'output'
]

annoying_formats_exts = ['.iso', '.vob']


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

    :param file:            Input file
    :param depth:           Output bitdepth
    :param ref:             Use another clip as reference for the clip's format,
                            resolution, and framerate (Default: None)
    :param force_lsmas:     Force files to be imported with L-SMASH (Default: False)
    :param matrix_prop:     Can either be bool or int, True to automatically detect it.
    :param kwargs:          Arguments passed to the indexing filter

    :return:                VideoNode
    """
    file = str(file)

    if file.startswith('file:///'):
        file = file[8::]

    check_perms(file, 'r', func=source)

    mime = FileType.parse(file)

    mimeType, mimeName = mime.mime.split('/')

    if ref:
        assert check_variable(ref, source)

    if mime.ext == '.ts':
        raise CustomRuntimeError('Please use another indexer for this .ts file', source)
    elif mime.ext in annoying_formats_exts:
        raise CustomRuntimeError(
            'Please use an external indexer like d2vwitch or DGIndexNV for this file and import that', source
        )

    if mimeType == "video":
        if force_lsmas:
            clip = core.lsmas.LWLibavSource(file, **index_args)
        elif mime.ext == '.d2v':
            clip = core.d2v.Source(file, **index_args)
        elif mime.ext == '.dgi':
            clip = core.dgdecodenv.DGSource(file, **index_args)
        else:
            if (mime.ext == '.m2ts') or (mimeName in ['mpeg-tts', 'hevc', 'mpeg2', 'vc1']):
                clip = core.lsmas.LWLibavSource(file, **index_args)
            elif (
                mimeName in ['h264', 'h263', 'vp8', 'mpeg1', 'mpeg4', 'ffv1'] or (
                    (mimeType == 'av1' and mime.ext == '.ivf') or (mimeType == 'vp9' and mime.ext == '.mkv')
                )
            ):
                clip = core.ffms2.Source(file, **index_args)
            elif mimeName == 'mpeg1':
                clip = core.ffms2.Source(file, seekmode=0, **index_args)
            else:
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


src = source
output = set_output
