from __future__ import annotations


from typing import Type

import vapoursynth as vs

from ..types import T

core = vs.core


def get_prop(
    obj: vs.FrameProps | vs.VideoFrame | vs.AudioFrame | vs.VideoNode | vs.AudioNode,
    key: str, t: Type[T], default: T | None = None
) -> T:
    """
    Gets FrameProp ``prop`` from frame ``frame`` with expected type ``t`` to satisfy the type checker.

    :param frame:       Frame containing props
    :param key:         Prop to get
    :param t:           Type of prop
    :param default:     Fallback value

    :return:            frame.prop[key]
    """

    if isinstance(obj, vs.FrameProps):
        props = obj
    elif isinstance(obj, vs.VideoFrame) or isinstance(obj, vs.AudioFrame):
        props = obj.props
    else:
        props = obj.get_frame(0).props

    try:
        prop = props[key]

        if not isinstance(prop, t):
            raise TypeError

        return prop
    except KeyError:
        if default is None:
            raise KeyError(f"get_prop: 'Key {key} not present in props!'")
    except TypeError:
        if default is None:
            raise ValueError(f"get_prop: 'Key {key} did not contain expected type: Expected {t} got {type(prop)}!'")
    finally:
        if default is not None:
            return default


def get_color_range(clip: vs.VideoNode) -> vs.ColorRange:
    return vs.ColorRange(get_prop(clip, '_ColorRange', int, 1))
