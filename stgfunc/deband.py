import vapoursynth as vs
from vsutil import depth
from typing import Dict, Union, List, Any

from .utils import get_bits
from .mask import detail_mask

core = vs.core


def masked_f3kdb(
    clip: vs.VideoNode, rad: int = 16, thr: Union[int, List[int]] = 24,
    grain: Union[int, List[int]] = [12, 0], mask_args: Dict[str, Any] = {}
) -> vs.VideoNode:
    from debandshit.debanders import dumb3kdb

    bits, clip = get_bits(clip)
    clip = depth(clip, 16)

    mask_kwargs: Dict[str, Any] = dict(brz=(1000, 2750)) | mask_args

    deband_mask = detail_mask(clip, **mask_kwargs)

    deband = dumb3kdb(clip, radius=rad, threshold=thr, grain=grain, seed=69420)
    deband_masked = deband.std.MaskedMerge(clip, deband_mask)

    return deband_masked if bits == 16 else depth(deband_masked, bits)
