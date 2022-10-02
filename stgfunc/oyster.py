# type: ignore

import math
from typing import Literal

from vstools import vs, core


__all__ = [
    'Core',
    'Super', 'Basic', 'Deringing', 'Destaircase', 'Deblocking'
]

fmtc_args = dict(fulls=True, fulld=True)
msuper_args = dict(hpad=0, vpad=0, sharp=2, levels=0)
manalyze_args = dict(search=3, truemotion=False, trymany=True, levels=0, badrange=-24, divide=0, dct=0)
mrecalculate_args = dict(truemotion=False, search=3, smooth=1, divide=0, dct=0)
mdegrain_args = dict(thscd1=16711680.0, thscd2=255.0)
nnedi_args = dict(field=1, dh=True, nns=4, qual=2, etype=1, nsize=0)
dfttest_args = dict(smode=0, sosize=0, tbsize=1, tosize=0, tmode=0)


class SuperClip(vs.VideoNode):
    ...


class BasicClip(vs.VideoNode):
    ...


class Core(vs.Core):
    def __init__(self):
        self.MSuper = core.proxied.mvsf.Super
        self.MAnalyze = core.proxied.mvsf.Analyze
        self.MRecalculate = core.proxied.mvsf.Recalculate
        self.MDegrain = core.proxied.mvsf.Degrain
        self.RGB2OPP = core.proxied.bm3d.RGB2OPP
        self.OPP2RGB = core.proxied.bm3d.OPP2RGB
        self.BMBasic = core.proxied.bm3d.VBasic
        self.BMFinal = core.proxied.bm3d.VFinal
        self.Aggregate = core.proxied.bm3d.VAggregate
        self.DFTTest = core.proxied.dfttest.DFTTest
        self.KNLMeansCL = core.proxied.knlm.KNLMeansCL
        self.NNEDI = core.proxied.nnedi3.nnedi3
        self.Resample = core.proxied.fmtc.resample
        self.Expr = core.proxied.std.Expr
        self.MakeDiff = core.proxied.std.MakeDiff
        self.MergeDiff = core.proxied.std.MergeDiff
        self.Crop = core.proxied.std.CropRel
        self.CropAbs = core.proxied.std.CropAbs
        self.Transpose = core.proxied.std.Transpose
        self.BlankClip = core.proxied.std.BlankClip
        self.AddBorders = core.proxied.std.AddBorders
        self.StackHorizontal = core.proxied.std.StackHorizontal
        self.StackVertical = core.proxied.std.StackVertical
        self.MaskedMerge = core.proxied.std.MaskedMerge
        self.ShufflePlanes = core.proxied.std.ShufflePlanes
        self.SetFieldBased = core.proxied.std.SetFieldBased

    def delete(self):
        del self

    def FreqMerge(self, low: vs.VideoNode, hi: vs.VideoNode, sbsize: int = 9,
                  slocation: float = [0.0, 0.0, 0.12, 1024.0, 1.0, 1024.0]) -> vs.VideoNode:
        hif = self.MakeDiff(hi, self.DFTTest(hi, sbsize=sbsize, slocation=slocation, **dfttest_args))
        return self.MergeDiff(self.DFTTest(low, sbsize=sbsize, slocation=slocation, **dfttest_args), hif)

    def Pad(self, src: vs.VideoNode, left: int, right: int, top: int, bottom: int) -> vs.VideoNode:
        return self.Resample(src, src.width + left + right, src.height + top + bottom, -left, -top,
                             src.width + left + right, src.height + top + bottom, kernel="point", **fmtc_args)

    def NLMeans(self, src: vs.VideoNode, d: int, a: int, s: int, h: float,
                rclip: vs.VideoNode, color: bool) -> vs.VideoNode:
        def duplicate(src: vs.VideoNode):
            if d <= 0:
                return src
            blank = self.Expr(src[0], "0.0") * d
            return blank + src + blank

        pad = self.AddBorders(src, a + s, a + s, a + s, a + s)
        pad = duplicate(pad)

        if rclip is not None:
            rclip = self.AddBorders(rclip, a + s, a + s, a + s, a + s)
            rclip = duplicate(rclip)

        nlm = self.KNLMeansCL(pad, d=d, a=a, s=s, h=h, channels="YUV" if color else "Y", wref=1.0, rclip=rclip)
        clip = self.Crop(nlm, a + s, a + s, a + s, a + s)

        return clip[d:clip.num_frames - d]

    def ThrMerge(self, src: vs.VideoNode, ref1: BasicClip, ref2: BasicClip = None,
                 thr: float = 0.0009765625, elast: float = None) -> vs.VideoNode:
        ref2 = ref1 if ref2 is None else ref2
        elast = thr / 2 if elast is None else elast

        BExp = [
            "x {thr} {elast} + z - 2 {elast} * / * y {elast} z + {thr} - 2 {elast} * / * +".format(thr=thr, elast=elast)
        ]
        BDif = self.Expr(ref1, "0.0")

        PDif = self.Expr([src, ref1], "x y - 0.0 max")
        PRef = self.Expr([src, ref2], "x y - 0.0 max")

        PBLD = self.Expr([PDif, BDif, PRef], BExp)

        NDif = self.Expr([src, ref1], "y x - 0.0 max")
        NRef = self.Expr([src, ref2], "y x - 0.0 max")

        NBLD = self.Expr([NDif, BDif, NRef], BExp)
        BLDD = self.MakeDiff(PBLD, NBLD)
        BLD = self.MergeDiff(ref1, BLDD)

        UDN = self.Expr([src, ref2, BLD], ["x y - abs {thr} {elast} - > z x ?".format(thr=thr, elast=elast)])
        return self.Expr([src, ref2, UDN, ref1], ["x y - abs {thr} {elast} + < z a ?".format(thr=thr, elast=elast)])

    def GenBlockMask(self, src: vs.VideoNode) -> vs.VideoNode:
        clip = self.BlankClip(src, 24, 24, color=0.0)

        clip = self.AddBorders(clip, 4, 4, 4, 4, color=1.0)

        clip = self.StackHorizontal([clip, clip, clip, clip])
        clip = self.StackVertical([clip, clip, clip, clip])

        clip = self.Resample(clip, 32, 32, kernel="point", **fmtc_args)

        clip = self.Expr(clip, ["x 0.0 > 1.0 0.0 ?"])

        clip = self.StackHorizontal([clip, clip, clip, clip, clip, clip, clip, clip])
        clip = self.StackVertical([clip, clip, clip, clip, clip, clip])
        clip = self.StackHorizontal([clip, clip, clip, clip, clip, clip])
        clip = self.StackVertical([clip, clip, clip, clip, clip])
        clip = self.StackHorizontal([clip, clip, clip, clip, clip, clip])
        clip = self.StackVertical([clip, clip, clip, clip, clip])

        clip = self.CropAbs(clip, src.width, src.height, 0, 0)

        return clip


class internal:
    @staticmethod
    def super(core: Core, src: vs.VideoNode, pel: Literal[2, 4]):
        src = core.Pad(src, 128, 128, 128, 128)
        clip = core.Transpose(core.NNEDI(core.Transpose(core.NNEDI(src, **nnedi_args)), **nnedi_args))
        if pel == 4:
            clip = core.Transpose(core.NNEDI(core.Transpose(core.NNEDI(clip, **nnedi_args)), **nnedi_args))
        return clip

    @staticmethod
    def basic(
        core: Core, src: vs.VideoNode, super: SuperClip, radius: int,
        pel: Literal[2, 4], sad: int, short_time: bool, color: bool
    ):
        plane = 4 if color else 0
        src = core.Pad(src, 128, 128, 128, 128)
        supersoft = core.MSuper(src, pelclip=super, rfilter=4, pel=pel, chroma=color, **msuper_args)
        supersharp = core.MSuper(src, pelclip=super, rfilter=2, pel=pel, chroma=color, **msuper_args)

        if short_time:
            constant = 0.0001989762736579584832432989326
            me_sad = [constant * math.pow(sad, 2.0) * math.log(1.0 + 1.0 / (constant * sad))]
            me_sad += [sad]
            mvbw_vmulti = core.MAnalyze(supersoft, isb=True, chroma=color, overlap=4, blksize=8, **manalyze_args)
            mvbw_vmulti = core.MRecalculate(
                supersoft,
                mvbw_vmulti,
                chroma=color,
                overlap=2,
                blksize=4,
                thsad=me_sad[0],
                **mrecalculate_args)
            mvbw_vmulti = core.MRecalculate(
                supersoft,
                mvbw_vmulti,
                chroma=color,
                overlap=1,
                blksize=2,
                thsad=me_sad[1],
                **mrecalculate_args)

            mvfw_vmulti = core.MAnalyze(supersoft, chroma=color, overlap=4, blksize=8, **manalyze_args)
            mvfw_vmulti = core.MRecalculate(
                supersoft,
                mvfw_vmulti,
                chroma=color,
                overlap=2,
                blksize=4,
                thsad=me_sad[0],
                **mrecalculate_args)
            mvfw_vmulti = core.MRecalculate(
                supersoft,
                mvfw_vmulti,
                chroma=color,
                overlap=1,
                blksize=2,
                thsad=me_sad[1],
                **mrecalculate_args)
        else:
            constant = 0.0000139144247313257680589719533
            me_sad = constant * math.pow(sad, 2.0) * math.log(1.0 + 1.0 / (constant * sad))
            mvbw_vmulti = core.MAnalyze(supersoft, isb=True, chroma=color, overlap=64, blksize=128, **manalyze_args)
            mvbw_vmulti = core.MRecalculate(
                supersoft,
                mvbw_vmulti,
                chroma=color,
                overlap=32,
                blksize=64,
                thsad=me_sad,
                **mrecalculate_args)
            mvbw_vmulti = core.MRecalculate(
                supersoft,
                mvbw_vmulti,
                chroma=color,
                overlap=16,
                blksize=32,
                thsad=me_sad,
                **mrecalculate_args)
            mvbw_vmulti = core.MRecalculate(
                supersoft,
                mvbw_vmulti,
                chroma=color,
                overlap=8,
                blksize=16,
                thsad=me_sad,
                **mrecalculate_args)
            mvbw_vmulti = core.MRecalculate(
                supersoft,
                mvbw_vmulti,
                chroma=color,
                overlap=4,
                blksize=8,
                thsad=me_sad,
                **mrecalculate_args)
            mvbw_vmulti = core.MRecalculate(
                supersoft,
                mvbw_vmulti,
                chroma=color,
                overlap=2,
                blksize=4,
                thsad=me_sad,
                **mrecalculate_args)

            mvfw_vmulti = core.MAnalyze(supersoft, chroma=color, overlap=64, blksize=128, **manalyze_args)
            mvfw_vmulti = core.MRecalculate(
                supersoft,
                mvfw_vmulti,
                chroma=color,
                overlap=32,
                blksize=64,
                thsad=me_sad,
                **mrecalculate_args)
            mvfw_vmulti = core.MRecalculate(
                supersoft,
                mvfw_vmulti,
                chroma=color,
                overlap=16,
                blksize=32,
                thsad=me_sad,
                **mrecalculate_args)
            mvfw_vmulti = core.MRecalculate(
                supersoft,
                mvfw_vmulti,
                chroma=color,
                overlap=8,
                blksize=16,
                thsad=me_sad,
                **mrecalculate_args)
            mvfw_vmulti = core.MRecalculate(
                supersoft,
                mvfw_vmulti,
                chroma=color,
                overlap=4,
                blksize=8,
                thsad=me_sad,
                **mrecalculate_args)
            mvfw_vmulti = core.MRecalculate(
                supersoft,
                mvfw_vmulti,
                chroma=color,
                overlap=2,
                blksize=4,
                thsad=me_sad,
                **mrecalculate_args)
        clip = core.MDegrain(src, supersharp, mvbw_vmulti, mvfw_vmulti, thsad=sad, plane=plane, **mdegrain_args)
        clip = core.Crop(clip, 128, 128, 128, 128)
        return clip

    @staticmethod
    def deringing(core: Core, src: vs.VideoNode, ref: BasicClip, radius: int, h: float, sigma: float,
                  mse: tuple[float, float],
                  hard_thr: float, block_size: int, block_step: int, group_size: int, bm_range: int, bm_step: int,
                  ps_num: int, ps_range: int, ps_step: int, lowpass: list[float],
                  color: bool, matrix: int):
        c1 = 0.1134141984932795312503328847998
        c2 = 2.8623043756241389436528021745239
        strength = [h]
        strength += [h * math.pow(c1 * h, c2) * math.log(1.0 + 1.0 / math.pow(c1 * h, c2))]
        strength += [None]

        def loop(flt: vs.VideoNode | None, init: vs.VideoNode, src: vs.VideoNode, n: int) -> vs.VideoNode:
            strength[2] = n * strength[0] / 4 + strength[1] * (1 - n / 4)
            window = int(32 / math.pow(2, n))
            flt = init if n == 4 else flt
            dif = core.MakeDiff(src, flt)
            dif = core.NLMeans(dif, 0, window, 1, strength[2], flt, color)
            fnl = core.MergeDiff(flt, dif)
            n -= 1
            return fnl if n == -1 else loop(fnl, init, src, n)
        ref = core.FreqMerge(src, ref, block_size // 2 * 2 + 1, lowpass)
        dif = core.MakeDiff(src, ref)

        dif = core.BMBasic(
            dif, ref, radius=radius, th_mse=mse[0], hard_thr=hard_thr, sigma=sigma,
            block_size=block_size, block_step=block_step, group_size=group_size, bm_range=bm_range,
            bm_step=bm_step, ps_num=ps_num, ps_range=ps_range, ps_step=ps_step, matrix=matrix
        )

        dif = core.Aggregate(dif, radius, 1)
        ref = core.MergeDiff(ref, dif)
        refined = loop(None, ref, src, 4)

        bm3d = core.BMFinal(
            refined, ref, radius=radius, th_mse=mse[1], sigma=sigma,
            block_size=block_size, block_step=block_step,
            group_size=group_size, bm_range=bm_range, bm_step=bm_step,
            ps_num=ps_num, ps_range=ps_range, ps_step=ps_step, matrix=matrix
        )

        bm3d = core.Aggregate(bm3d, radius, 1)
        bm3d = core.FreqMerge(refined, bm3d, block_size // 2 * 2 + 1, lowpass)
        clip = loop(None, bm3d, refined, 4)

        return clip

    @staticmethod
    def destaircase(
        core: Core, src: vs.VideoNode, ref: BasicClip, radius: int, sigma: float, mse: tuple[float, float],
        hard_thr: float, block_size: int, block_step: int, group_size: int, bm_range: int, bm_step: int,
        ps_num: int, ps_range: int, ps_step: int, thr: float, elast: float, lowpass: list[float],
        matrix: int
    ):
        mask = core.GenBlockMask(core.ShufflePlanes(src, 0, vs.GRAY))
        ref = core.FreqMerge(src, ref, block_size // 2 * 2 + 1, lowpass)
        ref = core.ThrMerge(src, ref, thr=thr, elast=elast)
        dif = core.MakeDiff(src, ref)
        dif = core.BMBasic(
            dif, ref, radius=radius, th_mse=mse[0], hard_thr=hard_thr, sigma=sigma,
            block_size=block_size, block_step=block_step, group_size=group_size,
            bm_range=bm_range, bm_step=bm_step, ps_num=ps_num, ps_range=ps_range,
            ps_step=ps_step, matrix=matrix
        )

        dif = core.Aggregate(dif, radius, 1)
        ref = core.MergeDiff(ref, dif)
        dif = core.MakeDiff(src, ref)
        dif = core.BMFinal(
            dif, ref, radius=radius, th_mse=mse[1], sigma=sigma, block_size=block_size,
            block_step=block_step, group_size=group_size, bm_range=bm_range, bm_step=bm_step,
            ps_num=ps_num, ps_range=ps_range, ps_step=ps_step, matrix=matrix
        )

        dif = core.Aggregate(dif, radius, 1)
        ref = core.MergeDiff(ref, dif)
        return core.MaskedMerge(src, ref, mask, first_plane=True)

    @staticmethod
    def deblocking(core: Core, src: vs.VideoNode, ref: BasicClip, radius: int, h: float, sigma: float,
                   mse: tuple[float, float],
                   hard_thr: float, block_size: int, block_step: int, group_size: int, bm_range: int, bm_step: int,
                   ps_num: int, ps_range: int, ps_step: int, lowpass: list[float],
                   color: bool, matrix: int):
        mask = core.GenBlockMask(core.ShufflePlanes(src, 0, vs.GRAY))
        cleansed = core.NLMeans(ref, radius, block_size, math.ceil(block_size / 2), h, ref, color)
        dif = core.MakeDiff(ref, cleansed)
        dif = core.BMBasic(
            dif, cleansed, radius=radius, th_mse=mse[0], hard_thr=hard_thr, sigma=sigma,
            block_size=block_size, block_step=block_step, group_size=group_size, bm_range=bm_range,
            bm_step=bm_step, ps_num=ps_num, ps_range=ps_range, ps_step=ps_step, matrix=matrix
        )
        dif = core.Aggregate(dif, radius, 1)
        cleansed = core.MergeDiff(cleansed, dif)
        dif = core.MakeDiff(ref, cleansed)
        dif = core.BMFinal(
            dif, cleansed, radius=radius, th_mse=mse[1], sigma=sigma,
            block_size=block_size, block_step=block_step, group_size=group_size,
            bm_range=bm_range, bm_step=bm_step, ps_num=ps_num, ps_range=ps_range,
            ps_step=ps_step, matrix=matrix
        )

        dif = core.Aggregate(dif, radius, 1)
        cleansed = core.MergeDiff(cleansed, dif)
        ref = core.FreqMerge(cleansed, ref, block_size // 2 * 2 + 1, lowpass)
        src = core.FreqMerge(cleansed, src, block_size // 2 * 2 + 1, lowpass)
        return core.MaskedMerge(src, ref, mask, first_plane=True)


def Super(src: vs.VideoNode, pel=4):
    if not isinstance(src, vs.VideoNode):
        raise TypeError("Oyster.Super: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
        raise TypeError("Oyster.Super: the sample type of src has to be single precision!")
    elif src.format.subsampling_w > 0 or src.format.subsampling_h > 0:
        raise RuntimeError("Oyster.Super: subsampled stuff not supported!")
    if not isinstance(pel, int):
        raise TypeError("Oyster.Super: pel has to be an integer!")
    elif pel not in [2, 4]:
        raise RuntimeError("Oyster.Super: pel has to be 2 or 4!")

    core = Core()
    src = core.SetFieldBased(src, 0)
    colorspace = src.format.color_family
    if colorspace == vs.RGB:
        src = core.RGB2OPP(src, 1)
    clip = internal.super(core, src, pel)
    del core
    return clip


def Basic(src, super=None, radius=6, pel=4, sad=2000.0, short_time=False):
    if not isinstance(src, vs.VideoNode):
        raise TypeError("Oyster.Basic: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
        raise TypeError("Oyster.Basic: the sample type of src has to be single precision!")
    elif src.format.subsampling_w > 0 or src.format.subsampling_h > 0:
        raise RuntimeError("Oyster.Basic: subsampled stuff not supported!")
    if not isinstance(super, vs.VideoNode) and super is not None:
        raise TypeError("Oyster.Basic: super has to be a video clip or None!")
    elif super is not None:
        if any([
            super.format.sample_type != vs.FLOAT,
            super.format.bits_per_sample < 32,
            super.format.subsampling_w > 0,
            super.format.subsampling_h > 0
        ]):
            raise RuntimeError("Oyster.Basic: corrupted super clip!")
    if not isinstance(radius, int):
        raise TypeError("Oyster.Basic: radius has to be an integer!")
    elif radius < 1:
        raise RuntimeError("Oyster.Basic: radius has to be greater than 0!")
    if not isinstance(pel, int):
        raise TypeError("Oyster.Basic: pel has to be an integer!")
    elif pel not in [1, 2, 4]:
        raise RuntimeError("Oyster.Basic: pel has to be 1, 2 or 4!")
    if not isinstance(sad, float) and not isinstance(sad, int):
        raise TypeError("Oyster.Basic: sad has to be a real number!")
    elif sad <= 0.0:
        raise RuntimeError("Oyster.Basic: sad has to be greater than 0!")
    if not isinstance(short_time, bool):
        raise TypeError("Oyster.Basic: short_time has to be boolean!")
    core = Core()
    color = True
    rgb = False
    colorspace = src.format.color_family
    if colorspace == vs.RGB:
        src = core.RGB2OPP(src, 1)
        rgb = True
    if colorspace == vs.GRAY:
        color = False
    src = core.SetFieldBased(src, 0)
    super = core.SetFieldBased(super, 0) if super is not None else None
    clip = internal.basic(core, src, super, radius, pel, sad, short_time, color)
    clip = core.OPP2RGB(clip, 1) if rgb else clip
    del core
    return clip


def Deringing(
    src: vs.VideoNode, ref: BasicClip, radius: int = 6, h: float = 6.4, sigma: float = 16.0,
    mse: tuple[float, float] = [None, None], hard_thr: float = 3.2, block_size: int = 8,
    block_step: int = 1, group_size: int = 32, bm_range: int = 24, bm_step: int = 1,
    ps_num: int = 2, ps_range: int = 8, ps_step: int = 1, lowpass: list[float] = None
):
    if not isinstance(src, vs.VideoNode):
        raise TypeError("Oyster.Deringing: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
        raise TypeError("Oyster.Deringing: the sample type of src has to be single precision!")
    elif src.format.subsampling_w > 0 or src.format.subsampling_h > 0:
        raise RuntimeError("Oyster.Deringing: subsampled stuff not supported!")
    if not isinstance(ref, vs.VideoNode):
        raise TypeError("Oyster.Deringing: ref has to be a video clip!")
    elif ref.format.sample_type != vs.FLOAT or ref.format.bits_per_sample < 32:
        raise TypeError("Oyster.Deringing: the sample type of ref has to be single precision!")
    elif ref.format.subsampling_w > 0 or ref.format.subsampling_h > 0:
        raise RuntimeError("Oyster.Deringing: subsampled stuff not supported!")
    if not isinstance(radius, int):
        raise TypeError("Oyster.Deringing: radius has to be an integer!")
    elif radius < 1:
        raise RuntimeError("Oyster.Deringing: radius has to be greater than 0!")
    if not isinstance(h, float) and not isinstance(h, int):
        raise TypeError("Oyster.Deringing: h has to be a real number!")
    elif h <= 0:
        raise RuntimeError("Oyster.Deringing: h has to be greater than 0!")
    if not isinstance(mse, list):
        raise TypeError("Oyster.Deringing: mse parameter has to be an array!")
    elif len(mse) != 2:
        raise RuntimeError("Oyster.Deringing: mse parameter has to contain 2 elements exactly!")
    for i in range(2):
        if not isinstance(mse[i], float) and not isinstance(mse[i], int) and mse[i] is not None:
            raise TypeError("Oyster.Deringing: elements in mse must be real numbers or None!")
    if not isinstance(lowpass, list) and lowpass is not None:
        raise TypeError("Oyster.Deringing: lowpass has to be a list or None!")
    core = Core()
    rgb = False
    color = True
    mse[0] = sigma * 160.0 + 1200.0 if mse[0] is None else mse[0]
    mse[1] = sigma * 120.0 + 800.0 if mse[1] is None else mse[1]
    lowpass = [0.0, sigma, 0.48, 1024.0, 1.0, 1024.0] if lowpass is None else lowpass
    matrix = None
    colorspace = src.format.color_family
    if colorspace == vs.RGB:
        rgb = True
        matrix = 100
        src = core.RGB2OPP(src, 1)
        ref = core.RGB2OPP(ref, 1)
    if colorspace == vs.GRAY:
        color = False
    src = core.SetFieldBased(src, 0)
    ref = core.SetFieldBased(ref, 0)
    clip = internal.deringing(
        core,
        src,
        ref,
        radius,
        h,
        sigma,
        mse,
        hard_thr,
        block_size,
        block_step,
        group_size,
        bm_range,
        bm_step,
        ps_num,
        ps_range,
        ps_step,
        lowpass,
        color,
        matrix)
    clip = core.OPP2RGB(clip, 1) if rgb else clip
    core.delete()
    return clip


def Destaircase(
    src: vs.VideoNode, ref: BasicClip, radius: int = 6, sigma: float = 16.0,
    mse: tuple[float, float] = [None, None], hard_thr: float = 3.2, block_size: int = 8,
    block_step: int = 1, group_size: int = 32, bm_range: int = 24, bm_step: int = 1,
    ps_num: int = 2, ps_range: int = 8, ps_step: int = 1,
    thr: float = 0.03125, elast: float = 0.015625, lowpass: list[float] = None
):
    if not isinstance(src, vs.VideoNode):
        raise TypeError("Oyster.Destaircase: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
        raise TypeError("Oyster.Destaircase: the sample type of src has to be single precision!")
    elif src.format.subsampling_w > 0 or src.format.subsampling_h > 0:
        raise RuntimeError("Oyster.Destaircase: subsampled stuff not supported!")
    if not isinstance(ref, vs.VideoNode):
        raise TypeError("Oyster.Destaircase: ref has to be a video clip!")
    elif ref.format.sample_type != vs.FLOAT or ref.format.bits_per_sample < 32:
        raise TypeError("Oyster.Destaircase: the sample type of ref has to be single precision!")
    elif ref.format.subsampling_w > 0 or ref.format.subsampling_h > 0:
        raise RuntimeError("Oyster.Destaircase: subsampled stuff not supported!")
    if not isinstance(radius, int):
        raise TypeError("Oyster.Destaircase: radius has to be an integer!")
    elif radius < 1:
        raise RuntimeError("Oyster.Destaircase: radius has to be greater than 0!")
    if not isinstance(mse, list):
        raise TypeError("Oyster.Destaircase: mse parameter has to be an array!")
    elif len(mse) != 2:
        raise RuntimeError("Oyster.Destaircase: mse parameter has to contain 2 elements exactly!")
    for i in range(2):
        if not isinstance(mse[i], float) and not isinstance(mse[i], int) and mse[i] is not None:
            raise TypeError("Oyster.Destaircase: elements in mse must be real numbers or None!")
    if not isinstance(thr, float) and not isinstance(thr, int):
        raise TypeError("Oyster.Destaircase: thr has to be a real number!")
    elif thr < 0 or thr > 1:
        raise RuntimeError("Oyster.Destaircase: thr has to fall in [0, 1]!")
    if not isinstance(elast, float) and not isinstance(elast, int):
        raise TypeError("Oyster.Destaircase: elast has to be a real number!")
    elif elast < 0 or elast > thr:
        raise RuntimeError("Oyster.Destaircase: elast has to fall in [0, thr]!")
    if not isinstance(lowpass, list) and lowpass is not None:
        raise TypeError("Oyster.Destaircase: lowpass has to be a list or None!")
    core = Core()
    rgb = False
    mse[0] = sigma * 160.0 + 1200.0 if mse[0] is None else mse[0]
    mse[1] = sigma * 120.0 + 800.0 if mse[1] is None else mse[1]
    lowpass = [0.0, sigma, 0.48, 1024.0, 1.0, 1024.0] if lowpass is None else lowpass
    matrix = None
    colorspace = src.format.color_family
    if colorspace == vs.RGB:
        rgb = True
        matrix = 100
        src = core.RGB2OPP(src, 1)
        ref = core.RGB2OPP(ref, 1)
    src = core.SetFieldBased(src, 0)
    ref = core.SetFieldBased(ref, 0)
    clip = internal.destaircase(
        core,
        src,
        ref,
        radius,
        sigma,
        mse,
        hard_thr,
        block_size,
        block_step,
        group_size,
        bm_range,
        bm_step,
        ps_num,
        ps_range,
        ps_step,
        thr,
        elast,
        lowpass,
        matrix)
    clip = core.OPP2RGB(clip, 1) if rgb else clip
    del core
    return clip


def Deblocking(
    src: vs.VideoNode, ref: BasicClip, radius: int = 6, h: float = 6.4, sigma: float = 16.0,
    mse: tuple[float, float] = [None, None], hard_thr: float = 3.2, block_size: int = 8,
    block_step: int = 1, group_size: int = 32, bm_range: int = 24, bm_step: int = 1,
    ps_num: int = 2, ps_range: int = 8, ps_step: int = 1,
    lowpass: list[float] = [0.0, 0.0, 0.12, 1024.0, 1.0, 1024.0]
):
    if not isinstance(src, vs.VideoNode):
        raise TypeError("Oyster.Deblocking: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
        raise TypeError("Oyster.Deblocking: the sample type of src has to be single precision!")
    elif src.format.subsampling_w > 0 or src.format.subsampling_h > 0:
        raise RuntimeError("Oyster.Deblocking: subsampled stuff not supported!")
    if not isinstance(ref, vs.VideoNode):
        raise TypeError("Oyster.Deblocking: ref has to be a video clip!")
    elif ref.format.sample_type != vs.FLOAT or ref.format.bits_per_sample < 32:
        raise TypeError("Oyster.Deblocking: the sample type of ref has to be single precision!")
    elif ref.format.subsampling_w > 0 or ref.format.subsampling_h > 0:
        raise RuntimeError("Oyster.Deblocking: subsampled stuff not supported!")
    if not isinstance(radius, int):
        raise TypeError("Oyster.Deblocking: radius has to be an integer!")
    elif radius < 1:
        raise RuntimeError("Oyster.Deblocking: radius has to be greater than 0!")
    if not isinstance(h, float) and not isinstance(h, int):
        raise TypeError("Oyster.Deblocking: h has to be a real number!")
    elif h <= 0:
        raise RuntimeError("Oyster.Deblocking: h has to be greater than 0!")
    if not isinstance(mse, list):
        raise TypeError("Oyster.Deblocking: mse parameter has to be an array!")
    elif len(mse) != 2:
        raise RuntimeError("Oyster.Deblocking: mse parameter has to contain 2 elements exactly!")
    for i in range(2):
        if not isinstance(mse[i], float) and not isinstance(mse[i], int) and mse[i] is not None:
            raise TypeError("Oyster.Deblocking: elements in mse must be real numbers or None!")
    if not isinstance(lowpass, list):
        raise TypeError("Oyster.Deblocking: lowpass has to be a list!")
    core = Core()
    rgb = False
    color = True
    mse[0] = sigma * 160.0 + 1200.0 if mse[0] is None else mse[0]
    mse[1] = sigma * 120.0 + 800.0 if mse[1] is None else mse[1]
    matrix = None
    colorspace = src.format.color_family
    if colorspace == vs.RGB:
        rgb = True
        matrix = 100
        src = core.RGB2OPP(src, 1)
        ref = core.RGB2OPP(ref, 1)
    if colorspace == vs.GRAY:
        color = False
    src = core.SetFieldBased(src, 0)
    ref = core.SetFieldBased(ref, 0)
    clip = internal.deblocking(
        core,
        src,
        ref,
        radius,
        h,
        sigma,
        mse,
        hard_thr,
        block_size,
        block_step,
        group_size,
        bm_range,
        bm_step,
        ps_num,
        ps_range,
        ps_step,
        lowpass,
        color,
        matrix)
    clip = core.OPP2RGB(clip, 1) if rgb else clip
    del core
    return clip
