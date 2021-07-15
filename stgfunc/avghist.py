# from https://forum.doom9.org/showthread.php?t=168521
# averagehist.py : average histogram for vapoursynth
# author : ganymede
# requirement : histogram

import vapoursynth as vs
core = vs.core


class AverageHist():
  """Average histogram for vapoursynth."""

  def __init__(self):
    """core : vapoursynth's core instance."""

  def get_hist(self, clip, mode):
    """Returns a cropped histogram."""
    if mode == 'Levels':
      clip = core.hist.Levels(clip)
      clip = core.std.CropRel(clip, (clip.width - 256), 0, 0, (clip.height - 256))
    elif mode == 'Color':
      clip = core.hist.Color(clip)
      clip = core.std.CropRel(clip, (clip.width - 256), 0, 0, (clip.height - 256))
    elif mode == 'Color2':
      clip = core.hist.Color2(clip)
      clip = core.std.CropRel(clip, (clip.width - 256), 0, 0, (clip.height - 256))
    elif mode == 'Combined1':
      c1 = core.hist.Levels(clip)
      c1 = core.std.CropRel(c1, (c1.width - 256), 0, 0, (c1.height - 256))
      c2 = core.hist.Color2(clip)
      c2 = core.std.CropRel(c2, (c2.width - 256), 0, 0, (c2.height - 256))
      clip = core.std.StackVertical([c1, c2])
    elif mode == "Combined2":
      c1 = core.hist.Levels(clip)
      c1 = core.std.CropRel(c1, (c1.width - 256), 0, 0, (c1.height - 256))
      c2 = core.hist.Color2(clip)
      c2 = core.std.CropRel(c2, (c2.width - 256), 0, 0, (c2.height - 256))
      c3 = core.hist.Classic(clip)
      c3 = core.std.CropRel(c3, (c3.width - 256), 0, 0, 0)
      c4 = core.std.StackVertical([c1, c2])
      if c3.height < c4.height:
        c3 = core.std.AddBorders(c3, 0, 0, 0, (c4.height - c3.height), [0, 128, 128])
      elif c3.height > c4.height:
        c4 = core.std.AddBorders(c4, 0, 0, 0, (c3.height - c4.height), [0, 128, 128])
      clip = core.std.StackHorizontal([c3, c4])
    else:
      clip = core.hist.Classic(clip)
      clip = core.std.CropRel(clip, (clip.width - 256), 0, 0, 0)
    return clip

  def get_average(self, clip, mode='Classic'):
    """mode can be one of 'Classic', 'Levels', 'Color', 'Color2', 'Combined1' or 'Combined2'."""
    hist = self.get_hist(clip, mode)
    average = hist[0]
    for i in range(1, clip.num_frames):
      average = core.std.Merge(average, hist[i], (1.0 / (i + 1)))
    return average
