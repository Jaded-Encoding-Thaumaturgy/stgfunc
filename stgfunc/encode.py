import os
import sys
import string
import random
import signal
import shutil
import pathlib
import __main__
import subprocess
from os import path
from typing import List
import vapoursynth as vs

core = vs.core

stream = sys.stderr
progress_bar = None

s_columns = shutil.get_terminal_size().columns


class GetTokens():
  def __init__(self, clip: vs.VideoNode) -> None:
    self.clip = clip
    self.width = clip.width
    self.height = clip.height
    self.fps_num = clip.fps_num
    self.fps_den = clip.fps_den
    self.fps = clip.fps
    self.bitdepth = self.clip.format.bits_per_sample
    self.script = pathlib.PurePath(__main__.__file__)
    self.script_dir = self.script.parent
    self.script_name = self.script.stem
    self.frames_num = self.clip.num_frames
    self.subsampling = self.clip.format.name


def encode(clip: vs.VideoNode, output_file: str, x265: bool, cmd_args: List[str] = [], binary: str = None, **args) -> None:
  """Stolen from varde who stole from lyfunc
  Args:
      clip (vs.VideoNode): Source filtered clip
      output_file (str): Path to the output file.
        You can inject
          {w}: width, {h}: height,
            {fpsn}: fps numerator, {fpsd}: fps denominator,
          {fps}: fps as fraction, {f}: total frame count,
            {sd}: script directory, {sn}: script name,
          {bits}: bit depth, {ss}: subsampling
      binary (str): Path to x264/x265 binary.

  """
  from tqdm import tqdm
  from simple_chalk import chalk

  if binary is None:
    binary = "x265" if x265 else "x264"

  if x265:
    cmd = [binary, "--y4m", "--log-level", "-1", "--no-progress"]
  else:
    cmd = [binary, "--demuxer", "y4m"]

  for i, v in args.items():
    i = "--" + i if i[:2] != "--" else i
    i = i.replace("_", "-")
    if isinstance(v, bool):
      v = "true" if v else "false"
    if i in cmd:
      cmd[cmd.index(i) + 1] = str(v)
    else:
      cmd.extend([i, str(v)])

  cmd.extend(cmd_args)

  cmd.extend(["-o", output_name(clip, output_file, x265), "-"])

  progress_bar = None

  try:
    def progress_update(current, end, stream):
      global progress_bar

      if progress_bar is None or end != progress_bar.total:
        if progress_bar is not None:
          print(chalk.orange("ENCODING TO NEXT KEYFRAME").center(s_columns), file=process.stderr)

        progress_bar = tqdm(
            file=stream,
            desc="Encoding: ",
            total=end,
            dynamic_ncols=True,
            unit=" frames",
            ncols=0,
            ascii=sys.stdout.encoding != "utf8"
        )

      progress_bar.update(current - progress_bar.n)

    print(cmd)

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=None)

    print(chalk.green(f"STARTING ENCODE: {GetTokens(clip).script_name}").center(s_columns), file=process.stderr)

    clip.output(process.stdin, y4m=True, progress_update=lambda curr, end: progress_update(curr, end, process.stderr))

    process.communicate()
  except KeyboardInterrupt:
    print(chalk.red("STOPPED ENCODING").center(s_columns), file=process.stderr)
    return signal.SIGINT + 128

  except Exception as err:
    print("Unexpected exception:", err, file=stream)
    return 1

  else:
    if process.returncode != 0:
      print(chalk.red("Error"))
    return process.returncode


def output_name(clip: vs.VideoNode, name: str, x265: bool):
  t = GetTokens(clip)
  return name.format(
      w=t.width, h=t.height,
      fpsn=t.fps_num, fpsd=t.fps_den,
      fps=t.fps, f=t.frames_num,
      sd=t.script_dir, sn=t.script_name,
      bits=t.bitdepth, ss=t.subsampling
  ) + "_" + ''.join(random.choices(string.ascii_uppercase + string.digits, k=7)) + (".265" if x265 else ".264")


def create_qpfile(clip: vs.VideoNode, filename: vs.VideoNode, force: bool = False):
  import lvsfunc as lvf

  if force and path.isfile(filename):
    os.remove(filename)

  if not path.isfile(filename):
    with open(filename, 'w') as o:
      for f in lvf.render.find_scene_changes(clip, lvf.render.SceneChangeMode.WWXD_SCXVID_UNION):
        o.write(f"{f} I -1\n")
