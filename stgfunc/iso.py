import json
import atexit
import shutil
import subprocess
from pathlib import Path
import vapoursynth as vs
from io import BufferedReader
from pyparsedvd import vts_ifo
from os import name as os_name
from lvsfunc.types import Range
from itertools import accumulate
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Union, Optional, Tuple, cast

core = vs.core

# Contains portion of code from
# https://github.com/Varde-s-Forks/lvsfunc/blob/patches/source/lvsfunc/source.py
# Will be replaced with vardefunc's import when it's going to be available in it


class DVDIndexer(ABC):
  """Abstract DVD indexer interface."""
  path: Union[Path, str]
  vps_indexer: Callable[..., vs.VideoNode]
  ext: str

  def __init__(self, path: Union[Path, str], vps_indexer: Callable[..., vs.VideoNode], ext: str) -> None:
    self.path = path
    self.vps_indexer = vps_indexer  # type: ignore
    self.ext = ext
    super().__init__()

  @abstractmethod
  def get_cmd(self, files: List[Path], output: Path) -> List[Any]:
    """Returns the indexer command"""
    raise NotImplementedError

  def _check_path(self) -> None:
    if not shutil.which(self.path):
      raise FileNotFoundError(f'DVDIndexer: `{self.path}` was not found!')

  def index(self, files: List[Path], output: Path) -> None:
    return subprocess.run(self.get_cmd(files, output), check=True, text=True, encoding='utf-8', stdout=subprocess.PIPE)

  def get_idx_file_path(self, path: Path, index: Optional[int] = None) -> Path:
    return path.with_suffix(self.ext)

  @abstractmethod
  def update_idx_file(self, index_path: Path, filepaths: List[Path]):
    raise NotImplementedError


class D2VWitch(DVDIndexer):
  """Built-in d2vwitch indexer"""

  def __init__(
      self, path: Union[Path, str] = 'd2vwitch',
      vps_indexer: Callable[..., vs.VideoNode] = core.d2v.Source, ext: str = '.d2v'
  ) -> None:
    super().__init__(path, vps_indexer, ext)

  def get_cmd(self, files: List[Path], output: Path) -> List[Any]:
    self._check_path()
    return [self.path, *files, '--output', output]

  def update_idx_file(self, index_path: Path, filepaths: List[Path]):
    with open(index_path, 'r') as file:
      content = file.read()

    str_filepaths = [str(path) for path in filepaths]

    firstsplit_idx = content.index('\n\n')

    if "DGIndex" not in content[:firstsplit_idx]:
      raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

    maxsplits = content[:firstsplit_idx].count('\n') + 1

    content = content.split('\n', maxsplits)

    n_files = int(content[1])

    if not n_files or n_files != len(str_filepaths):
      raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

    if content[2:maxsplits] == str_filepaths:
      return

    content[2:maxsplits] = str_filepaths

    content = '\n'.join(content)

    with open(index_path, 'w') as file:
      file.write(content)


class DGIndexNV(DVDIndexer):
  """Built-in DGIndexNV indexer"""

  def __init__(
      self, path: Union[Path, str] = 'DGIndexNV',
      vps_indexer: Callable[..., vs.VideoNode] = core.dgdecodenv.DGSource, ext: str = '.dgi'
  ) -> None:
    super().__init__(path, vps_indexer, ext)

  def get_cmd(self, files: List[Path], output: Path) -> List[Any]:
    self._check_path()
    return [self.path, '-i', ','.join(map(str, files)), '-o', output, '-h']

  def update_idx_file(self, index_path: Path, filepaths: List[Path]):
    with open(index_path, 'r') as file:
      content = file.read()

    str_filepaths = [str(path) for path in filepaths]

    firstsplit_idx = content.index('\n\n')

    if "DGIndexNV" not in content[:firstsplit_idx]:
      raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

    cut_content = content[firstsplit_idx + 2:]

    firstsplit = content[:firstsplit_idx].count('\n') + 2

    maxsplits = cut_content[:cut_content.index('\n\n')].count('\n') + firstsplit + 1

    content = content.split('\n', maxsplits)

    if maxsplits - firstsplit != len(str_filepaths):
      raise RuntimeError("IsoFile: Index file corrupted! Delete it and retry.")

    splitted = [content[i].split(' ') for i in range(firstsplit, maxsplits)]

    if [split[0] for split in splitted] == str_filepaths:
      return

    content[firstsplit:maxsplits] = [
        f"{filepaths[i]} {splitted[i][1]}" for i in range(maxsplits - firstsplit)
    ]

    content = '\n'.join(content)

    with open(index_path, 'w') as file:
      file.write(content)


class __IsoFile:
  path: Path
  indexer: DVDIndexer
  dv2_path: Path
  mount_path: Path
  clip: vs.VideoNode = None
  cut_clips: List[vs.VideoNode] = None
  chapters_frames: List[List[int]] = None

  def __init__(self, path: Path, indexer: DVDIndexer = D2VWitch()):
    self.path = Path(path)
    self.indexer = indexer

  def source(self):
    if not self.path.is_file():
      raise ValueError("IsoFile: File doesn't exist!")

    self.mount_path = self._get_mount_path()

    vob_files = [
        tfile for tfile in sorted(self.mount_path.glob('*.vob'))
        if tfile.name.upper() != 'VIDEO_TS.VOB'
    ]

    if not len(vob_files):
      raise FileNotFoundError('IsoFile: No VOBs found!')

    self.dv2_path = self.indexer.get_idx_file_path(self.path)

    if not self.dv2_path.is_file():
      self.indexer.index(vob_files, self.dv2_path)
    else:
      self.indexer.update_idx_file(self.dv2_path, vob_files)

    self.clip = self.indexer.vps_indexer(self.dv2_path)

    return self.clip

  def split_titles(self) -> Tuple[List[vs.VideoNode], List[List[int]]]:
    # Parse IFO info

    ifo_files = [
        tfile for tfile in sorted(self.mount_path.glob('*.ifo'))
        if tfile.name.upper() != 'VIDEO_TS.IFO'
    ]

    program_chains = []

    m_ifos = len(ifo_files) > 1

    for ifo_file in ifo_files:
      with open(ifo_file, 'rb') as file:
        curr_pgci = vts_ifo.load_vts_pgci(cast(BufferedReader, file))

      program_chains += curr_pgci.program_chains[int(m_ifos):]

    chapters_frames: List[List[int]] = []

    for prog in program_chains:
      dvd_fps_s = [pb_time.fps for pb_time in prog.playback_times]
      if all(dvd_fps_s[0] == dvd_fps for dvd_fps in dvd_fps_s):
        fps = vts_ifo.FRAMERATE[dvd_fps_s[0]]
      else:
        raise ValueError('IsoFile: No VFR allowed! (Yet)')

      raw_fps = 30 if fps.numerator == 30000 else 25

      # Convert in frames
      chapters_frames.append([0] + [
          pb_time.frames + (pb_time.hours * 3600 + pb_time.minutes * 60 + pb_time.seconds) * raw_fps
          for pb_time in prog.playback_times
      ])

    chapters_frames = [
        list(accumulate(chapter_frames))
        for chapter_frames in chapters_frames
    ]

    durations = list(accumulate([0] + [frame[-1] for frame in chapters_frames]))

    # Remove splash screen and DVD Menu
    clip = self.clip[-durations[-1]:]

    # Trim per title
    clips = [clip[s:e] for s, e in zip(durations[:-1], durations[1:])]

    clips.append(self.clip[:-durations[-1]])

    self.cut_clips = clips
    self.chapters_frames = chapters_frames

    return clips, chapters_frames

  def get_title(self, clip_index: int, chapters: Union[Range, List[Range]]) -> vs.VideoNode:
    if not self.clip:
      self.source()

    if not self.cut_clips or not self.chapters_frames:
      self.split_titles()

    ranges = self.chapters_frames[clip_index]
    clip = self.cut_clips[clip_index]

    rlength = len(ranges)

    if isinstance(chapters, int):
      start, end = ranges[0], ranges[-1]

      if chapters == rlength:
        start = ranges[-2]
      elif chapters == 0:
        end = ranges[1]
      elif chapters < 0:
          end = rlength - 1 + end
          start = end - 1
      else:
        start = ranges[chapters]
        end = ranges[chapters + 1]

      return clip[start:end]
    elif isinstance(chapters, tuple):
      start, end = chapters

      if start is None:
        start = 0
      if end is None:
        end = rlength - 1
    elif isinstance(chapters, list):
      return [self.get_title(clip_index, rchap) for rchap in chapters]
    elif not isinstance(chapters, tuple):
      raise ValueError("IsoFile: chapters must either be an int or tuple of Optional[int] (Range).")

    if start < 0:
      start = rlength - 1 + start
    if end < 0:
      end = rlength - 1 + end

    return clip[ranges[start]:ranges[end]]

  @abstractmethod
  def _get_mount_path(self) -> Path:
    raise NotImplementedError()


class __WinIsoFile(__IsoFile):
  class_mount: bool = False

  def _get_mount_path(self) -> Path:
    disc = self.__get_mounted_disc()

    if not disc:
      self.class_mount = True
      disc = self.__mount()

    if not disc:
      raise RuntimeError("IsoFile: Could not mount ISO file!")

    if self.class_mount:
      atexit.register(self.__unmount)

    return Path(fr"{disc['DriveLetter']}:\VIDEO_TS")

  def __run_disc_util(self, iso_path: str, util: str) -> dict:
    process = subprocess.Popen(["PowerShell", fr'{util}-DiskImage -ImagePath "{iso_path}" | Get-Volume | ConvertTo-Json'], stdout=subprocess.PIPE)

    bjson, err = process.communicate()

    if err or bjson == b'' or str(bjson[:len(util)], 'utf8') == util:
      return None

    bjson = json.loads(str(bjson, 'utf-8'))

    del bjson['CimClass'], bjson['CimInstanceProperties'], bjson['CimSystemProperties']

    return bjson

  def __get_mounted_disc(self):
    return self.__run_disc_util(self.path, 'Get')

  def __mount(self):
    return self.__run_disc_util(self.path, 'Mount')

  def __unmount(self):
    return self.__run_disc_util(self.path, 'Dismount')


class __LinuxIsoFile(__IsoFile):
  def __init__(self, path):
    raise NotImplementedError(
        "IsoFile: Linux filesystem not (yet) supported."
    )


IsoFile = __WinIsoFile if os_name == 'nt' else __LinuxIsoFile
