from __future__ import annotations

import ast
import inspect
import re
from typing import Iterator

from vstools import T

__all__ = [
    'remove_chars', 'destructure'
]


def remove_chars(string: str, chars: str = '') -> str:
    return string.translate({ord(char): None for char in chars})


def destructure(dict_: dict[str, T], ignore_no_key: bool = False) -> T | Iterator[T]:
    if not isinstance(dict_, object):
        raise TypeError(f"destructure: {dict_} is not an object!")

    parent_frame = inspect.currentframe()

    assert parent_frame and parent_frame.f_back

    previous_frame = parent_frame.f_back

    frame_info = inspect.getframeinfo(previous_frame)

    assert frame_info.code_context

    re_flags = re.I + re.M + re.U

    with open(previous_frame.f_code.co_filename, 'r') as f:
        source = f.read()

    end_index = previous_frame.f_lineno

    ast_root = ast.parse(source, previous_frame.f_code.co_filename)

    caller_node: ast.AST | None = None

    def _asttreewalk(parent: ast.AST, lineno: int) -> None:
        nonlocal caller_node
        assert parent_frame

        for child in ast.iter_child_nodes(parent):
            if hasattr(child, 'lineno'):
                lineno = child.lineno
            if lineno > end_index:
                break

            if isinstance(child, ast.Name):
                if (isinstance(parent, ast.Call) and child.id == parent_frame.f_code.co_name):
                    caller_node = parent
                    break

            _asttreewalk(child, lineno)

    _asttreewalk(ast_root, 0)

    assert caller_node, RuntimeError('destructure: Code not properly formatted!')

    start_index = caller_node.lineno - 1

    source_split = source.splitlines()

    source_lines = source_split[start_index:end_index]

    curr_line = ' '.join(source_lines)

    nospaces = re.sub(r"\s+", r'', curr_line, 0, re_flags)

    if (cr_Tk := ')=' in nospaces) or (br_Tk := ']=' in nospaces):
        tmp_idx = start_index
        while ('(' if cr_Tk else '[' if br_Tk else '{') != curr_line[0] or curr_line[0] not in {'(', '['}:
            tmp_idx -= 1
            curr_line = '\n'.join([source_split[tmp_idx], curr_line])

    curr_line = re.sub(r"\n\n+", r'\n', curr_line, 0, re_flags)
    curr_line = re.sub(
        r"([\[\(])[\n\s]*(.*)[\n\s]*([\]\)]).*(=)", r'\1\2\3\4', curr_line, 0, re_flags
    )

    curr_line = re.sub(r"\s+", r'', curr_line, 0, re_flags)

    (lvalues, *_) = curr_line.strip().partition('=')

    lvalues = remove_chars(lvalues, ')(][ }{')

    keys = [ss for s in lvalues.split(',') if (ss := s.strip())]

    def _generator() -> Iterator[T]:
        nonlocal keys, dict_

        for key in keys:
            value: T

            try:
                value = dict_.__getattribute__(key)
            except BaseException:
                try:
                    value = dict_.__dict__.__getitem__(key)
                except BaseException:
                    try:
                        value = dict_.__class__.__dict__.__getitem__(key)
                    except BaseException:
                        try:
                            value = dict_[key]
                        except BaseException:
                            if not ignore_no_key:
                                raise KeyError(key)

            yield value

    gen_result = _generator()

    if len(keys) == 1:
        return next(gen_result)
    else:
        return gen_result
