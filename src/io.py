from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Generator, Iterable, List, Union, Dict


def _iter_files(path: Union[str, Path], extensions: Iterable[str]):
    p = Path(path)
    if p.is_dir():
        for ext in extensions:
            yield from p.rglob(f"*{ext}")
    elif p.is_file():
        yield p
    else:
        raise FileNotFoundError(f"Path not found: {p}")


def discover_data_files(path: Union[str, Path]) -> List[Path]:
    """Return a list of supported data files under ``path``.

    Supported extensions: .json, .jsonl, .txt
    """
    exts = [".jsonl", ".json", ".txt"]
    files = []
    for f in _iter_files(path, exts):
        if f.suffix.lower() in exts:
            files.append(f)
    return sorted(files)


def iter_json_posts(files: Iterable[Union[str, Path]]) -> Generator[dict, None, None]:
    """Yield JSON objects from an iterable of files.

    This function expects an iterable of file paths (Path or str) pointing to
    .json or .jsonl files. To discover files under a directory, call
    ``discover_data_files(path)`` and pass the resulting list to this function.

    For convenience this function accepts a single Path/str as well and will
    internally call ``discover_data_files`` if a directory is provided.

    Warnings are emitted for malformed JSON lines but iteration continues.
    """
    # allow passing a single Path/str
    if isinstance(files, (str, Path)):
        p = Path(files)
        if p.is_dir():
            files_iter = discover_data_files(p)
        else:
            files_iter = [p]
    else:
        files_iter = [Path(f) for f in files]

    for f in files_iter:
        suffix = f.suffix.lower()
        if suffix == ".jsonl":
            with f.open("r", encoding="utf-8") as fh:
                for i, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception as e:
                        warnings.warn(f"Failed to parse JSON line {i} in {f}: {e}")
        elif suffix == ".json":
            try:
                with f.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception as e:
                warnings.warn(f"Failed to parse JSON file {f}: {e}")
                continue
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                yield data
        else:
            # skip other files
            continue


def load_json_posts(files: Iterable[Union[str, Path]]) -> List[dict]:
    """Load JSON posts into a list (materialises the iterator).

    ``files`` should be an iterable of file paths (Paths or strings). To pass a
    directory, call ``discover_data_files(dir_path)`` and pass the resulting
    list filtered to JSON/JSONL files.
    """
    return list(iter_json_posts(files))


def load_txt_lines(files: Iterable[Union[str, Path]]) -> List[str]:
    """Load lines from an iterable of text file paths.

    ``files`` should be an iterable of .txt file paths. To discover .txt files
    under a directory, call ``discover_data_files(dir_path)`` and filter for
    files with the .txt suffix.
    """
    lines: List[str] = []
    # allow single Path/str
    if isinstance(files, (str, Path)):
        p = Path(files)
        if p.is_dir():
            files_iter = [f for f in discover_data_files(p) if f.suffix.lower() == ".txt"]
        else:
            files_iter = [p]
    else:
        files_iter = [Path(f) for f in files]

    for f in files_iter:
        if not f.exists():
            continue
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if s:
                    lines.append(s)
    return lines

def load_jsonl_dataset(data_path: str) -> List[Dict]:
    """Load a JSON Lines file into a list of dictionaries."""
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rows.append(json.loads(line))
    return rows