"""
Utility wrappers for file I/O and progress display.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import json

from tqdm import tqdm
from utils.logger import CURRENT_LOG_BUF           # ← IMPORT GLOBALLY


# ---------- tqdm wrapper ----------------------------------------------------- #
def tqdm_std(*args, **kwargs):
    """
    Unified entry point so every pipeline shows its progress bar **inside**
    the Streamlit live-log pane.

    * If the caller did not pass `file=...` **and** the global buffer exists,
      we set `file=CURRENT_LOG_BUF`.

    * We also tweak bar_format / ascii so the bar prints a new line each
      update instead of re-writing one line (Streamlit can't interpret \r).
    """
    if "file" not in kwargs and CURRENT_LOG_BUF is not None:
        kwargs["file"] = CURRENT_LOG_BUF

    kwargs.setdefault(
        "bar_format",
        "{desc:<15} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
        "[{elapsed}<{remaining}]",
    )
    kwargs.setdefault("ascii", True)
    kwargs.setdefault("mininterval", 0)       # flush every tick
    return tqdm(*args, **kwargs)


# ---------- safe JSONL writer ------------------------------------------------ #
@contextmanager
def safe_jsonl_writer(path: Path, mode: str = "a", encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding=encoding) as fh:
        yield fh
