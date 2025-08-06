"""
alignment_data.py
=================

Handles both alignment tasks that rely on `agents.orpo_generator.generate_orpo_data`

    • single-turn  (mode == "single-align")
    • multi-turn   (mode == "multi-align")
"""
from __future__ import annotations
import json, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pathlib

import pandas as pd

from agents.orpo_generator import generate_orpo_data
from utils.io      import tqdm_std, safe_jsonl_writer
from utils.logger  import init_root
from config        import Config

init_root()
log = logging.getLogger(__name__)


def _worker(idx: int, ctx: str, conv_base, jf, rows: list[dict], n_variants=3):
    try:
        for k in range(n_variants):
            generated = generate_orpo_data({"context": ctx,
                                            "conversations": conv_base}, k)
            if generated:
                rows.append({"context": ctx, "conversation": generated})
                json.dump({"context": ctx, "conversation": generated},
                          jf, ensure_ascii=False)
                jf.write("\n")
    except Exception as e:
        log.exception("ORPO idx %s failed: %s", idx, e)


def _run_generic(cfg: Config, multi: bool):
    path = pathlib.Path(cfg.input_file)
    try:
        df = pd.read_json(path, lines=True)
    except ValueError as e:
        raise ValueError(
            f"{path.name} could not be parsed as JSON-Lines.\n"
            "Did you perhaps upload the CSV, or does the file start with "
            "a BOM / log line?\n"
            f"Original error from pandas: {e}"
        ) from None
    data = pd.read_json(cfg.input_file, lines=True)
    tag = "multi_align" if multi else "single_align"
    out_csv  = cfg.output_dir / f"{cfg.model_name.replace('/','-')}_{tag}.csv"
    out_json = cfg.output_dir / f"{cfg.model_name.replace('/','-')}_{tag}.jsonl"
    rows: list[dict] = []

    with safe_jsonl_writer(out_json) as jf, \
         ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:

        futures = {}
        for i in range(len(data)):
            ctx = data["context"][i]

            if multi:   # already multi-turn
                conv_base = data["conversation"][i]
            else:       # construct single-turn seed
                conv_base = [{"question": data["question"][i],
                              "answer":  data["answer"][i]}]

            futures[pool.submit(_worker, i, ctx, conv_base, jf, rows)] = i

        for _ in tqdm_std(as_completed(futures), total=len(futures),
                          desc=f"ORPO {'multi' if multi else 'single'}"):
            pass

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("✅  ORPO %s: %s rows → %s / %s",
             'multi' if multi else 'single', len(rows), out_csv, out_json)
    return out_csv, out_json


# public wrappers --------------------------------------------------------------
def run_single(cfg: Config):      # cfg.mode == "single-align"
    return _run_generic(cfg, multi=False)

def run_multi(cfg: Config):       # cfg.mode == "multi-align"
    return _run_generic(cfg, multi=True)
