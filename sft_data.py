"""
sft_data.py
===========

One module to generate either
    • single-turn SFT    (mode == "single-sft")
    • multi-turn  SFT    (mode == "multi-sft")

Public wrappers:
    run_single(cfg)   → for executor when cfg.mode == "single-sft"
    run_multi(cfg)    → for executor when cfg.mode == "multi-sft"
"""
from __future__ import annotations
import json, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ---- business-logic imports (your code) --------------------------------------
from agents.generator           import generate_qa
from agents.multiturngenerator  import generate_multi_turn_conversation
from agents.contextvalidator    import validate_context
from agents.qavalidator         import validate_qa
# ------------------------------------------------------------------------------
from utils.io      import tqdm_std, safe_jsonl_writer
from utils.logger  import init_root
from config        import Config

init_root()
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-TURN helpers
# ──────────────────────────────────────────────────────────────────────────────
def _single_worker(idx: int, chunk: str, jf, rows: list[dict]):
    try:
        if not validate_context(chunk):
            return
        for qa in generate_qa(chunk):
            q, a = qa["question"], qa["answer"]
            if validate_qa(q, a, chunk):
                rows.append({"question": q, "answer": a, "context": chunk})
                json.dump({"question": q, "answer": a, "context": chunk},
                          jf, ensure_ascii=False)
                jf.write("\n")
    except Exception as e:
        log.exception("single-turn chunk %s failed: %s", idx, e)


def _run_single(cfg: Config):
    data = pd.read_json(cfg.input_file, lines=True)
    out_csv  = cfg.output_dir / f"{cfg.model_name.replace('/','-')}_single_sft.csv"
    out_json = cfg.output_dir / f"{cfg.model_name.replace('/','-')}_single_sft.jsonl"
    rows: list[dict] = []

    with safe_jsonl_writer(out_json) as jf, \
         ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:

        futures = {pool.submit(_single_worker, i, txt, jf, rows): i
                   for i, txt in enumerate(data["text"])}
        for _ in tqdm_std(as_completed(futures), total=len(futures),
                          desc="Single-turn SFT"):
            pass

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("✅  single-turn: %s rows → %s / %s", len(rows), out_csv, out_json)
    return out_csv, out_json


# ──────────────────────────────────────────────────────────────────────────────
# MULTI-TURN helpers
# ──────────────────────────────────────────────────────────────────────────────
def _multi_worker(idx: int, chunk: str, jf, rows: list[dict]):
    try:
        conv = generate_multi_turn_conversation(chunk)
        if conv:
            rows.append({"context": chunk, "conversation": conv})
            json.dump({"context": chunk, "conversation": conv}, jf,
                      ensure_ascii=False)
            jf.write("\n")
    except Exception as e:
        log.exception("multi-turn chunk %s failed: %s", idx, e)


def _run_multi(cfg: Config):
    data = pd.read_json(cfg.input_file, lines=True)
    out_csv  = cfg.output_dir / f"{cfg.model_name.replace('/','-')}_multi_sft.csv"
    out_json = cfg.output_dir / f"{cfg.model_name.replace('/','-')}_multi_sft.jsonl"
    rows: list[dict] = []

    with safe_jsonl_writer(out_json) as jf, \
         ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:

        futures = {pool.submit(_multi_worker, i, txt, jf, rows): i
                   for i, txt in enumerate(data["text"])}
        for _ in tqdm_std(as_completed(futures), total=len(futures),
                          desc="Multi-turn SFT"):
            pass

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("✅  multi-turn: %s convs → %s / %s", len(rows), out_csv, out_json)
    return out_csv, out_json


# ──────────────────────────────────────────────────────────────────────────────
# public wrappers (called by executor.py)
# ──────────────────────────────────────────────────────────────────────────────
def run_single(cfg: Config):
    return _run_single(cfg)

def run_multi(cfg: Config):
    return _run_multi(cfg)
