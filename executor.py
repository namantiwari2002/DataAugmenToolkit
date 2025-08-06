"""
A single entry-point for *all* pipelines.  UI + CLI both call `run(cfg)`.
"""
import importlib
import logging
from config import Config

log = logging.getLogger(__name__)

_MAP = {
    "single-sft"  : ("sft_data",       "run_single"),
    "multi-sft"   : ("sft_data",       "run_multi"),
    "single-align": ("alignment_data", "run_single"),
    "multi-align" : ("alignment_data", "run_multi"),
}
def run(cfg: Config):
    mod_name, fn_name = _MAP[cfg.mode]
    mod = importlib.import_module(mod_name)
    run_fn = getattr(mod, fn_name)
    log.info("▶️  Running mode=%s via %s.%s", cfg.mode, mod_name, fn_name)
    return run_fn(cfg)
