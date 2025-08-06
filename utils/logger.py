"""
Shared logging helpers.

* init_root()           – set a single stdout handler for the whole app
* get_stream_handler()  – attach a second handler that writes to an
                          in-memory StringIO; returns (handler, buffer)
* remove_stream_handler() – detach the buffer handler

The global CURRENT_LOG_BUF is updated so other modules (tqdm wrapper)
can write progress into the same buffer.
"""
from __future__ import annotations
import io, logging, sys

# --------------------------------------------------------------------------- #
CURRENT_LOG_BUF: io.StringIO | None = None    # ←  NEEDED GLOBALLY
# --------------------------------------------------------------------------- #


def init_root(level: int = logging.INFO) -> None:
    """Call once, early, to configure the root logger."""
    if not logging.getLogger().handlers:
        hdl = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
        hdl.setFormatter(logging.Formatter(fmt))
        logging.basicConfig(level=level, handlers=[hdl])


def get_stream_handler() -> tuple[logging.Handler, io.StringIO]:
    """Attach a temporary StreamHandler that logs into a StringIO buffer."""
    global CURRENT_LOG_BUF
    buf = io.StringIO()
    hdl = logging.StreamHandler(buf)
    logging.getLogger().addHandler(hdl)
    CURRENT_LOG_BUF = buf           # ★ share with tqdm wrapper
    return hdl, buf


def remove_stream_handler(hdl: logging.Handler) -> None:
    """Detach the temporary handler and clear the global pointer."""
    global CURRENT_LOG_BUF
    logging.getLogger().removeHandler(hdl)
    CURRENT_LOG_BUF = None
