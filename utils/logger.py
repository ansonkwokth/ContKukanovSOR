import logging
import os
from datetime import datetime

LOG_COLORS = {
    "DEBUG": "\033[94m",    # Blue
    "INFO": "\033[92m",     # Green
    "WARNING": "\033[93m",  # Yellow
    "ERROR": "\033[91m",    # Red
    "CRITICAL": "\033[1;91m",  # Bold Red
    "RESET": "\033[0m"      # Reset
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        color = LOG_COLORS.get(levelname, "")
        reset = LOG_COLORS["RESET"]
        message = super().format(record)
        return f"{color}{message}{reset}"

def get_logger(name: str, log_level=logging.INFO):
    os.makedirs("logs", exist_ok=True)
    log_filename = datetime.now().strftime("logs/run_%Y%m%d_%H%M%S.log")

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with colors
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch_formatter = ColoredFormatter("[%(levelname)s] %(message)s")
    ch.setFormatter(ch_formatter)

    # File handler (plain)
    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh_formatter = logging.Formatter(
        "%(asctime)s — %(name)s — [%(levelname)s] — %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fh_formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

