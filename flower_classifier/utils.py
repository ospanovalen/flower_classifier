import logging
import os
import sys
from pathlib import Path


def init_basic_logger(
    name: str,
    level: int = None,
    with_tqdm: bool = False,
    file_handler: Path = None,
) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is None:
        logger.setLevel(os.environ.get("LOGLEVEL", "DEBUG").upper())
    else:
        logger.setLevel(level)
    if len(logger.handlers) == 0 and not with_tqdm:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter(fmt="[%(asctime)s: %(levelname)s %(name)s] %(message)s")
        )
        logger.addHandler(handler)
    if file_handler is not None:
        handler = logging.FileHandler(file_handler)
        handler.setFormatter(
            logging.Formatter(fmt="[%(asctime)s: %(levelname)s %(name)s] %(message)s")
        )
        logger.addHandler(handler)
    return logger
