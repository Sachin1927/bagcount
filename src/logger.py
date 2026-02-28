"""
Centralised logger factory for the BagCount pipeline.
Call get_logger(__name__) in any module to get a consistently
formatted logger without duplicating handler setup.
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger with a stdout StreamHandler.
    Calling this multiple times with the same name is safe — handlers
    are only attached once.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)

    logger.setLevel(level)
    # Prevent messages from propagating to the root logger (avoids duplicates)
    logger.propagate = False
    return logger
