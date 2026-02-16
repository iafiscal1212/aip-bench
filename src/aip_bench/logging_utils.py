"""
Bench Logging: Structured logging for benchmark pipelines.

Provides a configured logger with optional file output and
structured formatting for benchmark runs.

Usage:
    from aip_bench.logging_utils import get_logger

    log = get_logger(__name__)
    log.info("Running benchmark", extra={"benchmark": "halueval"})

Author: Carmen Esteban
"""

import logging
import sys


_CONFIGURED = False


def get_logger(name="aip_bench", level=None):
    """Get a configured logger for AIP Bench.

    Parameters
    ----------
    name : str
        Logger name (usually __name__).
    level : int, optional
        Logging level. Defaults to INFO.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    global _CONFIGURED
    logger = logging.getLogger(name)

    if not _CONFIGURED:
        level = level or logging.INFO
        logger.setLevel(level)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(level)
            fmt = logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%H:%M:%S",
            )
            handler.setFormatter(fmt)
            logger.addHandler(handler)

        _CONFIGURED = True

    return logger


def setup_file_logging(path, level=None):
    """Add file handler to the AIP Bench logger.

    Parameters
    ----------
    path : str
        Log file path.
    level : int, optional
        File logging level. Defaults to DEBUG.
    """
    level = level or logging.DEBUG
    logger = logging.getLogger("aip_bench")
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)


def quiet():
    """Suppress AIP Bench log output (useful for tests)."""
    logging.getLogger("aip_bench").setLevel(logging.WARNING)


def verbose():
    """Enable verbose (DEBUG) logging."""
    logging.getLogger("aip_bench").setLevel(logging.DEBUG)
