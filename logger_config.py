from typing import Optional, Any
from pathlib import Path
import sys
import os
from loguru import logger


def setup_logger(
    log_file: Optional[str] = "loop.log",
    level: str = "DEBUG",
    console: bool = True,
    rotation: Optional[str] = None,
    retention: Optional[str] = "30 days",
    compression: Optional[str] = None,
    fmt: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level}</level> | <cyan>{file}:{line}</cyan> | {function}: <level>{message}</level>",
    enqueue: bool = True,
    **kwargs: Any,
 ) -> Any:
    """Configure and return a loguru logger.

    Args:
        log_file: Path to log file. If None, no file sink will be added.
        level: Logging level (e.g. "INFO", "DEBUG").
        console: Whether to add a stdout console sink (colorized).
        rotation: Rotation policy passed to loguru (e.g. "100 MB" or "00:00").
        retention: Retention policy passed to loguru (e.g. "30 days").
        compression: Compression for old logs (e.g. "zip").
        fmt: Format string for log messages.
        enqueue: Use thread/process queue for safe concurrent logging.
        **kwargs: Extra kwargs forwarded to loguru.logger.add for advanced uses.

    Returns:
        The configured loguru logger instance.

    Notes:
        This function will remove all existing sinks before adding the new ones.
        Call this from the program entrypoint (e.g. in main.py) if you need to
        control where logs are written. A module-level default logger is also
        exported below for convenience.
    """

    # Remove all existing sinks to ensure configuration is deterministic
    logger.remove()
    # Allow overriding level and log file via environment variables for quick debugging
    env_level = os.getenv("LOG_LEVEL")
    if env_level:
        level = env_level
    env_log_file = os.getenv("LOG_FILE")
    if env_log_file is not None:
        log_file = env_log_file

    if console:
        logger.add(sys.stdout, colorize=True, level='INFO', format=fmt, enqueue=enqueue)

    if log_file is not None:
        p = Path(log_file)
        if not p.parent.exists():
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                # best-effort: if directory can't be created, fall back to current dir
                p = Path(log_file).name

        logger.add(
            str(p),
            rotation=rotation,
            retention=retention,
            compression=compression,
            level=level,
            format=fmt,
            encoding="utf-8",
            enqueue=enqueue,
            **kwargs,
        )

    return logger


# Keep a module-level default logger for convenience. This preserves legacy
# behaviour where importing this module immediately provides a usable logger.
# Entry scripts can still call setup_logger(...) to reconfigure sinks.
logger = setup_logger()
