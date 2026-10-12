import logging
import time


def utc_log_formatter():
    """Format server log records with explicit UTC millisecond timestamps."""
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03dZ %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time.gmtime
    return formatter
