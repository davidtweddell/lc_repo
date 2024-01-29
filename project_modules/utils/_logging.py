#===============================================
# set up logging
#===============================================
import logging
from typing import Union

def get_logger(fn: Union[str, None] = None,
               level: int           = logging.INFO,
               ) -> logging.Logger:


    # standard formatteres
    datefmt    = "%Y-%m-%d %H:%M:%S"
    stream_fmt = "[{asctime}] [{funcName}] {message:s}"
    file_fmt   = "[{asctime}] {funcName:<15s} {levelname:<10s}: {message:s}" 

    # stream handler and formatter
    sh = logging.StreamHandler()
    sf = logging.Formatter(stream_fmt, style = '{', datefmt = datefmt)
    sh.setFormatter(sf)

    if fn is not None:
        # optional file hander and formatter
        fh = logging.FileHandler(fn, mode='w')
        ff = logging.Formatter(file_fmt, style = '{', datefmt = datefmt)
        fh.setFormatter(ff)
        handlers = [sh, fh]

    else:
        handlers = [sh]


    # basic configuration
    logging.basicConfig(
        level=level,
        handlers=handlers,
    )

    # logger instance
    logger = logging.getLogger(__name__)

    logger.info("Logger initialized.")
    if fn is not None:
        logger.info(f"Logging to file: {fn}")

    return logger