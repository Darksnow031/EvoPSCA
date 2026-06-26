import logging
import os
_configured = False

def configure_logging(log_file=None, level=logging.INFO, fmt='%(asctime)s %(levelname)s %(message)s'):
    global _configured
    if _configured:
        return
    logging.basicConfig(level=level, format=fmt)
    if log_file:
        try:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(fmt))
            logging.getLogger().addHandler(fh)
        except Exception:
            pass
    _configured = True

def get_logger(name=None, log_file=None, level=logging.INFO):
    if not logging.getLogger().handlers:
        configure_logging(log_file=log_file, level=level)
    logger = logging.getLogger(name)
    return logger
