from __future__ import annotations

import logging


def get_logger(name: str = "spineseg_perfbench") -> logging.Logger:
    return logging.getLogger(name)
