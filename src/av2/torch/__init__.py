"""AV2 Pytorch interfaces."""

import logging

logger = logging.Logger(__file__)

try:
    import torch
except ImportError as _:
    logger.error("Please install Pytorch to use this module.")
