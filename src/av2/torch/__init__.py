"""AV2 Pytorch interface."""

import logging

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError as _:
    logger.error("Please install Pytorch to use this module.")
