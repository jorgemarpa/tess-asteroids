import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

from .tpf import MovingTargetTPF  # noqa: E402

__version__ = "0.1.0"
__all__ = ["MovingTargetTPF"]
