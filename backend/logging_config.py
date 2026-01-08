"""
Simple logging configuration for local development
"""
import logging
from config import get_settings

def setup_logging() -> None:
    """Configure basic logging for local development"""
    settings = get_settings()

    # Configure basic logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {settings.log_level}")

def get_logger(name: str):
    """Get a logger instance"""
    return logging.getLogger(name)
