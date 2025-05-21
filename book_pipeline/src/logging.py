"""
Logging configuration module for the book pipeline.
"""

import os
import sys
import time
import logging
import structlog
from typing import Optional, Dict, Any

# Define log directory
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Log file path
LOG_FILE = os.path.join(LOG_DIR, "book_pipeline.log")

def configure_logging(
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_file: Optional[str] = LOG_FILE,
):
    """
    Configure structured logging for the application.
    
    Args:
        console_level: Logging level for console output
        file_level: Logging level for file output
        log_file: Path to log file
    """
    # Set up timestamp processor
    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
    
    # Configure structlog pre-processors
    pre_chain = [
        # Add timestamps with human-readable format
        timestamper,
        # Add logger name
        structlog.stdlib.add_logger_name,
        # Add log level
        structlog.stdlib.add_log_level,
        # Add record file and line number
        structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.FILENAME,
             structlog.processors.CallsiteParameter.LINENO]
        ),
        # Convert exceptions to string
        structlog.processors.format_exc_info,
        # Add traceback
        structlog.processors.StackInfoRenderer(),
    ]
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *pre_chain,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Create formatter for console output
    console_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=True),
        foreign_pre_chain=pre_chain,
    )
    
    # Create formatter for file output
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=pre_chain,
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler if log file is provided
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(min(console_level, file_level))
    
    # Remove any existing handlers
    for hdlr in root_logger.handlers:
        root_logger.removeHandler(hdlr)
    
    # Add our handlers
    for handler in handlers:
        root_logger.addHandler(handler)

    return structlog.get_logger()


def get_logger(name: str = None):
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)