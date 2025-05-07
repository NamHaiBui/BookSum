import logging
import os
import re
from typing import Dict, Any, Tuple, Optional

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """
    Sets up logging configuration based on the provided config.
    
    Args:
        config: Dictionary containing logging configuration.
        
    Returns:
        Logger object configured according to the settings.
    """
    log_level = getattr(logging, config["logging"]["level"])
    log_file = config["paths"]["log_file"]
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("AudioBookPipeline")

def ensure_directory(directory_path: str) -> None:
    """
    Ensures the specified directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory.
    """
    os.makedirs(directory_path, exist_ok=True)

def extract_timestamp_from_text(text: str, timestamp_pattern: re.Pattern) -> Optional[Tuple[str, str]]:
    """
    Extracts the timestamp tuple from the beginning of a text snippet.
    
    Args:
        text: The text containing a timestamp.
        timestamp_pattern: Compiled regex pattern to match timestamps.
        
    Returns:
        A tuple (start, end) if found, otherwise None.
    """
    match = timestamp_pattern.match(text)
    if match:
        return match.groups()
    return None

def get_filename_without_extension(file_path: str) -> str:
    """
    Extracts the base filename without extension from a file path.
    
    Args:
        file_path: The full path to a file.
        
    Returns:
        The filename without directory or extension.
    """
    return os.path.splitext(os.path.basename(file_path))[0]