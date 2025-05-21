"""
Constants for the book pipeline that are unlikely to change between runs.
These are different from configuration parameters as they are not meant to be modified
by the user and are intrinsic to the algorithm's function.
"""

# Constants for Node hierarchy building
DUMMY_NODE_HEIGHT = 1000  # Used in hierarchy calculation as a sentinel value

# Regular expression patterns
PAGE_PATTERN = r'^\[Page:\s*(\d+)\s*\]\s*(.*)'

# Block break marker used during chunking
BLOCK_BREAK_MARKER = " [BLOCK_BREAK] "

# Special marker for nodes in the document tree
DOCUMENT_ROOT_TITLE = "Document Root"

# Watermark and filtering
MIN_SIMILARITY_THRESHOLD = 0.6  # Minimum similarity threshold for matching text blocks