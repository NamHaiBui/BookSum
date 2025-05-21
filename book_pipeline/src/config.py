import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Project paths
ROOT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash-preview-04-17")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# PDF Processing Configuration
PDF_ZOOM_FACTOR = int(os.getenv("PDF_ZOOM_FACTOR", "2"))
ADAPTIVE_THRESH_BLOCK_SIZE = int(os.getenv("ADAPTIVE_THRESH_BLOCK_SIZE", "13"))
ADAPTIVE_THRESH_C = int(os.getenv("ADAPTIVE_THRESH_C", "2"))
MORPH_KERNEL_SIZE = tuple(map(int, os.getenv("MORPH_KERNEL_SIZE", "8,8").split(",")))
DILATE_ITERATIONS = int(os.getenv("DILATE_ITERATIONS", "1"))
CANNY_THRESH1 = int(os.getenv("CANNY_THRESH1", "50"))
CANNY_THRESH2 = int(os.getenv("CANNY_THRESH2", "200"))
MIN_REGION_WIDTH = float(os.getenv("MIN_REGION_WIDTH", "8.0"))
MIN_REGION_HEIGHT = float(os.getenv("MIN_REGION_HEIGHT", "8.0"))
MERGE_ITERATIONS = int(os.getenv("MERGE_ITERATIONS", "6"))
REGION_PADDING = float(os.getenv("REGION_PADDING", "0.3"))
LINE_BREAK_THRESHOLD = float(os.getenv("LINE_BREAK_THRESHOLD", "0.6"))

# Text Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "50000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MIN_SNIPPET_WORD_COUNT = int(os.getenv("MIN_SNIPPET_WORD_COUNT", "5"))

# Highlighting Configuration
HIGHLIGHT_COLOR = tuple(map(float, os.getenv("HIGHLIGHT_COLOR", "0,1,1").split(",")))  # cyan in RGB

SUMMARY_EVIDENCE_PROMPT_STRUCTURED = os.getenv(
    "SUMMARY_EVIDENCE_PROMPT_STRUCTURED", 
    """
    Please analyze the provided document content and create:
    1. A concise summary of the main points
    2. A list of verbatim text snippets from the document that support key points
    Focus on extracting the most important information that represents the core ideas.
    For text snippets:
    - Extract verbatim quotes (exact text as it appears in the document)
    - Include any page annotations if present (e.g., [Page:5])
    - If there is no page annotation at the beginning of a snippet, use the previous page annotation and add it into the beginning of the snippet.
    - Each snippet should be meaningful and self-contained
    """
)
SUMMARY_PROMPT = os.getenv(
    "SUMMARY_PROMPT", 
    """Create a more concise summary of the following text. 
Focus on coherence, clarity, and readability. Document text:"""
)