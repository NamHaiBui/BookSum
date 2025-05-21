"""
Custom exceptions for the book pipeline.
These provide more specific error handling and better debugging.
"""

class PipelineError(Exception):
    """Base class for all pipeline exceptions."""
    pass


class PDFProcessingError(PipelineError):
    """Raised when there's an error processing a PDF file."""
    pass


class OCRError(PDFProcessingError):
    """Raised when there's an error with OCR processing."""
    pass


class TextExtractionError(PDFProcessingError):
    """Raised when text extraction from a PDF fails."""
    pass


class TextChunkingError(PipelineError):
    """Raised when there's an error chunking text."""
    pass


class StructureExtractionError(PipelineError):
    """Raised when there's an error extracting document structure."""
    pass


class LLMError(PipelineError):
    """Raised when there's an error with the LLM processing."""
    pass


class APIKeyError(PipelineError):
    """Raised when an API key is missing or invalid."""
    pass


class HighlightingError(PipelineError):
    """Raised when there's an error highlighting a PDF."""
    pass


class ConfigurationError(PipelineError):
    """Raised when there's an error with the pipeline configuration."""
    pass


class ValidationError(PipelineError):
    """Raised when input or output validation fails."""
    pass