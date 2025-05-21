"""
Pydantic models for data validation across the pipeline.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field


class EvidenceSnippet(BaseModel):
    """Represents a single verbatim evidence snippet."""
    verbatim_text: str = Field(..., 
        description="The exact verbatim text snippet extracted from the document, including any '[Page X]' annotation if present.")


class SummaryAndEvidence(BaseModel):
    """Schema for the LLM output containing summary and evidence."""
    summary: str = Field(..., 
        description="A concise summary based solely on the provided document content.")
    evidence: List[EvidenceSnippet] = Field(..., 
        description="A list of verbatim evidence snippets supporting the key points of the summary.")


class TextBlock(BaseModel):
    """Represents a block of text extracted from a PDF document."""
    page: int = Field(..., description="The page number (0-indexed)")
    bbox: Optional[Tuple[float, float, float, float]] = Field(None, 
        description="The bounding box coordinates (x0, y0, x1, y1)")
    text: str = Field(..., description="The text content")
    block_no: Optional[int] = Field(None, description="Block number assigned by the PDF extractor")
    block_type: Optional[int] = Field(None, description="Block type assigned by the PDF extractor")
    text_height: Optional[float] = Field(None, description="Height of the text")
    category: Optional[str] = Field(None, description="Category of the text block (main, footnote, extra)")
    norm_text: Optional[str] = Field(None, description="Normalized text for matching")


class ExtractedRegion(BaseModel):
    """Represents a region extracted via the image-based approach."""
    occupy_space: Any = Field(..., description="PyMuPDF Rect object of the region")
    content: str = Field(..., description="The text content")
    text_height_median: float = Field(..., description="Median height of text in the region")
    min_height: Optional[float] = Field(None, description="Minimum height of text in the region")
    page_num: int = Field(..., description="Page number (0-indexed)")


class NodeContent(BaseModel):
    """Content of a node in the document hierarchy."""
    content: Optional[str] = Field(None, description="The text content")
    occupy_space: Optional[Any] = Field(None, description="PyMuPDF Rect representing the space occupied")
    text_height_median: Optional[float] = Field(None, description="Median height of text")
    page_num: Optional[int] = Field(None, description="Page number")


class TextChunk(BaseModel):
    """Represents a chunk of text after chunking."""
    text: str = Field(..., description="The text content")
    page: Optional[int] = Field(None, description="Primary page number")
    bbox_list: Optional[List[Any]] = Field(None, description="List of bounding boxes")
    start_position: Optional[int] = Field(None, description="Start position in the combined text")
    chunk_index: Optional[int] = Field(None, description="Index of the chunk")


class ProcessedEvidence(BaseModel):
    """Represents processed evidence with page and bbox information."""
    page: int = Field(..., description="Page number")
    text: str = Field(..., description="The text content with page annotation")
    bbox: Optional[Any] = Field(None, description="Bounding box for highlighting")


class PipelineResult(BaseModel):
    """Full result of the pipeline execution."""
    summary: Optional[str] = Field(None, description="Generated summary")
    evidence: List[ProcessedEvidence] = Field(default_factory=list, description="List of evidence snippets")
    document_structure: Optional[Dict[str, Any]] = Field(None, description="Hierarchical document structure")
    source_file: str = Field(..., description="Path to the source PDF")
    output_file: str = Field(..., description="Path to the output highlighted PDF")
    error: Optional[str] = Field(None, description="Error message if pipeline failed")
    status: str = Field("success", description="Pipeline execution status (success/failure)")
    processing_time: Optional[float] = Field(None, description="Total processing time in seconds")

