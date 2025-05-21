"""
Main module for the AudioBookSum book pipeline.
Provides both CLI and programmatic interfaces.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import our modules
from src.logging import configure_logging, get_logger
from src.extract import extract_text, extract_and_prepare_elements, identify_and_filter_watermarks
from src.transform import process_pdf_to_structured_json,  chunk_text_v2
from src.load import initialize_llm, identify_interesting_points_structured, highlight_interesting_points, save_summary_to_file, save_evidence_to_file
from src.models import SummaryAndEvidence, EvidenceSnippet, PipelineResult
from src.utils import validate_pdf_path
from src.metrics import (
    start_metrics_server, record_pipeline_run, record_extraction_run, 
    record_llm_call, time_extraction, time_transform, time_llm_processing, 
    time_highlighting_context, record_error
)
from src.config import (
    OUTPUT_DIR, DATA_DIR, 
    LLM_MODEL, MIN_SNIPPET_WORD_COUNT
)

# Configure logging
logger = configure_logging()

# Define cache directory (use actual path from config if it exists)
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process a PDF file to extract and highlight key points")
    
    parser.add_argument("pdf_path", help="Path to the PDF file to process")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory to save output files")
    parser.add_argument("--cache-dir", default=CACHE_DIR, help="Directory to save/load cache files")
    parser.add_argument("--use-mistral", action="store_true", help="Use Mistral OCR for text extraction")
    parser.add_argument("--use-chunking-v1", action="store_true", help="Use original chunking algorithm instead of v2")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Maximum size of each text chunk")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Number of overlapping characters between chunks")
    parser.add_argument("--no-save-intermediate", action="store_true", help="Don't save intermediate files")
    parser.add_argument("--metrics-port", type=int, default=8001, help="Port for metrics server (0 to disable)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                      help="Set log level")
    
    return parser.parse_args()


@time_extraction("extract_text")
def extract_stage(pdf_path: str, use_mistral: bool = False) -> List[Dict[str, Any]]:
    """
    Extraction stage of the pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        use_mistral: Whether to use Mistral OCR
        
    Returns:
        Extracted text blocks
    """
    logger.info("Starting extraction stage", pdf_path=pdf_path, use_mistral=use_mistral)
    
    # Extract text from PDF
    method_used = "mistral" if use_mistral else "pymupdf"
    try:
        text_blocks, method = extract_text(pdf_path, use_mistral=use_mistral)
        record_extraction_run(method, "success")
        logger.info("Extraction completed", 
                   block_count=len(text_blocks), 
                   method=method)
        return text_blocks
    except Exception as e:
        logger.exception("Extraction failed", 
                        error=str(e), 
                        method=method_used)
        record_extraction_run(method_used, "failure")
        record_error("extraction")
        raise


@time_transform()
def transform_stage(
    pdf_path: str, 
    text_blocks: List[Dict[str, Any]],
    use_chunking_v2: bool = True,
    chunk_size: int = 50000,
    chunk_overlap: int = 100,
    save_intermediate: bool = True,
    output_dir: str = OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Transformation stage of the pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        text_blocks: Extracted text blocks
        use_chunking_v2: Whether to use chunking v2
        chunk_size: Maximum chunk size
        chunk_overlap: Chunk overlap
        save_intermediate: Whether to save intermediate files
        output_dir: Output directory
        
    Returns:
        Dictionary with transformation results
    """
    logger.info("Starting transformation stage", pdf_path=pdf_path)
    
    # Process PDF to structured JSON
    try:
        output_json_path = os.path.join(output_dir, f"{Path(pdf_path).stem}_structure.json") if save_intermediate else None
        elements, structured_json = process_pdf_to_structured_json(pdf_path, output_json_path)
        logger.info("Generated structured JSON", element_count=len(elements))
        
        # Always use chunking v2
        logger.info("Using chunking algorithm v2")
        chunked_data = chunk_text_v2(structured_json, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Convert list of strings to structured format
        if isinstance(chunked_data, list) and all(isinstance(chunk, str) for chunk in chunked_data):
            logger.info("Converting chunking_v2 string output to structured format")
            structured_chunks = []
            for i, chunked_text in enumerate(chunked_data):
                structured_chunks.append({
                    "text": chunked_text,
                    "page": 0,  # Default page number
                    "bbox_list": [],  # Empty bounding box list
                    "chunk_index": i,
                    "metadata": {
                        "source": pdf_path,
                        "chunk_type": "v2"
                    }
                })
            chunked_data = structured_chunks
            
        logger.info("Generated chunks", chunk_count=len(chunked_data))
        
        return {
            "elements": elements,
            "structured_json": structured_json,
            "chunked_data": chunked_data
        }
    except Exception as e:
        logger.exception("Transformation failed", error=str(e))
        record_error("transformation")
        raise


@time_llm_processing(LLM_MODEL)
def load_stage(
    pdf_path: str,
    chunked_data: List[Dict[str, Any]],
    text_blocks: List[Dict[str, Any]],
    cache_dir: str = CACHE_DIR,
    save_intermediate: bool = True,
    output_dir: str = OUTPUT_DIR
) -> Dict[str, Any]:
    """
    Load stage of the pipeline (LLM processing and highlighting).
    
    Args:
        pdf_path: Path to the PDF file
        chunked_data: Chunked text data
        text_blocks: Original text blocks
        cache_dir: Cache directory
        save_intermediate: Whether to save intermediate files
        output_dir: Output directory
        
    Returns:
        Dictionary with load stage results
    """
    logger.info("Starting load stage (LLM processing)", pdf_path=pdf_path)
    
    # Initialize LLM
    try:
        llm = initialize_llm()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache file path
        pdf_filename = os.path.basename(pdf_path)
        cache_file_path = os.path.join(cache_dir, f"{Path(pdf_filename).stem}_cache.json")
        
        # Identify interesting points
        logger.info("Identifying interesting points using LLM")
        summary, processed_segments = identify_interesting_points_structured(
            chunked_data=chunked_data,
            llm=llm,
            output_schema=SummaryAndEvidence,
            text_blocks=text_blocks,
            
            cache_file_path=cache_file_path,
            min_snippet_word_count=MIN_SNIPPET_WORD_COUNT,
        )
        
        record_llm_call(LLM_MODEL, "success")
        logger.info("Identified summary and evidence segments", 
                   evidence_count=len(processed_segments))
        
        # Save results if requested
        result_files = {}
        
        if save_intermediate and summary:
            summary_file = os.path.join(output_dir, f"{Path(pdf_path).stem}_summary.txt")
            save_summary_to_file(summary, summary_file)
            result_files["summary_file"] = summary_file
            
        if save_intermediate and processed_segments:
            evidence_file = os.path.join(output_dir, f"{Path(pdf_path).stem}_evidence.txt")
            save_evidence_to_file(processed_segments, evidence_file)
            result_files["evidence_file"] = evidence_file
        
        # Highlight the PDF
        highlighted_file = os.path.join(output_dir, f"{Path(pdf_path).stem}_highlighted.pdf")
        
        with time_highlighting_context():
            highlight_interesting_points(pdf_path, processed_segments, highlighted_file)
            
        result_files["highlighted_file"] = highlighted_file
        
        return {
            "summary": summary,
            "evidence": processed_segments,
            "files": result_files
        }
    except Exception as e:
        logger.exception("Load stage failed", error=str(e))
        record_llm_call(LLM_MODEL, "failure")
        record_error("load")
        raise


def process_pdf(
    pdf_path: str,
    output_dir: str = OUTPUT_DIR,
    cache_dir: str = CACHE_DIR,
    use_mistral: bool = False,
    use_chunking_v2: bool = True,
    chunk_size: int = 50000,
    chunk_overlap: int = 100,
    save_intermediate: bool = True
) -> PipelineResult:
    """
    Process a PDF file using the complete pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save output files
        cache_dir: Directory to save/load cache files
        use_mistral: Whether to use Mistral OCR
        use_chunking_v2: Whether to use chunking v2
        chunk_size: Maximum chunk size
        chunk_overlap: Chunk overlap
        save_intermediate: Whether to save intermediate files
        
    Returns:
        PipelineResult object
    """
    start_time = time.time()
    logger.info("Starting pipeline processing", 
               pdf_path=pdf_path, 
               output_dir=output_dir,
               chunking_version="v2" if use_chunking_v2 else "v1")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate default output file path for error cases
    default_output_file = os.path.join(output_dir, f"{Path(pdf_path).stem}_highlighted.pdf")
    
    try:
        # Validate PDF path
        validate_pdf_path(pdf_path)
        
        # Extract stage
        text_blocks = extract_stage(pdf_path, use_mistral)
        
        # Transform stage
        transform_result = transform_stage(
            pdf_path=pdf_path,
            text_blocks=text_blocks,
            use_chunking_v2=use_chunking_v2,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            save_intermediate=save_intermediate,
            output_dir=output_dir
        )
        
        # Load stage
        load_result = load_stage(
            pdf_path=pdf_path,
            chunked_data=transform_result["chunked_data"],
            text_blocks=text_blocks,
            cache_dir=cache_dir,
            save_intermediate=save_intermediate,
            output_dir=output_dir
        )
        
        # Create result object
        result = PipelineResult(
            status="success",
            summary=load_result["summary"],
            evidence=load_result["evidence"],
            source_file=pdf_path,
            output_file=load_result["files"]["highlighted_file"],
            processing_time=time.time() - start_time
        )
        
        logger.info("Pipeline completed successfully", 
                   processing_time=f"{result.processing_time:.2f}s",
                   evidence_count=len(load_result["evidence"]),
                   output_file=load_result["files"]["highlighted_file"])
        record_pipeline_run("success")
        return result
        
    except Exception as e:
        logger.exception("Pipeline failed", 
                        error=str(e), 
                        pdf_path=pdf_path)
        record_pipeline_run("failure")
        record_error("pipeline")
        return PipelineResult(
            status="error",
            error=str(e),
            source_file=pdf_path,
            output_file=default_output_file,
            processing_time=time.time() - start_time
        )


def main():
    """Main entry point for CLI."""
    args = parse_args()
    
    # Start metrics server if requested
    if args.metrics_port > 0:
        metrics_started = start_metrics_server(args.metrics_port)
        if metrics_started:
            logger.info("Metrics server started", port=args.metrics_port)
        else:
            logger.warning("Failed to start metrics server")
    
    # Process the PDF
    result = process_pdf(
        pdf_path=args.pdf_path,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        use_mistral=args.use_mistral,
        use_chunking_v2=not args.use_chunking_v1,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        save_intermediate=not args.no_save_intermediate
    )
    
    # Output results
    if result.status == "success":
        logger.info("Pipeline completed successfully")
        logger.info("Highlighted PDF saved", path=result.output_file)
        
        if result.summary:
            print("\n=== Summary ===\n")
            print(result.summary[:500] + "..." if len(result.summary) > 500 else result.summary)
            
        print(f"\nProcessed {len(result.evidence)} evidence points")
        print(f"Processing time: {result.processing_time:.2f} seconds")
    else:
        logger.error("Pipeline failed", error=result.error)
        sys.exit(1)


if __name__ == "__main__":
    main()