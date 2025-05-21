#!/usr/bin/env python
import argparse
import logging
import os
import sys
import re
import yaml
import time
from typing import Dict, Any
from dotenv import load_dotenv

# Import modules
from src.utils import setup_logging, ensure_directory, get_filename_without_extension
from src.audio_processor import extract_and_prepare_audio
from src.transcription import transcribe_pipeline
from src.srt_processor import process_srt, chunking_srt_by_chapter, generate_chapter_patterns
from src.llm_handler import initialize_llm, identify_interesting_points_structured, SummaryAndEvidence
from src.snippet_extractor import merge_and_extract_snippets

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration YAML file.
        
    Returns:
        Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

def main():
    """Main function to orchestrate the audio book processing pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Audio Book Processing Pipeline")
    parser.add_argument('--input_file', required=True, help="Path to the input audio/video file")
    parser.add_argument('--config', default="config/config.yaml", help="Path to the configuration file")
    parser.add_argument('--output_dir', help="Custom output directory (overrides config)")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup custom output directory if provided
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    
    # Setup logging
    logger = setup_logging(config)
    logger.info(f"Starting audio book processing pipeline for: {args.input_file}")
    
    # Initialize directories
    input_filename = get_filename_without_extension(args.input_file)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    
    output_dir = os.path.join(base_dir, config["paths"]["output_dir"])
    intermediate_dir = os.path.join(base_dir, config["paths"]["intermediate_dir"])
    cache_dir = os.path.join(base_dir, config["paths"]["cache_dir"])
    summary_dir = os.path.join(output_dir, "summaries")
    snippets_dir = os.path.join(output_dir, "snippets", input_filename)
    
    # Ensure directories exist
    for directory in [output_dir, intermediate_dir, cache_dir, summary_dir, snippets_dir]:
        ensure_directory(directory)
    
    # Define intermediate file paths
    audio_path = os.path.join(intermediate_dir, f"{input_filename}.wav")
    srt_path = os.path.join(intermediate_dir, f"{input_filename}.srt")
    cache_file_path = os.path.join(cache_dir, f"{input_filename}_summary_evidence_cache.json")
    
    # Record start time
    start_time = time.time()
    
    # Step 1: Extract and prepare audio
    logger.info("STEP 1: Extracting and preparing audio")
    if not os.path.exists(audio_path):
        audio_success = extract_and_prepare_audio(args.input_file, audio_path, config)
        if not audio_success:
            logger.error("Failed to extract audio. Exiting pipeline.")
            sys.exit(1)
    else:
        logger.info(f"Using existing audio file: {audio_path}")
    
    # Step 2: Transcribe audio to SRT
    logger.info("STEP 2: Transcribing audio to SRT")
    if not os.path.exists(srt_path):
        transcription_success = transcribe_pipeline(audio_path, srt_path, config)
        if not transcription_success:
            logger.error("Failed to transcribe audio. Exiting pipeline.")
            sys.exit(1)
    else:
        logger.info(f"Using existing SRT file: {srt_path}")
    
    # Step 3: Process SRT and chunk by chapter
    logger.info("STEP 3: Processing SRT and chunking by chapter")
    processed_srt_items, original_srt_items = process_srt(srt_path)
    if not processed_srt_items or not original_srt_items:
        logger.error("Failed to process SRT file. Exiting pipeline.")
        sys.exit(1)
    
    # Generate chapter patterns
    chapter_pattern_text, chapter_pattern_int = generate_chapter_patterns(config)
    
    # Compile timestamp pattern
    timestamp_pattern_str = config["srt_processing"]["timestamp_pattern_rg"]
    timestamp_pattern = re.compile(timestamp_pattern_str)
    
    # Chunk SRT by chapter
    chunked_data = chunking_srt_by_chapter(
        processed_srt_items,
        chapter_pattern_text,
        chapter_pattern_int,
        timestamp_pattern
    )
    
    if not chunked_data:
        logger.warning("No chapters detected. Using the entire SRT as a single chunk.")
        # Create a single chunk with all SRT items
        chunked_data = [{
            "start_index": processed_srt_items[0].index,
            "end_index": processed_srt_items[-1].index,
            "chunk": processed_srt_items
        }]
    
    logger.info(f"SRT processed successfully. Found {len(chunked_data)} chunks.")
    
    # Step 4: Initialize LLM
    logger.info("STEP 4: Initializing LLM")
    llm = initialize_llm(config)
    if not llm:
        logger.error("Failed to initialize LLM. Exiting pipeline.")
        sys.exit(1)
    
    # Step 5: Generate summary and evidence
    logger.info("STEP 5: Generating summary and evidence")
    summary, evidence = identify_interesting_points_structured(
        chunked_data,
        llm,
        SummaryAndEvidence,
        original_srt_items,
        config,
        cache_file_path,
        timestamp_pattern
    )
    
    if not summary and not evidence:
        logger.error("Failed to generate summary and evidence. Exiting pipeline.")
        sys.exit(1)
    
    # Step 6: Save summary to file
    logger.info("STEP 6: Saving summary")
    summary_path = os.path.join(summary_dir, f"{input_filename}_summary.txt")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info(f"Summary saved to {summary_path}")
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
    
    # Step 7: Extract and save audio snippets
    logger.info("STEP 7: Extracting and saving audio snippets")
    total_processed, total_saved = merge_and_extract_snippets(
        evidence,
        audio_path,
        snippets_dir,
        config
    )
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Final summary
    logger.info("="*50)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Processed file: {args.input_file}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"Snippets saved to: {snippets_dir}")
    logger.info(f"Total snippets processed: {total_processed}")
    logger.info(f"Total snippets saved: {total_saved}")
    logger.info(f"Total time: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info("="*50)

if __name__ == "__main__":
    main()