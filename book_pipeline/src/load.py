"""
Load module for the book pipeline.
Handles LLM processing and highlighting of the PDF.
"""

import os
import re
import json
import logging
import difflib
import datetime
from typing import List, Dict, Any, Tuple, Optional, DefaultDict, Type, Union
from collections import defaultdict

import pymupdf
import cv2
import numpy as np
from pydantic import BaseModel

from .exceptions import LLMError, HighlightingError, APIKeyError
from .utils import extract_page_number_from_text, find_matching_block
from .models import EvidenceSnippet, SummaryAndEvidence, ProcessedEvidence
from .config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MIN_SNIPPET_WORD_COUNT, HIGHLIGHT_COLOR, SUMMARY_EVIDENCE_PROMPT_STRUCTURED, SUMMARY_PROMPT

# Set up logging
logger = logging.getLogger(__name__)


def initialize_llm():
    """
    Initialize the LLM client for use in the pipeline.
    
    Returns:
        An initialized LLM client
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        # Check for API key
        if not GOOGLE_API_KEY:
            raise APIKeyError("GOOGLE_API_KEY not found in environment variables")
            
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE
        )
        
        logger.info(f"Initialized LLM client with model: {LLM_MODEL}")
        return llm
        
    except ImportError as e:
        logger.error(f"Failed to import required libraries: {str(e)}")
        raise LLMError(f"Required packages not installed: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {str(e)}")
        raise LLMError(f"Failed to initialize LLM: {str(e)}")


def summarize_text(text: str, llm) -> str:
    """
    Use Gemini to create a more concise summary of existing text.
    
    Args:
        text: The original text to summarize
        llm: Initialized LLM client
        
    Returns:
        A more concise summary
    """
    if not text or len(text.strip()) < 50:
        return text
        
    try:
        summarize_prompt = SUMMARY_PROMPT
        if not summarize_prompt:
            logger.error("Summary  prompt template is empty")
            return text
        response = llm.invoke(summarize_prompt)
        summarized = response.content.strip()
        
        if summarized:
            logger.info(f"Created concise summary: {len(text)} chars → {len(summarized)} chars")
            return summarized
        else:
            logger.warning("Empty response from LLM when summarizing text")
            return text
            
    except Exception as e:
        logger.error(f"Error summarizing text: {str(e)}")
        return text


def process_segment(
    raw_snippet_text: str,
    source_page_hint: Optional[int], 
    page_to_blocks: DefaultDict[int, List[Dict[str, Any]]],
    min_word_threshold: int = MIN_SNIPPET_WORD_COUNT
) -> Optional[Dict[str, Any]]:
    """
    Processes a raw snippet text provided by the LLM.
    
    Args:
        raw_snippet_text: Raw text snippet from the LLM
        source_page_hint: Hint about the page number from context
        page_to_blocks: Dictionary mapping page numbers to blocks
        min_word_threshold: Minimum number of words for a valid snippet
        
    Returns:
        Dictionary with processed evidence or None if invalid
    """
    original_segment_text = raw_snippet_text.strip()
    if len(original_segment_text.split()) < min_word_threshold:
        logger.debug(f"Snippet too short (< {min_word_threshold} words): '{original_segment_text}'")
        return None

    page_annotation = extract_page_number_from_text(original_segment_text)
    text_for_matching = re.sub(r'^\[Page:\s*\d+\]\s*', '', original_segment_text).strip()
    if not text_for_matching:
        logger.debug("No text content after removing page annotation")
        return None

    page_hint_for_search = page_annotation if page_annotation is not None else source_page_hint

    matched_block, match_similarity = find_matching_block(
        text_for_matching, page_hint_for_search, page_to_blocks
    )

    if matched_block:
        actual_page_num = matched_block.get("page", 0)
        correct_page_annotation_str = f"[Page:{actual_page_num}]"

        if page_annotation == actual_page_num:
            display_text = original_segment_text
            if page_annotation is None and actual_page_num >= 0:
                display_text = f"{correct_page_annotation_str} {text_for_matching}"
        elif page_annotation is not None and page_annotation != actual_page_num:
            display_text = f"{correct_page_annotation_str} {text_for_matching}"
        else:
            display_text = f"{correct_page_annotation_str} {text_for_matching}"

        return {
            "page": actual_page_num,
            "text": display_text.strip(),
            "bbox": matched_block.get("bbox"),
        }
    else:
        logger.debug("No matching block found for segment text. Assigning default page annotation.")
        assigned_page_num = page_hint_for_search if page_hint_for_search is not None else 0
        assigned_page_annotation_str = f"[Page:{assigned_page_num}]"

        if page_annotation is None:
            display_text = f"{assigned_page_annotation_str} {text_for_matching}"
        else:
            display_text = original_segment_text

        return {
            "page": assigned_page_num,
            "text": display_text.strip(),
            "bbox": None,
        }


def identify_interesting_points_structured(
    chunked_data: List[Dict[str, Any]],
    llm, 
    output_schema: Type[BaseModel],
    text_blocks: List[Dict[str, Any]],
    cache_file_path: str,
    min_snippet_word_count: int = MIN_SNIPPET_WORD_COUNT,
    is_one_p: bool = True
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Identifies summary and evidence snippets using an LLM with structured output.
    
    Args:
        chunked_data: List of dictionaries with text chunks
        llm: Initialized LLM client 
        output_schema: Pydantic model for structured LLM output
        text_blocks: Original list of text blocks from PDF extraction
        cache_file_path: Path to cache file for results
        min_snippet_word_count: Minimum word count for evidence snippets
        is_one_p: Whether to process all chunks in a single request
        
    Returns:
        Tuple of (summary, processed evidence snippets)
    """
    summary_result: Optional[str] = None
    processed_segments: List[Dict[str, Any]] = []

    if not chunked_data:
        logger.warning("No chunked data provided")
        return None, []

    # Prepare Block Index
    page_to_blocks: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for block in text_blocks:
        content = block.get("text", "")
        if not content and "content" in block:
            content = block.get("content", "")
            
        page_num = block.get("page") if "page" in block else block.get("page_num")
        if content and page_num is not None:
            block["norm_text"] = ' '.join(content.split())
            page_to_blocks[page_num].append(block)
    logger.info("Prepared page-to-blocks index")

    # Cache directory setup
    cache_dir = os.path.dirname(cache_file_path)
    cache_base_name = os.path.basename(cache_file_path).split('.')[0]
    chunks_dir = os.path.join(cache_dir, f"{cache_base_name}_chunks")
    master_cache_path = os.path.join(cache_dir, f"{cache_base_name}_master.json")
    
    # Create cache directories if they don't exist
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Cache Handling
    cache_loaded = False
    if os.path.exists(master_cache_path):
        try:
            logger.info(f"Loading master cache index from {master_cache_path}")
            with open(master_cache_path, "r", encoding="utf-8") as f:
                master_cache = json.load(f)
                
            summary_result = master_cache.get("summary", "")
            chunk_files = master_cache.get("chunk_files", [])
            
            # Load all chunk files
            raw_cached_snippets = []
            for chunk_file in chunk_files:
                chunk_path = os.path.join(chunks_dir, chunk_file)
                if os.path.exists(chunk_path):
                    try:
                        with open(chunk_path, "r", encoding="utf-8") as f:
                            chunk_data = json.load(f)
                            chunk_evidence = [snippet["verbatim_text"] for snippet in chunk_data.get("evidence", [])]
                            raw_cached_snippets.extend(chunk_evidence)
                    except Exception as e:
                        logger.warning(f"Error loading chunk file {chunk_path}: {str(e)}")
            
            # Process the snippets
            temp_processed_segments = []
            last_page_annot = None
            for snippet_text in raw_cached_snippets:
                possible_page = extract_page_number_from_text(snippet_text)
                if not possible_page:
                    possible_page = last_page_annot
                else:
                    last_page_annot = possible_page
                    
                processed = process_segment(snippet_text, possible_page, page_to_blocks, min_snippet_word_count)
                if processed:
                    temp_processed_segments.append(processed)

            processed_segments = temp_processed_segments

            if summary_result or processed_segments:
                logger.info(f"Cache loaded and validated: Summary: {summary_result is not None}, Snippets: {len(processed_segments)}")
                cache_loaded = True
            else:
                logger.warning("Cache files loaded but resulted in no valid data after processing. Will regenerate.")

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from master cache file {master_cache_path}")
        except Exception as e: 
            logger.error(f"Error loading cache files: {str(e)}")

    if not cache_loaded:
        logger.info("Generating new summary and evidence points using LLM")

        prompt_template_str = SUMMARY_EVIDENCE_PROMPT_STRUCTURED 
        if not prompt_template_str:
            logger.error("Summary/evidence prompt template is empty")
            return None, []

        try:
            response_object = SummaryAndEvidence(summary="", evidence=[])
            chunk_files = []
            all_text_for_llm = ""
            for chunk in chunked_data:
                chunked_text = chunk.get("text", "")
                if not chunked_text:
                    continue
                all_text_for_llm += " " + chunked_text.strip()
            is_one_p = len(all_text_for_llm.split(' ')) <= 8192
            if is_one_p:
                # Process all chunks in a single request
                if not all_text_for_llm.strip():
                    logger.error("No text content found in chunked_data")
                    return None, []
                
                final_prompt = prompt_template_str + "\n\nDocument Content:\n" + all_text_for_llm.strip()
                
                try:
                    structured_llm = llm.with_structured_output(output_schema)
                    logger.info("LLM configured for structured output")
                except (AttributeError, Exception) as e:
                    logger.error(f"Error configuring LLM for structured output: {str(e)}")
                    return None, []

                # Invoke LLM
                logger.info("Sending request to LLM for structured response")
                response: SummaryAndEvidence = structured_llm.invoke(final_prompt)
                logger.info("Structured response received from LLM")
                
                response_object.summary = response.summary.strip()
                response_object.evidence.extend(response.evidence)
                
                # Save the single chunk
                chunk_filename = f"chunk_all.json"
                chunk_path = os.path.join(chunks_dir, chunk_filename)
                try:
                    with open(chunk_path, "w", encoding="utf-8") as f:
                        json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)
                    chunk_files.append(chunk_filename)
                    logger.info(f"Saved chunk to {chunk_path}")
                except Exception as e:
                    logger.error(f"Error saving chunk file {chunk_path}: {str(e)}")
                
            else:
                # Process chunks individually
                for chunk_idx, chunk in enumerate(chunked_data):
                    chunked_text = chunk.get("text", "").strip()
                    if not chunked_text:
                        logger.warning(f"Empty chunk {chunk_idx + 1}/{len(chunked_data)}")
                        continue

                    # Prepare the prompt for this chunk
                    final_prompt = prompt_template_str + "\n\nDocument Content:\n" + chunked_text
                    if response_object.summary:
                        final_prompt += "\n\nSummary of previous chunks:\n" + response_object.summary.strip()
                    
                    try:
                        structured_llm = llm.with_structured_output(output_schema)
                    except Exception as e:
                        logger.error(f"Error configuring LLM for chunk {chunk_idx + 1}: {str(e)}")
                        continue

                    logger.info(f"Processing chunk {chunk_idx + 1}/{len(chunked_data)}")
                    response: SummaryAndEvidence = structured_llm.invoke(final_prompt)

                    # Merge the response into the main response_object
                    response_object.summary += response.summary + "\n" 
                    response_object.evidence.extend(response.evidence)
                    
                    # Save this chunk to its own file
                    chunk_filename = f"chunk_{chunk_idx:03d}.json"
                    chunk_path = os.path.join(chunks_dir, chunk_filename)
                    try:
                        with open(chunk_path, "w", encoding="utf-8") as f:
                            json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)
                        chunk_files.append(chunk_filename)
                        logger.info(f"Saved chunk {chunk_idx + 1} to {chunk_path}")
                    except Exception as e:
                        logger.error(f"Error saving chunk file {chunk_path}: {str(e)}")
                    
            # Clean up the summary
            summary_result = response_object.summary.strip()
            raw_snippet_objects = response_object.evidence
            raw_snippets_text = [snippet.verbatim_text for snippet in raw_snippet_objects]

            logger.info(f"Extracted Summary: {summary_result is not None}. Extracted Raw Snippets: {len(raw_snippets_text)}")
            processed_segments = []
            
            last_page_annot = None
            for snippet_text in raw_snippets_text:
                possible_page = extract_page_number_from_text(snippet_text)
                if not possible_page:
                    possible_page = last_page_annot
                else:
                    last_page_annot = possible_page
                    
                processed_section = process_segment(snippet_text, possible_page, page_to_blocks, min_snippet_word_count)
                
                if processed_section:
                    processed_segments.append(processed_section)

            # Create a more concise summary using Gemini
            concise_summary = summarize_text(summary_result, llm) if summary_result else ""
            
            # Save master cache file
            try:
                master_cache = {
                    "summary": summary_result,
                    "concise_summary": concise_summary,
                    "chunk_files": chunk_files,
                    "total_evidence_count": len(raw_snippets_text),
                    "timestamp": str(datetime.datetime.now())
                }
                
                with open(master_cache_path, "w", encoding="utf-8") as f:
                    json.dump(master_cache, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved master cache index to {master_cache_path}")
            except Exception as e:
                logger.error(f"Error saving master cache file {master_cache_path}: {str(e)}")

        except Exception as e:
            logger.error(f"Error during LLM interaction: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            summary_result = None
            processed_segments = []

    logger.info(f"Final results: Summary: {'Yes' if summary_result else 'No'}, Processed Snippets: {len(processed_segments)}")
    return summary_result, processed_segments


def highlight_interesting_points(pdf_path: str, interesting_points: List[Dict[str, Any]], output_path: str, highlight_color: Tuple[float, float, float] = HIGHLIGHT_COLOR) -> str:
    """
    Add highlights to the interesting points in the PDF.
    
    Args:
        pdf_path: Path to the PDF file
        interesting_points: List of points to highlight
        output_path: Path to the output highlighted PDF
        highlight_color: RGB color tuple for highlights
        
    Returns:
        Path to the highlighted PDF
    """
    try:
        logger.info(f"Highlighting PDF: {pdf_path} -> {output_path}")
        doc = pymupdf.open(pdf_path)
        
        fail_count = 0
        success_count = 0
        
        for point in interesting_points:
            page_num = point["page"]
            if page_num >= len(doc):
                logger.warning(f"Page {page_num} out of range (document has {len(doc)} pages)")
                continue
                
            page = doc[page_num]
            text = point["text"]
            if not text:
                logger.warning(f"Empty text for page {page_num}")
                continue
                
            # Extract content without page annotations for better matching
            clean_text = re.sub(r'\[Page:\s*\d+\]\s*', '', text).strip()
            if not clean_text:
                continue
                
            # Try exact match first
            text_instances = page.search_for(clean_text)
            
            found_matches = []
            if not text_instances:
                # Try with normalized text
                normalized_text = ' '.join(clean_text.split())
                text_instances = page.search_for(normalized_text)
                
                # Try with key phrases if text is long enough
                if len(normalized_text.split()) > 10:
                    # Extract significant phrases (5-8 words)
                    words = normalized_text.split()
                    for i in range(len(words) - 5):
                        phrase = ' '.join(words[i:i+min(8, len(words)-i)])
                        if len(phrase) > 15:  # Only phrases with enough content
                            phrase_instances = page.search_for(phrase)
                            if phrase_instances:
                                found_matches.extend(phrase_instances)
                
                # Try with sentences if available
                if '.' in normalized_text:
                    sentences = [s.strip() for s in normalized_text.split('.') if len(s.strip()) > 15]
                    for sentence in sentences:
                        sentence_instances = page.search_for(sentence)
                        if sentence_instances:
                            found_matches.extend(sentence_instances)
            
            # If we found partial matches, combine their bounding boxes
            if found_matches and not text_instances:
                text_instances = found_matches
            
            # Highlight found instances or use bbox as fallback
            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=highlight_color) 
                    highlight.update()
                success_count += 1
            elif point.get("bbox_list") and isinstance(point["bbox_list"], list) and point["bbox_list"]:
                for bbox in point["bbox_list"]:
                    if bbox: 
                        r = page.add_highlight_annot(bbox)
                        r.set_colors(stroke=highlight_color)
                        r.update()
                success_count += 1
            elif point.get("bbox"):
                # Single bbox case
                r = page.add_highlight_annot(point["bbox"])
                r.set_colors(stroke=highlight_color)    
                r.update()
                success_count += 1
            else:
                fail_count += 1
        
        logger.info(f"Successfully highlighted {success_count} segments")
        logger.info(f"Failed to highlight {fail_count} segments")
        
        # Save the highlighted PDF
        doc.save(output_path)
        doc.close()
        
        logger.info(f"Created highlighted PDF: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error highlighting PDF: {str(e)}")
        raise HighlightingError(f"Failed to highlight PDF: {str(e)}")


def save_summary_to_file(summary: str, output_path: str) -> str:
    """
    Save the generated summary to a text file.
    
    Args:
        summary: The summary text
        output_path: Path to save the summary
        
    Returns:
        Path to the saved summary file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info(f"Saved summary to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving summary to file: {str(e)}")
        raise IOError(f"Failed to save summary to file: {str(e)}")


def save_evidence_to_file(evidence: List[Dict[str, Any]], output_path: str) -> str:
    """
    Save the evidence points to a text file.
    
    Args:
        evidence: List of evidence points
        output_path: Path to save the evidence
        
    Returns:
        Path to the saved evidence file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for i, point in enumerate(evidence, 1):
                f.write(f"{i}. {point['text']}\n\n")
        logger.info(f"Saved evidence to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error saving evidence to file: {str(e)}")
        raise IOError(f"Failed to save evidence to file: {str(e)}")