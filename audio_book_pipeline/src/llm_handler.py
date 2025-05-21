import datetime
import json
import os
import re
import difflib
import logging
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional, Type, DefaultDict

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

from src.srt_processor import SRTItem, ChunkedSRTItem

logger = logging.getLogger("AudioBookPipeline.LLMHandler")

class EvidenceSnippet(BaseModel):
    """Represents a single verbatim evidence snippet."""
    verbatim_text: str = Field(..., description="The exact verbatim text snippet extracted from the document, including any '[]' annotation if present at the beginning of the snippet.")

class SummaryAndEvidence(BaseModel):
    """Schema for the desired LLM output containing summary and evidence."""
    summary: str = Field(..., description="A concise summary based solely on the provided document content.")
    evidence: List[EvidenceSnippet] = Field(..., description="A list of verbatim evidence snippets supporting the key points of the summary.")

def initialize_llm(config: Dict[str, Any]) -> Optional[ChatGoogleGenerativeAI]:
    """
    Initializes the LLM client based on configuration.
    
    Args:
        config: Configuration dictionary with LLM settings.
        
    Returns:
        Initialized LLM client or None if initialization fails.
    """
    try:
        model_name = config["llm"]["model_name"]
        temperature = config["llm"]["temperature"]
        
        logger.info(f"Initializing LLM with model: {model_name}, temperature: {temperature}")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )
        logger.info("LLM initialized successfully")
        return llm
        
    except Exception as e:
        logger.error(f"Error initializing LLM: {e}")
        logger.error("Ensure GOOGLE_API_KEY is set in your environment and the model name is correct.")
        return None

def proper_extract(raw_snippet_text: str, timestamp_pattern) -> Tuple[str, Optional[Tuple[str, str]]]:
    """
    Cleans the raw snippet text from the LLM.
    Removes the leading '[start,end]' annotation if present.
    
    Args:
        raw_snippet_text: Text with potential timestamp prefix.
        timestamp_pattern: Compiled regex pattern for timestamp extraction.
        
    Returns:
        Tuple of (cleaned_text, timestamp_tuple or None).
    """
    match = timestamp_pattern.match(raw_snippet_text)
    if match:
        # Extract timestamp tuple ('start', 'end')
        extracted_ts = (match.group(1), match.group(2))
        cleaned_text = raw_snippet_text[match.end():].strip()
        return cleaned_text, extracted_ts
    else:
        return raw_snippet_text.strip(), None

def _find_matching_block(
    segment_text: str,
    original_material: List[SRTItem],
    min_similarity_threshold: float = 0.4
) -> Tuple[Optional[SRTItem], float]:
    """
    Finds the best matching text block in the original SRT data for a given segment text.
    
    Args:
        segment_text: The text segment to match.
        original_material: List of original SRTItem objects.
        min_similarity_threshold: Minimum similarity score to consider a match.
        
    Returns:
        Tuple of (best_match_block, similarity_score) or (None, 0.0) if no match is found.
    """
    # Normalize the segment text
    norm_segment = ' '.join(segment_text.split())
    if not norm_segment:
        return None, 0.0

    best_match_block: Optional[SRTItem] = None
    highest_similarity: float = 0.0

    for block in original_material:
        # Ensure the block has text
        block_text = block.text if isinstance(block.text, str) else ""
        if not block_text:
            continue
            
        # Extract text without timestamp if present
        block_text, _ = proper_extract(block_text, timestamp_pattern)
        norm_block_text = ' '.join(block_text.split())
        if not norm_block_text:
            continue
        
        # Calculate similarity using sequence matcher
        matcher = difflib.SequenceMatcher(None, norm_segment, norm_block_text, autojunk=False)
        similarity = matcher.ratio()
        
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_block = block
        
        # If perfect match, exit early
        if similarity >= 0.95:
            break
            
    if best_match_block and highest_similarity >= min_similarity_threshold:
        return best_match_block, highest_similarity

    return None, 0.0

def _process_segment(
    raw_snippet_text: str,
    index_hint: Optional[Tuple[int, int]],
    original_material: List[SRTItem],
    min_word_threshold: int,
    timestamp_pattern,
    similarity_threshold: float
) -> Optional[Dict[str, Any]]:
    """
    Processes a raw evidence snippet text provided by the LLM.
    
    Args:
        raw_snippet_text: The raw evidence text from LLM.
        index_hint: Optional hint for index range to search within.
        original_material: List of original SRTItems.
        min_word_threshold: Minimum word count for valid snippets.
        timestamp_pattern: Compiled regex pattern for timestamp extraction.
        similarity_threshold: Threshold for similarity matching.
        
    Returns:
        Dictionary with matching information or None if no match found.
    """
    # Clean the snippet text
    cleaned_text, _ = proper_extract(raw_snippet_text, timestamp_pattern)

    # Skip if too short
    if not cleaned_text or len(cleaned_text.split()) < min_word_threshold:
        logger.debug(f"Skipping too short snippet: '{cleaned_text}'")
        return None
        
    # Define search scope
    search_material = original_material
    if index_hint and index_hint[0] > 0 and index_hint[1] > index_hint[0]:
        # Find items in the hint range
        search_material = [item for item in original_material 
                           if item.index >= index_hint[0] and item.index <= index_hint[1]]
    
    # Find matching block
    matched_block, match_similarity = _find_matching_block(
        cleaned_text, search_material, similarity_threshold
    )

    if matched_block:
        # Extract timestamp from the matched block
        actual_timestamp_tuple = extract_timestamp_from_text(matched_block.text, timestamp_pattern)
        
        if not actual_timestamp_tuple:
            logger.warning(f"Matched block (Index: {matched_block.index}) has missing or invalid timestamp")
            return None

        return {
            "timestamp": actual_timestamp_tuple,
            "text": cleaned_text,
            "match_score": match_similarity,
            "matched_index": matched_block.index
        }
    else:
        logger.warning(f"No matching block found for snippet: '{raw_snippet_text[:50]}...'")
        return None

def extract_timestamp_from_text(text: str, timestamp_pattern) -> Optional[Tuple[str, str]]:
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

def identify_interesting_points_structured(
    chunked_data: List[ChunkedSRTItem],
    llm: ChatGoogleGenerativeAI,
    output_schema: Type[BaseModel],
    original_data: List[SRTItem],
    config: Dict[str, Any],
    cache_file_path: str,
    timestamp_pattern
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Identifies summary and evidence snippets using an LLM with structured output.
    
    Args:
        chunked_data: List of ChunkedSRTItem objects.
        llm: Initialized LLM client.
        output_schema: Pydantic model class for structured output.
        original_data: List of original SRTItem objects.
        config: Configuration dictionary.
        cache_file_path: Path to cache file.
        timestamp_pattern: Compiled regex pattern for timestamp extraction.
        
    Returns:
        Tuple of (summary_text, evidence_list).
    """
    logger.info("Starting identify_interesting_points_structured")
    
    # Initialize return values
    summary_result: Optional[str] = None
    processed_segments: List[Dict[str, Any]] = []

    # Validate inputs
    if not chunked_data:
        logger.warning("No chunked data provided")
        return None, []

    if not isinstance(original_data, list) or not original_data or not isinstance(original_data[0], SRTItem):
        logger.error("Invalid original_data format")
        return None, []
    
    # Setup cache paths
    cache_dir = os.path.dirname(cache_file_path)
    cache_base_name = os.path.basename(cache_file_path).split('.')[0]
    chunks_dir = os.path.join(cache_dir, f"{cache_base_name}_chunks")
    master_cache_path = os.path.join(cache_dir, f"{cache_base_name}_master.json")
    
    # Create cache directories
    os.makedirs(chunks_dir, exist_ok=True)
    
    # Check for existing cache
    cache_loaded = False
    if os.path.exists(master_cache_path):
        logger.info(f"Found master cache at {master_cache_path}, attempting to load")
        try:
            with open(master_cache_path, "r", encoding="utf-8") as f:
                master_cache = json.load(f)
                
            summary_result = master_cache.get("summary", "")
            chunk_files = master_cache.get("chunk_files", [])
            
            # Load evidence from chunk files
            raw_cached_snippets = {}
            for chunk_file in chunk_files:
                chunk_path = os.path.join(chunks_dir, chunk_file)
                if os.path.exists(chunk_path):
                    try:
                        with open(chunk_path, "r", encoding="utf-8") as f:
                            chunk_data = json.load(f)
                            chunk_evidence = [
                                snippet["verbatim_text"] 
                                for snippet in chunk_data.get("evidence", [])
                            ]
                            
                            start_index = chunk_data.get("start_index", -1)
                            end_index = chunk_data.get("end_index", -1)
                            raw_cached_snippets[(start_index, end_index)] = chunk_evidence
                            
                    except Exception as e:
                        logger.warning(f"Error loading chunk file {chunk_path}: {str(e)}")
            
            # Process cached snippets to match with original material
            temp_processed_segments = []
            min_snippet_word_count = config["llm"]["min_snippet_word_count"]
            similarity_threshold = config["llm"]["similarity_threshold"]
            
            for (start_idx, end_idx), snippets in raw_cached_snippets.items():
                for snippet_text in snippets:
                    index_hint = (start_idx, end_idx) if start_idx != -1 and end_idx != -1 else None
                    processed = _process_segment(
                        snippet_text, 
                        index_hint, 
                        original_data, 
                        min_snippet_word_count,
                        timestamp_pattern,
                        similarity_threshold
                    )
                    if processed:
                        temp_processed_segments.append(processed)

            processed_segments = temp_processed_segments
            
            if summary_result or processed_segments:
                logger.info(f"Cache loaded successfully: Summary: {bool(summary_result)}, Evidence: {len(processed_segments)}")
                cache_loaded = True
            else:
                logger.warning("Cache loaded but no valid data found")
                
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from master cache file {master_cache_path}")
        except Exception as e:
            logger.error(f"Error loading cache: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Generate content using LLM if cache not loaded
    if not cache_loaded:
        logger.info("Generating new summary and evidence using LLM")
        
        # Get prompt from configuration or environment
        prompt_template_str = os.getenv("SUMMARY_EVIDENCE_PROMPT_STRUCTURED_AB")
        if not prompt_template_str:
            prompt_file = config["llm"]["prompt_file"]
            try:
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_template_str = f.read()
            except Exception as e:
                logger.error(f"Error reading prompt file: {str(e)}")
                return None, []
        
        # Determine processing mode
        process_chunks_mode = config["llm"]["process_chunks_mode"]
        max_words_for_single_prompt = config["llm"]["max_words_for_single_prompt"]
        min_snippet_word_count = config["llm"]["min_snippet_word_count"]
        similarity_threshold = config["llm"]["similarity_threshold"]
        
        # Count total words
        total_words = 0
        for chunk in chunked_data:
            for item in chunk.chunk:
                total_words += len(item.text.split())
        
        # Auto mode: decide based on total words
        if process_chunks_mode == "auto":
            is_one_prompt = total_words <= max_words_for_single_prompt
            logger.info(f"Auto mode selected. Total words: {total_words}. " +
                       f"Using {'single' if is_one_prompt else 'multi'} prompt mode.")
        else:
            is_one_prompt = process_chunks_mode == "single"
            logger.info(f"Using {'single' if is_one_prompt else 'sequential'} prompt mode as configured")
        
        chunk_files = []  # List to store chunk file names
        verbatim_result_to_chunk_mapping = defaultdict(list)
        
        try:
            response_object = SummaryAndEvidence(summary="", evidence=[])
            
            if is_one_prompt:
                # Concatenate all chunks into one prompt
                all_text_for_llm = "\n\n".join([
                    "\n".join([srt_item.text for srt_item in chunked_srt.chunk])
                    for chunked_srt in chunked_data if chunked_srt.chunk
                ]).strip()

                if not all_text_for_llm:
                    logger.error("No text content found in chunked_data")
                    return None, []

                final_prompt = prompt_template_str + "\n\nDocument Content:\n" + all_text_for_llm

                try:
                    structured_llm = llm.with_structured_output(output_schema)
                    logger.info("Sending request to LLM for structured response")
                    response = structured_llm.invoke(final_prompt)
                    logger.info("Received structured response from LLM")
                    
                    response_object.summary = response.summary.strip()
                    response_object.evidence.extend(response.evidence)
                    
                    # Save the single chunk
                    chunk_filename = f"chunk_all.json"
                    chunk_path = os.path.join(chunks_dir, chunk_filename)
                    with open(chunk_path, "w", encoding="utf-8") as f:
                        json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)
                    chunk_files.append(chunk_filename)
                    logger.info(f"Saved LLM response to {chunk_path}")
                    
                except Exception as e:
                    logger.error(f"Error in LLM structured output: {str(e)}")
                    return None, []
                    
            else:
                # Process chunks sequentially
                for chunk_idx, chunk in enumerate(chunked_data):
                    if not chunk.chunk: 
                        continue
                        
                    chunk_text = "\n".join([item.text for item in chunk.chunk]).strip()
                    if not chunk_text: 
                        continue

                    logger.info(f"Processing chunk {chunk_idx+1}/{len(chunked_data)}")
                    chunk_prompt = prompt_template_str + "\n\nDocument Content:\n" + chunk_text
                    
                    # Add current summary for context if available
                    if response_object.summary:
                        chunk_prompt += f"\n\nCurrent Summary:\n{response_object.summary.strip()}"
                    
                    try:
                        structured_llm = llm.with_structured_output(output_schema)
                        chunk_response = structured_llm.invoke(chunk_prompt)
                        
                        # Update main response
                        response_object.summary += chunk_response.summary.strip() + "\n"
                        evidence_start_idx = len(response_object.evidence)
                        evidence_end_idx = evidence_start_idx + len(chunk_response.evidence)
                        
                        # Store index mapping for this chunk's evidence
                        verbatim_result_to_chunk_mapping[(evidence_start_idx, evidence_end_idx)] = (
                            chunk.start_index, 
                            chunk.end_index
                        )
                        
                        response_object.evidence.extend(chunk_response.evidence)
                        
                        # Save chunk response
                        chunk_filename = f"chunk_{chunk_idx:03d}.json"
                        chunk_response_json = chunk_response.model_dump()
                        chunk_response_json["start_index"] = chunk.start_index
                        chunk_response_json["end_index"] = chunk.end_index
                        
                        chunk_path = os.path.join(chunks_dir, chunk_filename)
                        with open(chunk_path, "w", encoding="utf-8") as f:
                            json.dump(chunk_response_json, f, ensure_ascii=False, indent=2)
                            
                        logger.info(f"Saved chunk {chunk_idx+1} response to {chunk_path}")
                        chunk_files.append(chunk_filename)
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx+1}: {str(e)}")
                        logger.error(traceback.format_exc())
            
            # --- Process LLM responses ---
            summary_result = response_object.summary.strip()
            raw_snippet_objects = response_object.evidence
            raw_snippets_text = [snippet.verbatim_text for snippet in raw_snippet_objects]

            logger.info(f"Extracted summary and {len(raw_snippets_text)} raw evidence snippets")
            
            # Process each evidence snippet
            logger.info(f"Processing {len(raw_snippets_text)} raw snippets...")
            processed_segments = []
            
            # Chunk index tracking for sequential mode
            if not is_one_prompt:
                chunks_indices = [(chunk.start_index, chunk.end_index) for chunk in chunked_data if chunk.chunk]
                curr_chunk_idx = 0
                curr_evidence_range = next(iter(verbatim_result_to_chunk_mapping.keys())) if verbatim_result_to_chunk_mapping else (0, 0)
            
            for i, snippet_text in enumerate(raw_snippets_text):
                # Determine index hint
                index_hint = None
                if not is_one_prompt and verbatim_result_to_chunk_mapping:
                    for (start_idx, end_idx), chunk_range in verbatim_result_to_chunk_mapping.items():
                        if start_idx <= i < end_idx:
                            index_hint = chunk_range
                            break
                
                # Process the segment
                processed = _process_segment(
                    snippet_text, 
                    index_hint, 
                    original_data, 
                    min_snippet_word_count,
                    timestamp_pattern,
                    similarity_threshold
                )
                
                if processed:
                    processed_segments.append(processed)
            
            # Save master cache
            try:
                master_cache = {
                    "summary": summary_result,
                    "chunk_files": chunk_files,
                    "total_evidence_count": len(raw_snippets_text),
                    "timestamp": str(datetime.datetime.now())
                }
                
                with open(master_cache_path, "w", encoding="utf-8") as f:
                    json.dump(master_cache, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved master cache to {master_cache_path}")
                
            except Exception as e:
                logger.error(f"Error saving master cache: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error during LLM processing: {str(e)}")
            logger.error(traceback.format_exc())
            return None, []
    
    # Validate final processed segments
    valid_segments = [
        seg for seg in processed_segments
        if isinstance(seg, dict) and 
        isinstance(seg.get("timestamp"), tuple) and 
        len(seg["timestamp"]) == 2 and 
        isinstance(seg.get("text"), str)
    ]
    
    if len(valid_segments) != len(processed_segments):
        logger.warning(f"Filtered out {len(processed_segments) - len(valid_segments)} invalid segments")
        processed_segments = valid_segments
    
    logger.info(f"identify_interesting_points_structured complete. Summary: {bool(summary_result)}, Evidence: {len(processed_segments)}")
    return summary_result, processed_segments