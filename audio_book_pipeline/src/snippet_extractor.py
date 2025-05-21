import logging
import os
import re
import traceback
from typing import List, Dict, Any, Tuple, Optional
from pydub import AudioSegment

logger = logging.getLogger("AudioBookPipeline.SnippetExtractor")

def srt_timestamp_to_ms(timestamp_str: str) -> int:
    """
    Converts an SRT timestamp string (HH:MM:SS,ms) to milliseconds.
    
    Args:
        timestamp_str: Timestamp string in SRT format (HH:MM:SS,ms)
        
    Returns:
        Timestamp in milliseconds
    """
    try:
        h, m, s_ms = timestamp_str.split(':')
        s, ms = s_ms.split(',')
        # Calculate total milliseconds
        total_ms = int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
        return total_ms
    except ValueError as e:
        logger.error(f"Error converting timestamp '{timestamp_str}': {e}")
        raise

def save_merged_snippet(
    interval_ms: Tuple[int, int], 
    text: str, 
    audio_segment: AudioSegment, 
    output_dir: str, 
    snippet_index: int,
    output_format: str,
    end_time_buffer_ms: int
) -> bool:
    """
    Saves a merged audio snippet and its corresponding transcript.
    
    Args:
        interval_ms: Tuple of (start_ms, end_ms) timestamps.
        text: Text content of the snippet.
        audio_segment: The full audio recording.
        output_dir: Directory to save the snippet.
        snippet_index: Index of the snippet for file naming.
        output_format: Format to save audio (e.g., 'mp3').
        end_time_buffer_ms: Buffer to add to end time when cutting audio.
        
    Returns:
        True if successful, False otherwise.
    """
    start_ms, end_ms = interval_ms
    # Add buffer to end time for audio cutting
    buffered_end_ms = end_ms + end_time_buffer_ms
    
    logger.info(f"Saving merged snippet {snippet_index}: {text[:50]}... from {start_ms}ms to {end_ms}ms")
    
    try:
        # Cut the audio
        snippet = audio_segment[start_ms:buffered_end_ms]
        
        # Create filenames
        clean_text_preview = re.sub(r'[\\/*?:"<>|,.]', '', text)[:30]
        safe_filename_part = re.sub(r'\s+', '_', clean_text_preview)
        base_filename = f"snippet_{snippet_index:03d}_{start_ms}_{safe_filename_part}"
        audio_output_filename = os.path.join(output_dir, f"{base_filename}.{output_format}")
        text_output_filename = os.path.join(output_dir, f"{base_filename}.txt")
        
        # Save audio
        snippet.export(audio_output_filename, format=output_format)
        logger.info(f"Audio saved to: {audio_output_filename}")
        
        # Save transcript
        with open(text_output_filename, 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info(f"Transcript saved to: {text_output_filename}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving merged snippet: {e}")
        logger.error(traceback.format_exc())
        return False

def merge_and_extract_snippets(
    evidence_list: List[Dict[str, Any]], 
    audio_file_path: str, 
    output_dir: str, 
    config: Dict[str, Any]
) -> Tuple[int, int]:
    """
    Processes a list of evidence snippets to merge consecutive ones and extract audio.
    
    Args:
        evidence_list: List of evidence dictionaries from LLM processing.
        audio_file_path: Path to the audio file.
        output_dir: Directory to save snippets.
        config: Configuration dictionary.
        
    Returns:
        Tuple of (total_snippets_processed, total_snippets_saved)
    """
    logger.info(f"Starting to merge and extract snippets from {len(evidence_list)} evidence items")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration parameters
    merge_proximity_ms = config["snippet_extraction"]["merge_proximity_ms"]
    output_format = config["snippet_extraction"]["output_format"]
    end_time_buffer_ms = config["snippet_extraction"]["end_time_buffer_ms"]
    
    # Initialize counters
    total_snippets_processed = 0
    total_snippets_saved = 0
    
    # Sort evidence by timestamp
    try:
        sorted_evidence = sorted(
            evidence_list, 
            key=lambda x: srt_timestamp_to_ms(x["timestamp"][0]) if "timestamp" in x else 0
        )
    except Exception as e:
        logger.error(f"Error sorting evidence: {e}")
        return 0, 0
    
    # Load audio file
    try:
        logger.info(f"Loading audio file: {audio_file_path}")
        audio = AudioSegment.from_file(audio_file_path)
        logger.info("Audio file loaded successfully")
    except FileNotFoundError:
        logger.error(f"Audio file not found: {audio_file_path}")
        return 0, 0
    except Exception as e:
        logger.error(f"Error loading audio file: {e}")
        logger.error("Ensure ffmpeg is installed and in your system's PATH")
        logger.error(traceback.format_exc())
        return 0, 0
    
    # Process evidence, merge consecutive, and cut snippets
    current_merged_interval = None
    current_merged_text = ""
    
    for i, evidence in enumerate(sorted_evidence):
        total_snippets_processed += 1
        
        try:
            # Extract timestamp and text
            if "timestamp" not in evidence or "text" not in evidence:
                logger.warning(f"Skipping evidence item {i+1}: Missing timestamp or text")
                continue
                
            start_ts_str, end_ts_str = evidence["timestamp"]
            text_content = evidence["text"]
            
            # Convert to milliseconds
            start_ms = srt_timestamp_to_ms(start_ts_str)
            end_ms = srt_timestamp_to_ms(end_ts_str)
            
            # Basic validation
            if start_ms >= end_ms:
                logger.warning(f"Skipping snippet {i+1}: Start time ({start_ms}ms) is not before end time ({end_ms}ms)")
                continue
            
            # Merging logic
            if current_merged_interval is None:
                # Start a new merged interval
                current_merged_interval = (start_ms, end_ms)
                current_merged_text = text_content
            elif start_ms - current_merged_interval[1] <= merge_proximity_ms:
                # Merge with previous interval
                current_merged_interval = (
                    current_merged_interval[0], 
                    max(current_merged_interval[1], end_ms)
                )
                current_merged_text += " " + text_content
            else:
                # Gap detected: Save previous merged interval
                if save_merged_snippet(
                    current_merged_interval, 
                    current_merged_text, 
                    audio, 
                    output_dir, 
                    total_snippets_saved + 1,
                    output_format,
                    end_time_buffer_ms
                ):
                    total_snippets_saved += 1
                
                # Start a new merged interval
                current_merged_interval = (start_ms, end_ms)
                current_merged_text = text_content
                
        except ValueError as e:
            logger.error(f"Error processing snippet {i+1}: {e}")
            # Reset current merge state to avoid cascading issues
            current_merged_interval = None
            current_merged_text = ""
            continue
        except Exception as e:
            logger.error(f"Unexpected error processing snippet {i+1}: {e}")
            logger.error(traceback.format_exc())
            current_merged_interval = None
            current_merged_text = ""
            continue
    
    # Save the last merged interval if it exists
    if current_merged_interval is not None:
        if save_merged_snippet(
            current_merged_interval, 
            current_merged_text, 
            audio, 
            output_dir, 
            total_snippets_saved + 1,
            output_format,
            end_time_buffer_ms
        ):
            total_snippets_saved += 1
    
    logger.info(f"Snippet extraction complete. Processed {total_snippets_processed} items, saved {total_snippets_saved} merged snippets.")
    return total_snippets_processed, total_snippets_saved