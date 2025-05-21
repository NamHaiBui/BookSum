import re
import logging
import os
from typing import List, Tuple, Dict, Any, Optional, Pattern

logger = logging.getLogger("AudioBookPipeline.SRTProcessor")

class SRTItem:
    """Class representing a single SRT subtitle item with timestamp-prefixed text."""
    def __init__(self, index: int, text: str):
        self.index = index
        self.text = text
        
    def __repr__(self) -> str:
        return f"SRTItem(index={self.index}, text={self.text[:30]}...)" if len(self.text) > 30 else f"SRTItem(index={self.index}, text={self.text})"
    
    def __str__(self) -> str:
        return f"{self.index}\n{self.text}"

class ChunkedSRTItem:
    """Class representing a chunk of SRT items grouped by chapter."""
    def __init__(self, start_index: int, end_index: int, chunk: List[SRTItem] = None):
        self.start_index = start_index
        self.end_index = end_index
        self.chunk = chunk if chunk else []

    def __repr__(self) -> str:
        return f"ChunkedSRTItem(start={self.start_index}, end={self.end_index}, items={len(self.chunk)})"

def generate_chapter_patterns(config: Dict[str, Any]) -> Tuple[Pattern, Pattern]:
    """
    Generate compiled regex patterns for chapter detection.
    
    Args:
        config: Configuration dictionary with pattern strings.
        
    Returns:
        Tuple of (text_pattern, int_pattern) compiled regex patterns.
    """
    try:
        # Get pattern strings from config
        chapter_pattern_int_rg = config["srt_processing"]["chapter_pattern_int_rg"]
        
        # Define number words for text-based pattern (or read from config if available)
        num_words = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
            "hundred", "thousand"
        ]
        
        # Construct the text-based pattern
        chapter_pattern_text_based_rg = r'Chapter\s+(?:' + '|'.join(num_words) + r')\b[\s:.]*'
        
        # Compile patterns
        chapter_pattern_text = re.compile(chapter_pattern_text_based_rg, re.IGNORECASE)
        chapter_pattern_int = re.compile(chapter_pattern_int_rg, re.IGNORECASE)
        
        return chapter_pattern_text, chapter_pattern_int
        
    except Exception as e:
        logger.error(f"Error generating chapter patterns: {e}")
        # Return basic patterns as fallback
        return re.compile(r"Chapter\s+[a-z]+\b", re.IGNORECASE), re.compile(r"Chapter\s+\d+\b", re.IGNORECASE)

def process_srt(srt_file: str) -> Tuple[List[SRTItem], List[SRTItem]]:
    """
    Processes an SRT file to create two versions:
    1. A list with timestamp-prefixed SRTItems (for chunking)
    2. The original SRTItems (for matching later)
    
    Args:
        srt_file: Path to the SRT file.
        
    Returns:
        Tuple of (timestamp_prefixed_items, original_items)
    """
    try:
        logger.info(f"Processing SRT file: {srt_file}")
        
        # Read SRT content
        with open(srt_file, 'r', encoding='utf-8') as file:
            content = file.read()
            
        if not content.strip():
            logger.warning("SRT file is empty")
            return [], []
            
        # Split into subtitles
        subtitles = [block.strip() for block in re.split(r'\n\s*\n', content.strip())]
        logger.info(f"Found {len(subtitles)} subtitle blocks in SRT")
        
        # Process blocks
        processed_items = []
        original_items = []
        processed_indices = set()
        
        for block in subtitles:
            lines = block.splitlines()
            # Handle invalid blocks
            if len(lines) < 3:
                logger.warning(f"Skipping invalid subtitle block (less than 3 lines): {block[:50]}...")
                continue
                
            try:
                # Extract index
                index_str = lines[0].strip()
                index = int(index_str)
                
                # Skip duplicates
                if index in processed_indices:
                    logger.warning(f"Skipping duplicate subtitle index: {index}")
                    continue
                    
                processed_indices.add(index)
                
                # Extract timestamp
                timestamp_line = lines[1].strip()
                match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
                
                if not match:
                    logger.warning(f"Invalid timestamp format in block {index}: {timestamp_line}")
                    continue
                    
                start_timestamp = match.group(1)
                end_timestamp = match.group(2)
                
                # Extract text lines
                text_lines = lines[2:]
                if not text_lines:
                    logger.warning(f"No text content in block {index}")
                    continue
                
                # Create original item (without timestamp prefix)
                original_text = "\n".join(text_lines)
                original_items.append(SRTItem(index, original_text))
                
                # Create timestamp-prefixed item
                modified_first_text_line = f"[{start_timestamp},{end_timestamp}] {text_lines[0]}"
                modified_text_lines = [modified_first_text_line] + text_lines[1:]
                modified_text = "\n".join(modified_text_lines)
                processed_items.append(SRTItem(index, modified_text))
                
            except (ValueError, IndexError) as e:
                logger.warning(f"Error processing subtitle block: {e}")
                continue
                
        logger.info(f"Successfully processed {len(processed_items)} SRT items")
        return processed_items, original_items
        
    except Exception as e:
        logger.error(f"Error processing SRT file: {e}")
        return [], []

def _get_text_after_timestamp(text: str, timestamp_pattern: Pattern) -> str:
    """
    Extracts text content after the initial timestamp pattern.
    
    Args:
        text: Text with potential timestamp prefix.
        timestamp_pattern: Compiled regex pattern to match timestamps.
        
    Returns:
        Text without the timestamp prefix.
    """
    if not text or not timestamp_pattern:
        return text.strip() if text else ""
        
    match = timestamp_pattern.match(text)
    return text[match.end():].strip() if match else text.strip()

def chunking_srt_by_chapter(
    srt_items: List[SRTItem],
    chapter_pattern_text: Pattern,
    chapter_pattern_int: Pattern,
    timestamp_pattern: Pattern
) -> List[ChunkedSRTItem]:
    """
    Chunks a list of SRTItem objects based on chapter markers found in the text.
    
    Args:
        srt_items: List of SRTItem objects from process_srt.
        chapter_pattern_text: Compiled regex for text-based chapter markers.
        chapter_pattern_int: Compiled regex for integer-based chapter markers.
        timestamp_pattern: Compiled regex for the timestamp pattern.
        
    Returns:
        A list of ChunkedSRTItem objects.
    """
    if not srt_items:
        logger.warning("No SRT items provided for chunking")
        return []

    # Validate patterns
    if not all(isinstance(p, Pattern) for p in [chapter_pattern_text, chapter_pattern_int, timestamp_pattern]):
        logger.error("Invalid regex patterns provided")
        return []

    chunk_data_by_chapter = []
    current_chunk_items = []

    logger.info("Starting SRT chunking by chapter")
    
    for item in srt_items:
        # Validate item
        if not isinstance(item, SRTItem) or not hasattr(item, 'text') or not hasattr(item, 'index'):
            logger.warning(f"Skipping invalid SRT item: {item}")
            continue
            
        if not isinstance(item.text, str):
            logger.warning(f"Skipping item with non-string text (Index: {item.index})")
            continue

        # Extract text after timestamp
        text_after_ts = _get_text_after_timestamp(item.text, timestamp_pattern)

        # Check if it's a chapter start
        is_chapter_start = False
        if text_after_ts:
            is_chapter_start = bool(
                chapter_pattern_text.search(text_after_ts) or 
                chapter_pattern_int.search(text_after_ts)
            )

        if is_chapter_start and current_chunk_items:
            # End the current chunk and start a new one
            chunk_data_by_chapter.append(ChunkedSRTItem(
                start_index=current_chunk_items[0].index,
                end_index=current_chunk_items[-1].index,
                chunk=current_chunk_items
            ))
            # Start a new chunk with this chapter item
            current_chunk_items = [item]
        else:
            # Add to the current chunk
            current_chunk_items.append(item)

    # Add the last chunk if not empty
    if current_chunk_items:
        chunk_data_by_chapter.append(ChunkedSRTItem(
            start_index=current_chunk_items[0].index,
            end_index=current_chunk_items[-1].index,
            chunk=current_chunk_items
        ))

    logger.info(f"Chunking complete. Generated {len(chunk_data_by_chapter)} chapter-based chunks.")
    return chunk_data_by_chapter