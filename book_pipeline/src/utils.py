"""
Utility functions and classes for the book pipeline.
"""

import heapq
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, DefaultDict
import difflib
from collections import defaultdict
import pymupdf
import cv2
import numpy as np

from .constants import DUMMY_NODE_HEIGHT, PAGE_PATTERN
from .exceptions import ValidationError

# Configure logging
logger = logging.getLogger(__name__)


class MedianFinder:
    """
    Efficient data structure to find the median of a stream of numbers.
    Uses two heaps to track the median value.
    """
    def __init__(self):
        self.minheap = []  # Min heap for the larger half
        self.maxheap = []  # Max heap for the smaller half

    def addNum(self, num: float) -> None:
        """Add a number to the data structure."""
        if not self.minheap and not self.maxheap:
            heapq.heappush(self.maxheap, -num)
        else:
            if len(self.minheap) == len(self.maxheap):
                if -self.maxheap[0] <= num:
                    heapq.heappush(self.minheap, num)
                    heapq.heappush(self.maxheap, -heapq.heappop(self.minheap))
                else:
                    heapq.heappush(self.maxheap, -num)
            else:
                if -self.maxheap[0] <= num:
                    heapq.heappush(self.minheap, num)
                else:
                    heapq.heappush(self.minheap, -heapq.heappop(self.maxheap))
                    heapq.heappush(self.maxheap, -num)
                
    def findMedian(self) -> float:
        """Return the median of all numbers seen so far."""
        if len(self.minheap) == len(self.maxheap):
            return (self.minheap[0] - self.maxheap[0]) / 2.0
        else:
            return -self.maxheap[0]


class Node:
    """
    Node class for representing hierarchical document structure.
    Used to build a tree structure of the document.
    """
    def __init__(self, content):
        self.content = content
        self.children = []
        
    def add_child(self, child):
        """Add a child node to this node."""
        self.children.append(child)
        
    def __str__(self):
        return f"Node({self.content}, {len(self.children)} children)"


def should_merge(
    rect1: pymupdf.Rect,
    rect2: pymupdf.Rect,
    h_threshold: float = 8,
    v_threshold: float = 1,
    min_h_overlap_factor: float = 0.75
) -> bool:
    """
    Determine if two rectangles should be merged based on proximity or overlap.

    Args:
        rect1, rect2: PyMuPDF Rect objects to compare.
        h_threshold: Maximum horizontal gap allowed for merging adjacent rectangles
                     on the same line.
        v_threshold: Maximum vertical gap allowed for merging rectangles that
                     have significant horizontal overlap.
        min_h_overlap_factor: Minimum horizontal overlap (as a fraction of the *smaller* width)
                              required to consider vertical merging.

    Returns:
        bool: True if rectangles should be merged, False otherwise.
    """
    # Basic Properties and Checks
    if not rect1 or not rect2 or rect1.is_empty or rect2.is_empty:
        return False  # Cannot merge with invalid or empty rectangles

    height1 = rect1.height
    height2 = rect2.height
    # Avoid division by zero if heights are zero
    avg_height = (height1 + height2) / 2.0 if (height1 + height2) > 0 else 0

    # Condition 1: Horizontal Proximity on Same Line
    # Check if vertically aligned enough to be on the same 'line'
    # Use y-centers for potentially better alignment check than just y0
    y_center1 = rect1.y0 + height1 / 2.0
    y_center2 = rect2.y0 + height2 / 2.0
    # Rects are on same line if vertical distance between centers is less than avg height
    is_same_line = abs(y_center1 - y_center2) < avg_height if avg_height > 0 else (rect1.y0 == rect2.y0)

    # Calculate horizontal gap (only if they don't overlap horizontally)
    h_gap = -1.0
    if rect1.x1 <= rect2.x0:  # rect1 is left of rect2
        h_gap = rect2.x0 - rect1.x1
    elif rect2.x1 <= rect1.x0:  # rect2 is left of rect1
        h_gap = rect1.x0 - rect2.x1

    if is_same_line and h_gap >= 0 and h_gap < h_threshold:
        logger.debug(f"H-Merge: {rect1} & {rect2} (gap {h_gap:.2f})")
        return True  # Merge if close horizontally on the same line

    # Condition 2: Vertical Proximity with Significant Horizontal Overlap
    # Calculate horizontal overlap width
    h_overlap_width = max(0.0, min(rect1.x1, rect2.x1) - max(rect1.x0, rect2.x0))

    # Check if horizontal overlap is significant
    min_width = min(rect1.width, rect2.width)
    has_significant_h_overlap = False
    if min_width > 0 and h_overlap_width / min_width >= min_h_overlap_factor:
        has_significant_h_overlap = True
    elif h_overlap_width > 0 and min_width <= 0:  # Overlap exists, one rect has no width? Consider overlap significant.
        has_significant_h_overlap = True
    
    # Alternative check: One rect is contained horizontally within the other
    is_contained_horizontally = (rect1.x0 >= rect2.x0 and rect1.x1 <= rect2.x1) or \
                               (rect2.x0 >= rect1.x0 and rect2.x1 <= rect1.x1)

    if has_significant_h_overlap or is_contained_horizontally:
        # Calculate vertical gap (only if they don't overlap vertically)
        v_gap = -1.0
        if rect1.y1 <= rect2.y0:  # rect1 is above rect2
            v_gap = rect2.y0 - rect1.y1
        elif rect2.y1 <= rect1.y0:  # rect2 is above rect1
            v_gap = rect1.y0 - rect2.y1

        if v_gap >= 0 and v_gap < v_threshold:
            logger.debug(f"V-Merge: {rect1} & {rect2} (gap {v_gap:.2f})")
            return True  # Merge if close vertically and overlap significantly horizontally

    # Condition 3: Direct Overlap (Optional but good fallback)
    # Calculate vertical overlap
    v_overlap_height = max(0.0, min(rect1.y1, rect2.y1) - max(rect1.y0, rect2.y0))

    # If they overlap both horizontally and vertically, merge them.
    if h_overlap_width > 0 and v_overlap_height > 0:
        logger.debug(f"O-Merge: {rect1} & {rect2}")
        return True

    return False  # No merge conditions met


def merge_text_regions(regions: List[pymupdf.Rect], iterations: int = 3) -> List[pymupdf.Rect]:
    """
    Merge text regions that are close to each other.
    
    Args:
        regions: List of PyMuPDF Rect objects
        iterations: Number of merging passes to perform
        
    Returns:
        List of merged PyMuPDF Rect objects
    """
    if not regions:
        return []
    
    # Perform multiple iterations of merging to handle chains of regions
    for _ in range(iterations):
        merged = False
        i = 0
        while i < len(regions):
            j = i + 1
            while j < len(regions):
                if should_merge(regions[i], regions[j], h_threshold=6, v_threshold=1):
                    # Merge rectangles
                    merged_rect = pymupdf.Rect(
                        min(regions[i].x0, regions[j].x0),
                        min(regions[i].y0, regions[j].y0),
                        max(regions[i].x1, regions[j].x1),
                        max(regions[i].y1, regions[j].y1)
                    )
                    regions[i] = merged_rect
                    regions.pop(j)
                    merged = True
                else:
                    j += 1
            i += 1
            
        if not merged:
            break
            
    return regions


def extract_page_number_from_text(text: str) -> Optional[int]:
    """
    Extracts 0-indexed page number from '[Page:X]' annotation at the start.
    
    Args:
        text: Text that may start with a page annotation
        
    Returns:
        Extracted page number or None if not found
    """
    page_match = re.match(PAGE_PATTERN, text)
    if page_match:
        try:
            return int(page_match.group(1))
        except (ValueError, IndexError):
            return None
    return None


def find_matching_block(
    segment_text: str,
    page_hint: Optional[int],
    page_to_blocks: DefaultDict[int, List[Dict[str, Any]]],
    min_similarity_threshold: float = 0.6
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Finds the best matching text block for a given segment text.
    
    Args:
        segment_text: The text to match
        page_hint: Optional hint about which page to check first
        page_to_blocks: Dictionary mapping page numbers to blocks
        min_similarity_threshold: Minimum similarity to consider a match
        
    Returns:
        Tuple of (best matching block or None, similarity score)
    """
    norm_segment = ' '.join(segment_text.split())
    if not norm_segment:
        return None, 0.0

    best_match_block: Optional[Dict[str, Any]] = None
    highest_similarity: float = 0.0

    search_pages: List[int] = []
    # If page_hint is None or 0 (or less, though 0 is the first valid index), search all pages.
    if page_hint is None or page_hint < 1:
        logger.debug(f'Page hint is {page_hint}, searching all pages for: "{segment_text[:50]}..."')
        # Search pages in order, or reverse order if preferred (e.g., recent pages first)
        search_pages = list(page_to_blocks.keys())
    # If a valid page hint (>= 0) is given and exists in blocks, prioritize it.
    elif page_hint in page_to_blocks:
        logger.debug(f"Using page hint {page_hint}")
        search_pages.append(page_hint)
        # Add a padding of 2 page back and forth just in case:
        search_pages.extend(
            [page_hint + i for i in range(-2, 3) if (page_hint + i) in page_to_blocks]
        )
    # If hint is invalid or not found, search all pages
    else:
        logger.debug(f'Page hint {page_hint} not found in page_to_blocks, searching all pages')
        search_pages = list(page_to_blocks.keys())

    for page_num in search_pages:
        for block in page_to_blocks.get(page_num, []):
            block_text = block.get("norm_text", "")
            if not block_text:
                continue

            if norm_segment in block_text:
                return block, 1.0

            # Calculate similarity if not contained
            similarity = difflib.SequenceMatcher(None, norm_segment, block_text).ratio()

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_block = block
                # Optional: Add a condition to return early if similarity is very high
                if highest_similarity > 0.95:
                    logger.debug(f"High similarity ({highest_similarity:.2f}) match found on page {page_num+1}")
                    return best_match_block, highest_similarity

    if highest_similarity >= min_similarity_threshold:
        logger.debug(f"Best match found with similarity {highest_similarity:.2f}")
        return best_match_block, highest_similarity

    logger.debug(f"No suitable match found (highest similarity {highest_similarity:.2f})")
    return None, 0.0


def validate_pdf_path(pdf_path: str) -> bool:
    """
    Validate that the PDF path exists and is accessible.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        True if the PDF exists and is accessible, False otherwise
        
    Raises:
        ValidationError: If the path is invalid or the file is not accessible
    """
    import os
    
    if not pdf_path:
        raise ValidationError("PDF path cannot be empty")
    
    if not os.path.exists(pdf_path):
        raise ValidationError(f"PDF file not found at {pdf_path}")
    
    if not os.path.isfile(pdf_path):
        raise ValidationError(f"Path {pdf_path} is not a file")
    
    # Check if the file is readable
    try:
        with open(pdf_path, 'rb') as f:
            pass
    except Exception as e:
        raise ValidationError(f"PDF file is not accessible: {e}")
    
    return True