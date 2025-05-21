"""
Extract module for the book pipeline.
Handles PDF text extraction using various methods.
"""

import os
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import pymupdf
import cv2
import numpy as np

from src.classification_model import LayoutClassificationModel

from .exceptions import TextExtractionError, OCRError, PDFProcessingError
from .utils import MedianFinder, merge_text_regions
from .config import (
    PDF_ZOOM_FACTOR, 
    ADAPTIVE_THRESH_BLOCK_SIZE,
    ADAPTIVE_THRESH_C,
    MORPH_KERNEL_SIZE,
    DILATE_ITERATIONS,
    CANNY_THRESH1,
    CANNY_THRESH2,
    MIN_REGION_WIDTH,
    MIN_REGION_HEIGHT,
    MERGE_ITERATIONS,
    REGION_PADDING,
    LINE_BREAK_THRESHOLD,
    MISTRAL_API_KEY
)

# Set up logging
logger = logging.getLogger(__name__)


def extract_text_with_pymupdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    PyMuPDF-based function to extract text with bounding boxes from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text blocks with metadata
        
    Raises:
        TextExtractionError: If text extraction fails
    """
    try:
        logger.info(f"Extracting text from {pdf_path} using PyMuPDF")
        doc = pymupdf.open(pdf_path, filetype="pdf")
        prev_block = None
        all_blocks = []
        
        for page_num, page in enumerate(doc):
            words = page.get_text("words")
            page_block = []
            
            # Take threshold based on page_width and page_height
            WIDTH_threshold = (0.02 if page.rect.width > page.rect.height else 0.0092625) * page.rect.width
            HEIGHT_threshold = (0.01 if page.rect.width > page.rect.height else 0.05) * page.rect.height
            
            for curr_word in words:
                # Each block is (x0, y0, x1, y1, text, block_no, line_no, block_type)
                x0, y0, x1, y1, text, block_no, line_no, block_type = curr_word
                text_height = y1 - y0
                
                if not text.strip():
                    continue
                    
                is_mergable = False
                
                # Check if the current block is close to the previous block.
                if prev_block and abs(text_height - prev_block[-1]) <= 0 and \
                    (abs(x0 - prev_block[2]) <= WIDTH_threshold and (y0 - prev_block[3]) <= HEIGHT_threshold):
                    prev_block[2] = max(prev_block[2], x1) 
                    prev_block[3] = max(prev_block[3], y1) 
                    prev_block[4] += " " + text.strip()
                    is_mergable = True

                if is_mergable and page_block:
                    page_block.pop()
                    page_block.append(
                        {
                            "page": page_num,
                            "bbox": (prev_block[0], prev_block[1], prev_block[2], prev_block[3]),
                            "text": prev_block[4],
                            "block_no": block_no,
                            "block_type": block_type,
                            "text_height": prev_block[-1]
                        }
                    )    
                    prev_block = [prev_block[0], prev_block[1], prev_block[2], prev_block[3], prev_block[4], block_no, block_type, prev_block[-1]]
                else:
                    page_block.append({
                        "page": page_num,
                        "bbox": (x0, y0, x1, y1),
                        "text": text.strip(),
                        "block_no": block_no,
                        "block_type": block_type,
                        "text_height": text_height
                    })
                    # Update the previous block
                    prev_block = [x0, y0, x1, y1, text.strip(), block_no, block_type, text_height]
                    
            all_blocks.extend(page_block)
            
        doc.close()
        
        # Remove empty blocks
        if not all_blocks:
            raise TextExtractionError("No text found in the PDF.")
            
        logger.info(f"Extracted {len(all_blocks)} text blocks using PyMuPDF")
        return all_blocks
        
    except Exception as e:
        logger.error(f"Error extracting text with PyMuPDF: {str(e)}")
        raise TextExtractionError(f"Failed to extract text with PyMuPDF: {str(e)}")


def extract_text_with_mistral_ocr(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract text and bounding boxes from a PDF file using Mistral OCR.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text blocks with metadata
        
    Raises:
        OCRError: If OCR processing fails
    """
    try:
        logger.info(f"Extracting text from {pdf_path} using Mistral OCR")
        
        # Check for API key
        if not MISTRAL_API_KEY:
            raise OCRError("MISTRAL_API_KEY not found in environment variables")
            
        try:
            from mistralai import Mistral
            from mistralai import DocumentURLChunk
        except ImportError:
            logger.error("Mistral AI package not installed")
            raise OCRError("Required package 'mistralai' not installed. Please install it with 'pip install mistralai'")
        
        client = Mistral(api_key=MISTRAL_API_KEY)
        
        # Upload file to Mistral
        uploaded_file = client.files.upload(
            file={
                "file_name": pdf_path,
                "content": open(pdf_path, "rb"),
            },
            purpose="ocr",
        )
        
        # Get signed URL for processing
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
        
        # Process the PDF with Mistral OCR
        pdf_response = client.ocr.process(
            document=DocumentURLChunk(document_url=signed_url.url), 
            model="mistral-ocr-latest", 
            include_image_base64=True
        )
        
        # Process OCR results into the expected format
        all_blocks = []
        
        for page_idx, page in enumerate(pdf_response.pages):
            for block in page.blocks:
                # Extract text and bounding box
                text = block.text.strip()
                if not text:
                    continue
                
                # Convert bounding box to format expected by rest of code
                x0 = block.bbox.x
                y0 = block.bbox.y
                x1 = x0 + block.bbox.width
                y1 = y0 + block.bbox.height
                
                all_blocks.append({
                    "page": page_idx,
                    "bbox": (x0, y0, x1, y1),
                    "text": text,
                    "block_no": 0,  # Default block number
                    "block_type": 0  # Default block type
                })
        
        logger.info(f"Extracted {len(all_blocks)} text blocks using Mistral OCR")
        return all_blocks
        
    except Exception as e:
        logger.error(f"Error using Mistral OCR: {str(e)}")
        raise OCRError(f"Failed to extract text with Mistral OCR: {str(e)}")


def extract_page_text_via_image_regions(
    page: pymupdf.Page,
    zoom_factor: int = PDF_ZOOM_FACTOR,
    adaptive_thresh_block_size: int = ADAPTIVE_THRESH_BLOCK_SIZE,
    adaptive_thresh_c: int = ADAPTIVE_THRESH_C,
    morph_kernel_size: tuple = MORPH_KERNEL_SIZE,
    dilate_iterations: int = DILATE_ITERATIONS,
    canny_thresh1: int = CANNY_THRESH1,
    canny_thresh2: int = CANNY_THRESH2,
    min_region_width: float = MIN_REGION_WIDTH,
    min_region_height: float = MIN_REGION_HEIGHT,
    merge_iterations: int = MERGE_ITERATIONS,
    region_padding: float = REGION_PADDING,
    line_break_threshold: float = LINE_BREAK_THRESHOLD,
    filter_numeric_blocks: bool = True,
    handle_unassigned_words: bool = False,
    classification_model: LayoutClassificationModel = None,
    visual_proof: bool = False,
    visual_proof_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extracts text from a PDF page by detecting text regions using OpenCV
    on a rendered image, merging regions, and then extracting PyMuPDF text
    within those regions.
    
    Args:
        page: The pymupdf.Page object to process.
        zoom_factor: Factor to render the page image (higher zoom = more detail, slower).
        adaptive_thresh_block_size: Block size for OpenCV adaptive thresholding.
        adaptive_thresh_c: Constant subtracted in adaptive thresholding.
        morph_kernel_size: Kernel size for morphological operations.
        dilate_iterations: Number of iterations for dilation.
        canny_thresh1: Lower threshold for Canny edge detection.
        canny_thresh2: Upper threshold for Canny edge detection.
        min_region_width: Minimum width of a valid detected region (in PDF points).
        min_region_height: Minimum height of a valid detected region (in PDF points).
        merge_iterations: Iterations for the merge_text_regions function.
        region_padding: Padding added around merged regions before capturing words.
        line_break_threshold: Factor of word height used for line break detection.
        filter_numeric_blocks: If True, blocks containing only digits are discarded.
        handle_unassigned_words: If True, attempt to process words not captured.

        visual_proof: If True, display intermediate processing images.
        visual_proof_path: Path to save visual proof images (if visual_proof is True).
    Returns:
        A list of dictionaries with the extracted text regions
    """
    logger.info(f"Processing Page {page.number} via Image Regions")
    page_width_pts = page.rect.width
    page_height_pts = page.rect.height
    # 1. Render Page to Image
    try:
        mat = pymupdf.Matrix(zoom_factor, zoom_factor)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    except Exception as e:
        logger.error(f"Error rendering page {page.number} to pixmap: {e}")
        return []
        
    # 2. OpenCV Image Processing to Find Contours
    # Ensure image is grayscale (single channel) before thresholding
    if len(img.shape) == 3 and img.shape[2] >= 3:  # Check if color channels exist
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif len(img.shape) == 2:  # Already grayscale
         gray = img
    else:  # Handle unexpected image format
         logger.error(f"Cannot convert image with shape {img.shape} to grayscale")
         return []
    
    adaptive = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 
        adaptive_thresh_block_size, 
        adaptive_thresh_c
    )
    
    # Morphological operations to connect characters/words into text block shapes
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
    # Dilate to make text regions larger and connect nearby components
    dilated = cv2.dilate(adaptive, rectKernel, iterations=dilate_iterations)
    # Closing to fill small gaps within text regions
    closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, rectKernel)
    # Canny edge detection (sometimes helps refine contours)
    edged = cv2.Canny(closing, canny_thresh1, canny_thresh2, apertureSize=3)
    # Find external contours on the edge map
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    logger.debug(f"Found {len(contours)} initial contours")
    
    # 3. Create Initial Regions from Contours
    initial_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # Convert back to PDF coordinates
        pdf_x0, pdf_y0 = x / zoom_factor, y / zoom_factor
        pdf_x1, pdf_y1 = (x + w) / zoom_factor, (y + h) / zoom_factor
        rect = pymupdf.Rect(pdf_x0, pdf_y0, pdf_x1, pdf_y1)
        # Pre-filter small contours (looser threshold before merging)
        if rect.width >= min_region_width / 2 and rect.height >= min_region_height / 2:
             initial_regions.append(rect)
    logger.debug(f"Created {len(initial_regions)} initial regions after pre-filtering")
    
    # 4. Merge Overlapping/Nearby Regions
    merged_regions = merge_text_regions(initial_regions, iterations=merge_iterations)
    # Filter out regions that are too small after merging
    final_regions = [r for r in merged_regions
                     if r.width >= min_region_width and r.height >= min_region_height]
    logger.debug(f"{len(final_regions)} regions remaining after merging and size filtering")
    
    if classification_model:
        feature_order = classification_model.scaler.get_feature_names_out()
        features = {
        'num_blocks': 0.0,
        'median_width': 0.0, 'std_width': 0.0,
        'median_height': 0.0, 'std_height': 0.0,
        'median_x_center': 0.0, 'std_x_center': 0.0,
        'median_y_center': 0.0, 'std_y_center': 0.0,
        'total_width_ratio': 0.0,
        'median_aspect_ratio': 0.0,
        'norm_median_width': 0.0,
        'norm_std_x_center': 0.0,
        }
        if final_regions:
            widths = [r.width for r in final_regions]
            heights = [r.height for r in final_regions]
            x_centers = [r.x0 + r.width / 2.0 for r in final_regions]
            y_centers = [r.y0 + r.height / 2.0 for r in final_regions]
            aspect_ratios = [r.width / r.height if r.height > 0 else 0 for r in final_regions]

            features['num_blocks'] = float(len(final_regions))
            features['median_width'] = np.median(widths) if widths else 0.0
            features['std_width'] = np.std(widths) if widths else 0.0
            features['median_height'] = np.median(heights) if heights else 0.0
            features['std_height'] = np.std(heights) if heights else 0.0
            features['median_x_center'] = np.median(x_centers) if x_centers else 0.0
            features['std_x_center'] = np.std(x_centers) if x_centers else 0.0
            features['median_y_center'] = np.median(y_centers) if y_centers else 0.0
            features['std_y_center'] = np.std(y_centers) if y_centers else 0.0
            features['total_width_ratio'] = sum(widths) / page_width_pts if page_width_pts > 0 else 0.0
            features['median_aspect_ratio'] = np.median(aspect_ratios) if aspect_ratios else 0.0

            # Normalized features (using page dimensions in points)
            features['norm_median_width'] = features['median_width'] / page_width_pts if page_width_pts > 0 else 0.0
            features['norm_std_x_center'] = features['std_x_center'] / page_width_pts if page_width_pts > 0 else 0.0

        # --- 7. Create DataFrame with Correct Column Order ---
        # Ensure all expected features are present, filling with 0 if calculation failed for some reason
        ordered_features = {feat: features.get(feat, 0.0) for feat in feature_order}

        # Replace potential NaN/inf from std dev calculations on single-item lists etc.
        feature_df = pd.DataFrame([ordered_features], columns=feature_order)
        feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_df.fillna(0.0, inplace=True)
        
        
        classification_prediction = classification_model.predict(feature_df)
        layout_type = classification_prediction if isinstance(classification_prediction, str) else "single column"
        logger.info(f"  Predicted layout type: {layout_type}")

        if "multi" in layout_type.lower():
             is_multicolumn_layout = True
        else:
             is_multicolumn_layout = False
    else:
        is_multicolumn_layout = False

    
    if is_multicolumn_layout:
        logger.info("Detected multi-column layout, adjusting region processing accordingly")
        page_center_x = page.rect.width / 2.0
        center_tolerance = page.rect.width * 0.05 # 5% tolerance zone
        left_regions = []
        right_regions = []
        center_regions = []
        for region in final_regions:
            region_center_x = (region.x0 + region.x1) / 2.0
            crosses_center_zone = (region.x0 < page_center_x + center_tolerance / 2 and
                                   region.x1 > page_center_x - center_tolerance / 2)
            if crosses_center_zone and (region.width > page.rect.width * 0.6): # Heuristic for wide, centered titles
                 center_regions.append(region)
            elif region_center_x < page_center_x:
                 left_regions.append(region)
            else:
                 right_regions.append(region)
        left_regions.sort(key=lambda r: (r.y0, r.x0))
        right_regions.sort(key=lambda r: (r.y0, r.x0))
        center_regions.sort(key=lambda r: (r.y0, r.x0))
        final_regions = center_regions + left_regions + right_regions
        logger.debug(f"  Sorted into Center ({len(center_regions)}), Left ({len(left_regions)}), Right ({len(right_regions)}) regions.")
    else:
        # Sort final regions by reading order (top-to-bottom, left-to-right)
        final_regions.sort(key=lambda r: (r.y0, r.x0))
    
    # 5. Get all words from the page
    words_data = page.get_text("words")
    if not words_data:
        logger.warning(f"No text words found on page {page.number} by PyMuPDF")
        logger.info("Trying OCR fallback")
        words_data = page.get_textpage_ocr("words")
        
    if not words_data:
        return []
        
    # Prepare word items with rectangles and assignment status
    word_items = [{
        "rect": pymupdf.Rect(w[:4]),
        "text": w[4],
        "baseline": w[1],  # Using y0 of word bbox as approx baseline
        "height": pymupdf.Rect(w[:4]).height,
        "assigned": False
    } for w in words_data]
    
    # 6. Assign Words to Regions and Reconstruct Text Blocks
    extracted_blocks = []
    for region_rect in final_regions:
        # Define the padded capture area for the current region
        padded_rect = pymupdf.Rect(
            region_rect.x0 - region_padding,
            region_rect.y0 - region_padding,
            region_rect.x1 + region_padding,
            region_rect.y1 + region_padding
        )
        
        words_in_region_indices = []
        median_finder = MedianFinder()
        
        # Find unassigned words that intersect the padded region
        for idx, item in enumerate(word_items):
            if not item["assigned"] and item["rect"].intersects(padded_rect):
                words_in_region_indices.append(idx)
                if item["height"] > 0:  # Only add valid heights
                    median_finder.addNum(item["height"])
                item["assigned"] = True
                
        if not words_in_region_indices:
            continue  # Skip region if no words are found within it
            
        # Sort words within the region by reading order
        current_block_words = [word_items[i] for i in words_in_region_indices]
        current_block_words.sort(key=lambda w: (w["rect"].y0, w["rect"].x0))
        
        block_text = ""
        min_height_in_block = float('inf')
        prev_baseline = current_block_words[0]["baseline"] if current_block_words else 0
        prev_rect = current_block_words[0]["rect"] if current_block_words else None

        for word_item in current_block_words:
            current_rect = word_item["rect"]
            current_baseline = word_item["baseline"]
            current_height = word_item["height"]

            if current_height > 0:
                 min_height_in_block = min(min_height_in_block, current_height)

            # Heuristic for line breaks
            vertical_distance = abs(prev_baseline - current_baseline)
            # Estimate horizontal gap (if rects don't overlap horizontally)
            h_gap = -1
            if prev_rect and current_rect.x0 >= prev_rect.x1:
                h_gap = current_rect.x0 - prev_rect.x1

            # A small vertical distance suggests same line
            is_same_line = vertical_distance < (current_height * line_break_threshold) if current_height > 0 else (vertical_distance < 2)

            separator = " " if is_same_line else "\n"
            if block_text:  # Add separator only if text already exists
                block_text += separator
            block_text += word_item["text"]

            prev_baseline = current_baseline
            prev_rect = current_rect

        content = block_text.strip()
        if not content:
            continue
            
        if filter_numeric_blocks and content.isdigit():
            logger.debug(f"Skipping numeric block: '{content}'")
            continue
            
        # Store the extracted block information
        extracted_blocks.append({
            "occupy_space": region_rect,
            "content": content,
            "text_height_median": median_finder.findMedian() if (median_finder.minheap or median_finder.maxheap) else 0, 
            "min_height": min_height_in_block if min_height_in_block != float('inf') else 0,
            "page_num": page.number
        })
    
    logger.info(f"Extracted {len(extracted_blocks)} text blocks")
    
    # 7. Handle Unassigned Words (Optional)
    unassigned_word_count = sum(1 for item in word_items if not item["assigned"])
    if unassigned_word_count > 0:
        logger.warning(f"{unassigned_word_count} words were not assigned to any detected region")
        if handle_unassigned_words:
            logger.info("Handling unassigned words not implemented")
    
    if visual_proof and visual_proof_path:
        try:
            # Ensure the output directory exists
            os.makedirs(visual_proof_path, exist_ok=True)
            output_filename = os.path.join(visual_proof_path, f"page_{page.number}_proof.png")

            viz_img = img.copy() 
            img_h, img_w = viz_img.shape[:2]
            middle_x = img_w // 2

            # Draw a vertical line down the middle
            if  is_multicolumn_layout:
                cv2.line(viz_img, (middle_x, 0), (middle_x, img_h), (0, 0, 255), 2) 
            for item in extracted_blocks:
                rect = item["occupy_space"]
                # Convert PDF rect back to image coordinates for drawing
                x0, y0 = int(rect.x0 * zoom_factor), int(rect.y0 * zoom_factor)
                x1, y1 = int(rect.x1 * zoom_factor), int(rect.y1 * zoom_factor)
                cv2.rectangle(viz_img, (x0, y0), (x1, y1), (0, 255, 0), 2) # Green boxes

            fig = plt.figure(figsize=(15, 15)) # Assign figure to a variable

            # Show edge/contour image (derived from binary image)
            plt.subplot(1, 2, 1)
            plt.imshow(edged, cmap='gray')
            plt.title(f"Page {page.number} - Edges from Binary")
            plt.axis('off')

            # Show original image with detected regions overlaid
            plt.subplot(1, 2, 2)
            # Convert viz_img from BGR (OpenCV default) to RGB (matplotlib default) if necessary
            if len(viz_img.shape) == 3 and viz_img.shape[2] == 3:
                viz_img_rgb = cv2.cvtColor(viz_img, cv2.COLOR_BGR2RGB)
                plt.imshow(viz_img_rgb)
            else: # Grayscale or other
                 plt.imshow(viz_img, cmap='gray')
            plt.title(f"Page {page.number} - Detected Text Regions ({len(extracted_blocks)} blocks)")
            plt.axis('off')

            plt.tight_layout()
            # plt.show() # Replaced with savefig
            plt.savefig(output_filename)
            plt.close(fig) # Close the figure to free memory
            print(f"  Saved visual proof to: {output_filename}")

        except Exception as e:
            print(f"  Error generating or saving visual proof: {e}")
    elif visual_proof and not visual_proof_path:
        print("  Warning: visual_proof is True, but visual_proof_path is not provided. Cannot save visuals.")

    return extracted_blocks


def extract_and_prepare_elements(pdf_path):
    """
    Opens PDF, classifies layout, extracts text elements using the provided function,
    ensures required keys are present, sorts by reading order per page,
    and returns a single list of all elements.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF {pdf_path}: {e}")
        return []

    # Try to initialize the layout classification model
    try:
        model_path  = r".\src\classification_model"
        classifier = joblib.load(os.path.join(model_path, "KNN_classifier_model.joblib"))
        scaler = joblib.load(os.path.join(model_path, "feature_scaler.joblib"))
        label_encoder = joblib.load(os.path.join(model_path, "label_encoder.joblib"))

        layout_classifier = LayoutClassificationModel(
            model_name='KNN Layout Classification Model',
            classifier=classifier,
            scaler=scaler,
            label_encoder=label_encoder
        )    
    except Exception as e:
        logger.error(f"Error loading layout classification model: {e}")        
        layout_classifier = None

    all_elements = []
    num_pages = len(doc)
    
    for i in range(num_pages):
        try:
            # Call the external extraction function with layout type
            page_elements = extract_page_text_via_image_regions(page=doc[i], classification_model=layout_classifier, visual_proof=False)

            # Validate and potentially enrich elements
            valid_page_elements = []
            for element in page_elements:
                 # Ensure essential keys exist (add defaults or skip if necessary)
                if not all(k in element for k in ["content", "occupy_space", "text_height_median"]):
                     print(f"    Warning: Skipping element on page {i+1} due to missing keys: {element.get('content', '[No Content]')[:50]}...")
                     continue
                element['page_num'] = i
                valid_page_elements.append(element)

            all_elements.extend(valid_page_elements)

        except Exception as e:
            print(f"Error processing page {i + 1}: {e}")
            continue

    print(f"Finished extraction. Total elements: {len(all_elements)}")
    doc.close()
    return all_elements


def identify_and_filter_watermarks(elements: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], set]:
    """
    Identifies elements with identical stripped content appearing more than once
    and filters them out.
    
    Args:
        elements: List of text elements
        
    Returns:
        Tuple of (filtered elements, watermark candidates)
    """
    if not elements:
        return [], set()

    logger.info("Identifying potential watermarks...")
    # Count occurrences of each unique stripped text content
    text_counts = defaultdict(int)
    
    for el in elements:
        if el.get("content"):
            key = (el["content"].strip(), el.get("text_height_median", 0))
            text_counts[key] += 1
            
    watermark_candidates = {text for text, count in text_counts.items() if count > 1 and text[0]}

    if watermark_candidates:
        logger.info(f"Identified {len(watermark_candidates)} potential watermarks")
        # Filter out elements whose stripped content is in the watermark set
        filtered_elements = [
            el for el in elements
            if el.get("content") and (el["content"].strip(), el["text_height_median"]) not in watermark_candidates
        ]
        logger.info(f"Removed {len(elements) - len(filtered_elements)} watermark instances")
        return filtered_elements, watermark_candidates
    else:
        logger.info("No potential watermarks found")
        return elements, set()
    
    
def extract_text(pdf_path: str, use_mistral: bool = False) -> Tuple[List[Dict[str, Any]], str]:
    """
    Main text extraction function that combines multiple extraction methods.
    
    Args:
        pdf_path: Path to the PDF file
        use_mistral: Whether to try Mistral OCR first
        
    Returns:
        Tuple of (extracted text blocks, method used)
    """
    logger.info(f"Starting text extraction for {pdf_path}")
    
    if use_mistral:
        try:
            # Try Mistral OCR first if specified
            logger.info("Attempting extraction with Mistral OCR")
            blocks = extract_text_with_mistral_ocr(pdf_path)
            return blocks, 'mistral'
        except OCRError as e:
            logger.warning(f"Mistral OCR failed, falling back to PyMuPDF: {str(e)}")
    
    try:
        # Try PyMuPDF
        blocks = extract_text_with_pymupdf(pdf_path)
        return blocks, 'pymupdf'
    except TextExtractionError as e:
        logger.error(f"PyMuPDF extraction failed: {str(e)}")
        raise TextExtractionError(f"All extraction methods failed: {str(e)}")