import os
import fitz  # PyMuPDF
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate


import argparse
from dotenv import load_dotenv
from mistralai import Mistral
load_dotenv()
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
from pathlib import Path
import json

def extract_text_with_bboxes(pdf_path):
    """Extract text and bounding boxes from a PDF file using Mistral OCR."""
    try:
        # Get API key from environment variables
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY not found in environment variables")
        uploaded_file = client.files.upload(
        file={
                "file_name": pdf_path,
                "content": open(pdf_path, "rb"),
            },
            purpose="ocr",
        )
        client = Mistral(api_key=api_key)
        
        # Process the PDF with Mistral OCR
        signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

        pdf_response = client.ocr.process(document=DocumentURLChunk(document_url=signed_url.url), model="mistral-ocr-latest", include_image_base64=True)

        response_dict = json.loads(pdf_response)
        
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
        
        return all_blocks
    except Exception as e:
        print(f"Error using Mistral OCR: {e}")
        print("Falling back to PyMuPDF extraction.")
        return extract_text_with_pymupdf(pdf_path)

def extract_text_with_pymupdf(pdf_path):
    """Original PyMuPDF extraction as fallback."""
    doc = fitz.open(pdf_path)
    all_blocks = []
    
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")
        # Saving every written block to a text file for human inspection
        # with open(f"page_{page_num}.txt", "w", encoding="utf-8") as f:
        #     for block in blocks:
        #         x0, y0, x1, y1, text, block_no, block_type = block
        #         f.write(f"Block {block_no} ({block_type}) at ({x0}, {y0})-({x1}, {y1}): {text.strip()}\n")
        for block in blocks:
            # Each block is (x0, y0, x1, y1, text, block_no, block_type)
            x0, y0, x1, y1, text, block_no, block_type = block
            if not text.strip():
                continue
            
            # Add page number to the block info
            all_blocks.append({
                "page": page_num,
                "bbox": (x0, y0, x1, y1),
                "text": text.strip(),
                "block_no": block_no,
                "block_type": block_type
            })
    
    doc.close()
    return all_blocks

def classify_text_blocks(blocks):
    """Classify text blocks into main content, footnotes, and extra information."""
    classified_blocks = []
    heights = []
    for block in blocks:
        bbox = block["bbox"]
        height = bbox[3] - bbox[1]  # y1 - y0
        heights.append(height)
    if not heights:
        print("No blocks to analyze")
        return None
    
    median_height = np.median(heights)
    # Process each block
    for block in blocks:
        # Simple heuristics for initial classification
        text = block["text"]
        bbox = block["bbox"]
        classification = "main"  # Default classification
        box_height = bbox[3] - bbox[1]
        # Footnote heuristics: typically smaller text at bottom of page or starts with numbers/symbols
        if any(text.startswith(prefix) for prefix in ["1.", "2.", "*", "†"]):
            classification = "footnote"
        # Extra info heuristics: sidebars, captions, etc. (often in boxes or with different formatting)
        elif np.floor(median_height)<= box_height <= np.ceil(median_height): 
            classification = "main"
        else:
            classification = "extra"
            
        # print(classification, text)
        if classification == "main":
            # Check if the text is too short to be main content
            if "Letter" in text or text == "Carus" or text.startswith("1") or text.startswith("6"):
                classification = "extra"
        block["category"] = classification
        classified_blocks.append(block)
    
    return classified_blocks

def chunk_text(blocks, chunk_size=1000, chunk_overlap=200):
    """Chunk text blocks into smaller pieces for processing."""
    # Separate blocks by category
    categories = {"main": [], "footnote": [], "extra": []}
    for block in blocks:
        categories[block["category"]].append(block)
    
    chunked_data = {}
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    for category, category_blocks in categories.items():
        # Extract text from blocks
        texts = [block["text"] for block in category_blocks]
        full_text = "\n\n".join(texts)
        
        # Chunk the text
        chunks = text_splitter.split_text(full_text)
        chunked_data[category] = chunks
    
    return chunked_data


def identify_interesting_points(chunked_data, llm, blocks, file_name):
    """Use an LLM to identify interesting or important points in the text."""
    interesting_sections = []
    
    # Only process the 'main' category
    if 'main' not in chunked_data:
        print("No main content identified in the document")
        return interesting_sections
        
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="""
        You are a curious student, tasked with annotating a piece of text for a minimum of 10 interesting points.
        Return a listing of interesting points in the text in the form of direct snippets extracted from the text itself.
        
        DO NOT include any other text nor return nothing.
        
        Format each segment on a new line, wrapped in triple backticks, like:
        ```segment 1```
        ```segment 2```

        Below is text from the main content of a document in English:
        {text}
        """
    )
    
    # for chunk in chunked_data['main']:
    chunk = "\n".join(chunked_data['main'])
    
    prompt = prompt_template.format(text=chunk)
    interesting_points = []
    
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if not len(line.strip()):
                    continue
                interesting_points.append(line)
            for segment in interesting_points:
                threshold = 5
                segment = segment.strip()
                if len(segment.split()) < threshold:
                    continue
                for block in blocks:
                    intersect_more_more_than_threshold = False
                    if len(segment.split()) >= threshold:
                        count = 0
                        for chip in block["text"].split(" "):
                            if chip in segment.split(" "):
                                count += 1
                        print("count:", count, "len:", len(segment.split()), "threshold:", len(segment.split())*(1/threshold), "seg:", segment)
                        if count >= len(segment.split(" "))*(1/threshold):
                            intersect_more_more_than_threshold = True
                            # print('seg:', segment,"\n block:\n", block["text"],"\n")
                                
                    if (segment in block["text"] or intersect_more_more_than_threshold):
                        interesting_sections.append({
                            "page": block["page"],
                            "text": segment,
                            "category": "main",
                            "bbox": block["bbox"]
                        })
                        # print(segment)
    if not interesting_points:
        try:
            response = llm.invoke(prompt)
            # Extract segments between triple backticks
            import re
            segments = re.findall(r'```(.*?)```', response.content, re.DOTALL)
            
            for segment in segments:
                segment = segment.strip()
                # print(segment)
                for block in blocks:
                    if segment in block["text"] and block["category"] == "main":
                        # print(block["page"])
                        interesting_sections.append({
                            "page": block["page"],
                            "text": segment,
                            "category": "main",
                            "bbox": block["bbox"]
                        })
                        break
            with open(file_name, "w", encoding="utf-8") as f:
                f.write("\n\n".join(segments))
        except Exception as e:
            print(f"Error during interesting point extraction: {e}")
        
    return interesting_sections

def highlight_interesting_points(pdf_path, interesting_points, output_path):
    """Add highlights to the interesting points in the PDF."""
    doc = fitz.open(pdf_path)
    
    # Using cyan highlight color for main content
    highlight_color = (0, 1, 1)  # RGB for cyan
    fail_count = 0
    for point in interesting_points:
        page = doc[point["page"]]
        text = point["text"]
        if not text:
            print(f"Empty text for page {point['page']}")
            continue
        # Search for text on the page
        # cross check with bbox
        text_instances = page.search_for(text)
        if not text_instances:
            # print(f"Text not found on page {point['page']}: {text}")
            fail_count += 1
            continue
        # if text in highlighted_text:
        #     print(f"Text already highlighted on page {point['page']}: {text}")
        # Highlight each instance of the text
        for inst in text_instances:
            # Add highlight annotation
            highlight = page.add_highlight_annot(inst)
            # Set color
            highlight.set_colors(stroke=highlight_color)
            highlight.update()
        # highlighted_text += text
    print(f"Failed to highlight {fail_count} segments")
    # Save the highlighted PDF
    doc.save(output_path)
    doc.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract, analyze and highlight text from PDF documents")
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--output", default=None, help="Output path for highlighted PDF")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model to use (default: gemini-pro)")
    args = parser.parse_args()
    
    pdf_file = Path(args.pdf)
    assert pdf_file.is_file()
    # Set default output paths if not provided
    if not args.output:
        base_name = os.path.splitext(args.pdf)[0]
        args.output = f"{base_name}_highlighted.pdf"
    
    # Initialize LLM
    try:
        llm = ChatGoogleGenerativeAI(model=args.model, temperature=0)
    except Exception as e:
        print(f"Error initializing Gemini LLM: {e}")
        print("Make sure you have set GOOGLE_API_KEY in your environment or .env file")
        exit(1)
    
    print(f"Processing PDF: {args.pdf}")
    
    # Extract text and bounding boxes
    blocks = extract_text_with_pymupdf(args.pdf)
    


    print(f"Extracted {len(blocks)} text blocks")
    
    # Classify text blocks
    classified_blocks = classify_text_blocks(blocks)
    print("Classified text blocks")
    # Print all main text concatenated
    with open('content.txt', "w", encoding="utf-8") as f:
        f.write(" ".join([block["text"] for block in classified_blocks if block["category"] == "main"]))
    # # Chunk text
    chunked_data = chunk_text(classified_blocks)
    print("Chunked text for processing")
    
    # # Identify interesting points (main content only)
    interesting_points = identify_interesting_points(chunked_data, llm, classified_blocks, f"{base_name}_interesting_points.txt")  # Save to file
    # load from memory
    print(f"Identified {len(interesting_points)} interesting points in main content")
    
    # # Highlight interesting points in the PDF
    highlight_interesting_points(args.pdf, interesting_points, args.output)
    print(f"Created highlighted PDF: {args.output}")
    
    print("Processing complete!")
    # uv run D:\DATA300\AudioBookSum\highlighter.py --pdf D:\DATA300\AudioBookSum\pdf\Nasim1_15.pdf