# %% [markdown]
# We go from processing the srt by concatenating with embedded timestamp [] text and chunk using keyword "Chapter #integer based english word#, "

# %%
import re
class SRTItem:
    def __init__(self, index, text):
        self.index = index
        self.text = text
    def __repr__(self):
        return f"SRTItem(index={self.index}, text={self.text})"
    def __str__(self):
        return f"{self.index}\n{self.text}"
    
def process_srt(srt_file):
    """
    Processes an SRT string to embed the start timestamp at the beginning
    of its corresponding text lines.

    Args:
        srt_content: A string containing the content of an SRT file.

    Returns:
        A string with the modified SRT content, or an empty string
        if the input is invalid or empty.
        Returns the original content if no valid SRT blocks are found.
    """
    with open(srt_file, 'r', encoding='utf-8') as file:
        content = file.read()
    if not content.strip():
        print("There is no content in the SRT")
        return ""
    # Split the content into individual subtitles
    subtitles = [block.strip() for block in re.split(r'\n\s*\n', content.strip())]

    modified_blocks = []
    processed_indicies = set()
    for block in subtitles:
        lines = block.splitlines()
        # Handling invalid blocks
        if len(lines) < 3:
            modified_blocks.append(block)
            continue
        try:
            ### index
            ### timestamp [00:00:13,666 --> 00:00:15,427]
            ### content 
            index_str = lines[0].strip() # timestamp
            index = int(index_str)
            if index in processed_indicies:
                modified_blocks.append(block)
                continue
            processed_indicies.add(index)
            
            # --timestamp--
            timestamp_line = lines[1].strip()
            match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', timestamp_line)
            start_timestamp = match.group(1)
            end_timestamp = match.group(2)
            text_lines = lines[2:]
            if not text_lines:
                modified_blocks.append(block)
                continue
            modified_first_text_line = f"[{start_timestamp},{end_timestamp}] {text_lines[0]}"
            modified_text_lines = [modified_first_text_line] + text_lines[1:]

            modified_block = SRTItem(index, "\n".join(modified_text_lines))
            modified_blocks.append(modified_block)
            
        except (ValueError, IndexError) as e:
            modified_blocks.append(block)

    return modified_blocks

# processed_srt_str =(process_srt(r"D:\DATA300\AudioBookSum\data\AtomicHabit\transcribed (2).srt"))
# print(processed_srt_str)

# %%
import re

def text2int(textnum, numwords={}):
    if not numwords:
      units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
      ]

      tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

      scales = ["hundred", "thousand", "million", "billion", "trillion"]

      numwords["and"] = (1, 0)
      for idx, word in enumerate(units):    numwords[word] = (1, idx)
      for idx, word in enumerate(tens):     numwords[word] = (1, idx * 10)
      for idx, word in enumerate(scales):   numwords[word] = (10 ** (idx * 3 or 2), 0)

    current = result = 0
    for word in textnum.split():
        try:
            if word not in numwords:
                raise Exception("Illegal word: " + word)

            scale, increment = numwords[word]
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
        except Exception as e:
            print(e)
            return None
    return result + current
def int2text(num):
    if num == 0:
        return "Zero"
    
    bigString = ["Thousand", "Million", "Billion"]
    result = numberToWordsHelper(num % 1000)
    num //= 1000
    
    for i in range(len(bigString)):
        if num > 0 and num % 1000 > 0:
            result = numberToWordsHelper(num % 1000) + bigString[i] + " " + result
        num //= 1000
    
    return result.strip()

def numberToWordsHelper(num: int) -> str:
    digitString = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]
    teenString = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    tenString = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
    
    result = ""
    if num > 99:
        result += digitString[num // 100] + " Hundred "
    
    num %= 100
    if num < 20 and num > 9:
        result += teenString[num - 10] + " "
    else:
        if num >= 20:
            result += tenString[num // 10] + " "
        num %= 10
        if num > 0:
            result += digitString[num] + " "
    
    return result

NUM_WORDS = [int2text(i).lower() for i in range(1,100)]
chapter_pattern_text_based_rg = r'Chapter\s+(?:' + '|'.join(NUM_WORDS) + r')\b[\s:.]*'
chapter_pattern_int_based_rg = r'Chapter\s+\d+\b[\s:.]*'
chapter_pattern_text = re.compile(chapter_pattern_text_based_rg, re.IGNORECASE)
chapter_pattern_int = re.compile(chapter_pattern_int_based_rg, re.IGNORECASE)

# %%
# In cell with id 'c8ce6d60'
from typing import List
# Remove RecursiveCharacterTextSplitter import if no longer needed here
# from langchain.text_splitter import RecursiveCharacterTextSplitter
import re # Add import for re

# Assumes SRTItem class is defined in a previous cell
# Assumes chapter_pattern_text, chapter_pattern_int are defined in a previous cell
# Assumes timestamp_pattern is defined in a previous cell (e.g., cell 'ebac4212')

class ChunkedSRTItem:
    def __init__(self, start_index, end_index, chunk: List[SRTItem]): # Ensure chunk type hint is List[SRTItem]
        self.start_index = start_index
        self.end_index = end_index
        self.chunk = chunk if chunk else []

    def __repr__(self):
        return f"ChunkedSRTItem(start={self.start_index}, end={self.end_index}, items={len(self.chunk)})"

# Helper function to get text after timestamp pattern
def _get_text_after_timestamp(text: str, pattern: re.Pattern) -> str:
    """Extracts text content after the initial timestamp pattern."""
    if not text or not pattern:
        return text.strip() if text else ""
    match = pattern.match(text)
    return text[match.end():].strip() if match else text.strip()

def chunking_srt_by_chapter(
    srt_items: List[SRTItem],
    chapter_pattern_text: re.Pattern,
    chapter_pattern_int: re.Pattern,
    timestamp_pattern: re.Pattern # Pass the compiled pattern
) -> List[ChunkedSRTItem]:
    """
    Chunks a list of SRTItem objects based on chapter markers found in the text.
    Each chunk starts with an SRTItem whose text begins with a timestamp annotation '[start,end]'.

    Args:
        srt_items: List of SRTItem objects from process_srt.
        chapter_pattern_text: Compiled regex for text-based chapter markers.
        chapter_pattern_int: Compiled regex for integer-based chapter markers.
        timestamp_pattern: Compiled regex for the timestamp pattern '[start,end]'.

    Returns:
        A list of ChunkedSRTItem objects, where each object's 'chunk' attribute
        is a list of SRTItems belonging to that chapter/segment.
    """
    if not srt_items:
        print("Warning: chunking_srt_by_chapter received empty srt_items list.")
        return []

    # Ensure patterns are compiled regex objects
    if not isinstance(chapter_pattern_text, re.Pattern) or \
       not isinstance(chapter_pattern_int, re.Pattern) or \
       not isinstance(timestamp_pattern, re.Pattern):
        print("Error: Chapter or timestamp patterns are not compiled regex objects.")
        # Potentially raise an error or return empty list
        return []

    chunk_data_by_chapter: List[ChunkedSRTItem] = []
    current_chunk_items: List[SRTItem] = []

    for item in srt_items:
        # Basic validation of the item structure
        if not isinstance(item, SRTItem) or not hasattr(item, 'text') or not hasattr(item, 'index'):
             print(f"Warning: Skipping invalid item in srt_items: {item}")
             continue
        # Ensure text is a string
        if not isinstance(item.text, str):
            print(f"Warning: Skipping item with non-string text (Index: {item.index}): {type(item.text)}")
            continue

        text_after_ts = _get_text_after_timestamp(item.text, timestamp_pattern)

        # Check if the text (after potential timestamp) contains a chapter marker
        is_chapter_start = False
        if text_after_ts: # Only search if there's text after the timestamp
             is_chapter_start = bool(chapter_pattern_text.search(text_after_ts) or \
                                     chapter_pattern_int.search(text_after_ts))

        if is_chapter_start and current_chunk_items:
            # Finalize the previous chunk if it's not empty
            chunk_data_by_chapter.append(ChunkedSRTItem(
                start_index=current_chunk_items[0].index,
                end_index=current_chunk_items[-1].index,
                chunk=current_chunk_items
            ))
            # Start new chunk with the chapter item
            current_chunk_items = [item]
        else:
            # Add item to the current chunk
            current_chunk_items.append(item)

    # Add the last chunk if it exists and is not empty
    if current_chunk_items:
        chunk_data_by_chapter.append(ChunkedSRTItem(
            start_index=current_chunk_items[0].index,
            end_index=current_chunk_items[-1].index,
            chunk=current_chunk_items
        ))

    print(f"Chunking complete. Generated {len(chunk_data_by_chapter)} chapter-based chunks.")
    return chunk_data_by_chapter

# Example usage within this cell (if any) should be updated:
# Replace the line:
# len(chunking_srt(processed_srt_str))
# With:
# Assuming processed_srt_str, chapter_pattern_text, chapter_pattern_int, timestamp_pattern exist
timestamp_re_str = r'\s*\[\s*(\d{2}:\d{2}:\d{2},\d{3})\s*,\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\]\s*'
timestamp_pattern = re.compile(timestamp_re_str)
# chapter_chunks = chunking_srt_by_chapter(processed_srt_str, chapter_pattern_text, chapter_pattern_int, timestamp_pattern)
# # print(f"Number of chapter chunks: {len(chapter_chunks)}")
# # if chapter_chunks:
# #     print(f"First chunk details: {chapter_chunks[0]}")
# print(chapter_chunks)

# %%


# %% [markdown]
# We then run the chapter-based chunks through an llm with a structured output

# %%
# In cell with id 'ebac4212'
import datetime
import os
import json
import re
import difflib
from collections import defaultdict
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Optional, DefaultDict, Type

from pydantic import BaseModel, Field

# Assuming SRTItem and ChunkedSRTItem classes are defined in previous cells
# from <previous_cell_module> import SRTItem, ChunkedSRTItem # Or ensure they are defined above

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    LLMClientType = ChatGoogleGenerativeAI
except ImportError:
    print("Warning: langchain-google-genai not found. Using 'Any' for LLM type hint.")
    print("Please install it: pip install langchain-google-genai")
    LLMClientType = Any

load_dotenv()

class EvidenceSnippet(BaseModel):
    """Represents a single verbatim evidence snippet."""
    verbatim_text: str = Field(..., description="The exact verbatim text snippet extracted from the document, including any '[]' annotation if present at the beginning of the snippet.")

class SummaryAndEvidence(BaseModel):
    """Schema for the desired LLM output containing summary and evidence."""
    summary: str = Field(..., description="A concise summary based solely on the provided document content.")
    evidence: List[EvidenceSnippet] = Field(..., description="A list of verbatim evidence snippets supporting the key points of the summary.")

# Regex to find and capture the [start,end] annotation at the beginning of a string
# Allows for whitespace variations
timestamp_re_str = r'\s*\[\s*(\d{2}:\d{2}:\d{2},\d{3})\s*,\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\]\s*'
timestamp_pattern = re.compile(timestamp_re_str)


def proper_extract(raw_snippet_text: str) -> Tuple[str, Optional[Tuple[str, str]]]:
    """
    Cleans the raw snippet text from the LLM.
    Removes the leading '[start,end]' annotation if present.
    Returns the cleaned text and the extracted timestamp tuple (or None).
    """
    match = timestamp_pattern.match(raw_snippet_text)
    if match:
        # Extract the timestamp tuple ('start', 'end')
        extracted_ts = (match.group(1), match.group(2)) # Correctly capture both groups
        cleaned_text = raw_snippet_text[match.end():].strip()
        return cleaned_text, extracted_ts
    else:
        return raw_snippet_text.strip(), None

def _find_matching_block(
    segment_text: str,
    original_material: List[SRTItem], # List of parsed ORIGINAL SRT dicts
    min_similarity_threshold: float = 0.4 # Threshold for considering a match
) -> Tuple[Optional[Dict[str, Any]], float]:
    """
    Finds the best matching text block in the *original* parsed SRT data for a given segment text
    using similarity matching. Matches against the 'text' field in original_material.
    """
    norm_segment = ' '.join(segment_text.split())
    if not norm_segment:
        return None, 0.0

    best_match_block: Optional[Dict[str, Any]] = None
    highest_similarity: float = 0.0

    for block in original_material:
        # Ensure 'text' key exists and is a non-empty string
        # Use the ORIGINAL text for matching
        block_text = block.text if isinstance(block, SRTItem) else block.get("text", "")
        if not isinstance(block_text, str) or not block_text:
            continue
        block_text, timestamp = proper_extract(block_text)
        norm_block_text = ' '.join(block_text.split())
        if not norm_block_text:
             continue
        
        # Use SequenceMatcher for similarity calculation.
        # autojunk=False can be important for short strings or repetitive text.
        matcher = difflib.SequenceMatcher(None, norm_segment, norm_block_text, autojunk=False)
        similarity = matcher.ratio()
        # print(f"Comparing segment: '{norm_segment[:50]}...' with block: '{norm_block_text[:50]}...' - Similarity: {similarity:.4f}")
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match_block = block
        if similarity == 1:
            break
    if best_match_block and highest_similarity >= min_similarity_threshold:
        return best_match_block, highest_similarity

    return None, 0.0


def _process_segment(
    raw_snippet_text: str,
    index_hint: Optional[Tuple[int, int]], # Optional index hint for the segment
    original_material: List[Dict[str, Any]], # This is the list of ORIGINAL parsed SRT dicts
    min_word_threshold: int
) -> Optional[Dict[str, Any]]:
    """
    Processes a raw evidence snippet text provided by the LLM.
    1. Cleans the snippet (removes potential leading '[start,end]' annotation).
    2. Finds the best matching block in the original parsed SRT data using the cleaned text.
    3. Returns a dictionary with the matched block's actual timestamp tuple
       and the *cleaned* text content from the snippet.
    """
    cleaned_text, _ = proper_extract(raw_snippet_text)

    if not cleaned_text or len(cleaned_text.split()) < min_word_threshold:
        return None
    search_perimeter = original_material if not index_hint else original_material[index_hint[0]-1:index_hint[1]-1]
    # print(index_hint)
    # Step 2: Find the best matching block in the original data
    matched_block, match_similarity = _find_matching_block(
        cleaned_text, search_perimeter 
    )

    if matched_block:
        actual_timestamp_tuple = extract_timestamp_from_text(matched_block.text)
        if not actual_timestamp_tuple or not (isinstance(actual_timestamp_tuple, tuple) and len(actual_timestamp_tuple) == 2):
             print(f"Warning: Matched block (Index: {matched_block.index}) has missing or invalid timestamp: {actual_timestamp_tuple}. Skipping snippet.")
             return None

        # Return the actual timestamp from the matched block and the cleaned text from the LLM snippet
        return {
            "timestamp": actual_timestamp_tuple, 
            "text": cleaned_text, 
            "match_score": match_similarity,
            "matched_index": matched_block.index
        }
    else:
        print(f"Warning: No matching block found (score < threshold) for snippet: '{raw_snippet_text}...' (Cleaned: '{cleaned_text}...')")
        return None

def extract_timestamp_from_text(text: str) -> Optional[Tuple[str, str]]:
    """
    Extracts the timestamp tuple from the beginning of a text snippet.
    Returns a tuple (start, end) if found, otherwise None.
    """
    match = timestamp_pattern.match(text)
    if match:
        return match.groups()
    return None


def identify_interesting_points_structured(
    chunked_data: List[ChunkedSRTItem],
    llm: LLMClientType,
    output_schema: Type[BaseModel],
    original_data: List[SRTItem],
    min_snippet_word_count: int = 5,
    cache_file_path: str = "summary_evidence_cache.json", # Path to cache file
    is_one_p: bool = True # Flag to process all chunks in one prompt
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """
    Identifies summary and evidence snippets using an LLM with structured output.

    Args:
        chunked_data: List of ChunkedSRTItem objects from chunking_srt_by_chapter.
        llm: Initialized LLM client supporting structured output.
        output_schema: The Pydantic BaseModel class (e.g., SummaryAndEvidence).
        original_data: The list of original parsed SRT block dictionaries (before timestamp prefixing).
        cache_file_name: Path to the JSON file for caching LLM results.
        min_snippet_word_count: Minimum word count for a processed evidence snippet.
        is_one_p: If True, concatenates all chunks into one LLM prompt.
                  If False, processes chunks sequentially (requires careful implementation).

    Returns:
        A tuple containing:
        - The generated summary string (or None).
        - A list of processed evidence snippet dictionaries [{'timestamp': (start, end), 'text': ...}].
    """
    summary_result: Optional[str] = None
    processed_segments: List[Dict[str, Any]] = []

    if not chunked_data:
        print("Warning: No chunked data provided.")
        return None, []

    # Input validation for original_data
    if not isinstance(original_data, list) or (original_data and not isinstance(original_data[0], SRTItem)):
         print("Error: `original_data` must be a list of dictionaries (parsed original SRT data).")
         return None, []
    

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
            # logger.info(f"Loading master cache index from {master_cache_path}")
            with open(master_cache_path, "r", encoding="utf-8") as f:
                master_cache = json.load(f)
            summary_result = master_cache.get("summary", "")
            chunk_files = master_cache.get("chunk_files", [])
            
            # Load all chunk files
            raw_cached_snippets = dict()
            for chunk_file in chunk_files:
                chunk_path = os.path.join(chunks_dir, chunk_file)
                if os.path.exists(chunk_path):
                    try:
                        with open(chunk_path, "r", encoding="utf-8") as f:
                            chunk_data = json.load(f)
                            chunk_evidence = [snippet["verbatim_text"] for snippet in chunk_data.get("evidence", [])]
                            
                            raw_cached_snippets[(chunk_data.get("start_index",-1), chunk_data.get("end_index",-1))] = chunk_evidence
                    except Exception as e:
                        # logger.warning(f"Error loading chunk file {chunk_path}: {str(e)}")
                        print(f"Error loading chunk file {chunk_path}: {str(e)}")
            # print(raw_cached_snippets)
            temp_processed_segments = []
            for start_idx, end_idx in raw_cached_snippets.keys():
                for i in raw_cached_snippets[(start_idx, end_idx)]:
                    snippet_text = i
                    possible_idx_range = (start_idx, end_idx) if start_idx != -1 and end_idx != -1 else None
                    processed = _process_segment(snippet_text, possible_idx_range, original_data, min_snippet_word_count)
                    if processed:
                        temp_processed_segments.append(processed)

            processed_segments = temp_processed_segments

            if summary_result or processed_segments:
                # logger.info(f"Cache loaded and validated: Summary: {summary_result is not None}, Snippets: {len(processed_segments)}")
                cache_loaded = True
            else:
                # logger.warning("Cache files loaded but resulted in no valid data after processing. Will regenerate.")
                print("Cache files loaded but resulted in no valid data after processing. Will regenerate.")
        except json.JSONDecodeError:
            # logger.error(f"Error decoding JSON from master cache file {master_cache_path}")
            print(f"Error decoding JSON from master cache file {master_cache_path}. Will regenerate cache.")
        except Exception as e: 
            print(f"Error loading cache files: {e}")
            # logger.error(f"Error loading cache files: {str(e)}")
    if not cache_loaded:
        print("Generating new summary and evidence points using LLM with structured output...")
        chunk_files = []  # List to store chunk file names
        prompt_template_str = os.getenv("SUMMARY_EVIDENCE_PROMPT_STRUCTURED_AB")
        if not prompt_template_str:
            print("Error: Environment variable 'SUMMARY_EVIDENCE_PROMPT_STRUCTURED_AB' not set.")
            return None, []
        verbatim_result_to_chunk_mapping = defaultdict(list) # For mapping verbatim results to chunk indices 
        try:
            response_object = SummaryAndEvidence(summary="", evidence=[])
            if is_one_p:
                all_text_for_llm = "\n\n".join([
                    "\n".join([srt_item.text for srt_item in chunked_srt.chunk])
                    for chunked_srt in chunked_data if chunked_srt.chunk 
                ]).strip()


                if not all_text_for_llm:
                    print("Error: No text content found in chunked_data to send to LLM.")
                    return None, []

                final_prompt = prompt_template_str + "\n\nDocument Content:\n" + all_text_for_llm

                try:
                    structured_llm = llm.with_structured_output(output_schema)
                    print("LLM configured for structured output.")
                except AttributeError:
                    print(f"Error: The provided LLM client object (type: {type(llm)}) does not appear to have a '.with_structured_output' method.")
                    return None, []
                except Exception as e:
                    print(f"Error configuring LLM for structured output: {e}")
                    return None, []

                # logger.info("Sending request to LLM for structured response")
                response: SummaryAndEvidence = structured_llm.invoke(final_prompt)
                # logger.info("Structured response received from LLM")
                
                response_object.summary = response.summary.strip()
                response_object.evidence.extend(response.evidence)
                
                # Save the single chunk
                chunk_filename = f"chunk_all.json"
                chunk_path = os.path.join(chunks_dir, chunk_filename)
                try:
                    with open(chunk_path, "w", encoding="utf-8") as f:
                        json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)
                    chunk_files.append(chunk_filename)
                    # logger.info(f"Saved chunk to {chunk_path}")
                except Exception as e:
                    print(f"Error saving chunk file {chunk_path}: {e}")
                    # logger.error(f"Error saving chunk file {chunk_path}: {str(e)}")

            else:
                chunk_files = []  # List to store chunk file names
                for chunk_idx, chunk in enumerate(chunked_data):
                    if not chunk.chunk: continue
                    chunk_text = "\n".join([item.text for item in chunk.chunk]).strip()
                    if not chunk_text: continue

                    print(f"Processing chunk {chunk_idx+1}...")
                    chunk_prompt = prompt_template_str + "\n\nDocument Content:\n" + chunk_text
                    if response_object.summary:
                        chunk_prompt += f"\n\nCurrent Summary:\n{response_object.summary.strip()}"
                    try:
                        structured_llm = llm.with_structured_output(output_schema)
                        chunk_response: SummaryAndEvidence = structured_llm.invoke(chunk_prompt)
                        response_object.summary += chunk_response.summary.strip() + "\n"
                        verbatim_result_to_chunk_mapping[(len(response_object.evidence),len(response_object.evidence) + len(chunk_response.evidence))] = (chunk.start_index, chunk.end_index)  # Store chunk index for each evidence snippet
                        response_object.evidence.extend(chunk_response.evidence)
                        
                        # Save each chunk separately
                        chunk_filename = f"chunk_{chunk_idx:03d}.json"
                        chunk_response_json = chunk_response.model_dump()
                        chunk_response_json["start_index"] = chunk.start_index
                        chunk_response_json["end_index"] = chunk.end_index 
                        chunk_path = os.path.join(chunks_dir, chunk_filename)
                        with open(chunk_path, "w", encoding="utf-8") as f:
                            json.dump(chunk_response_json, f, ensure_ascii=False, indent=2)
                        print(f"Saved chunk {chunk_idx+1} to {chunk_path}")
                        chunk_files.append(chunk_filename)
                    except Exception as e:
                        print(f"Error processing chunk {chunk_idx+1}: {e}")

            # --- Process LLM response (common for is_one_p=True or after aggregation) ---
            summary_result = response_object.summary
            raw_snippet_objects = response_object.evidence
            raw_snippets_text = [snippet.verbatim_text for snippet in raw_snippet_objects]

            print(f"Extracted Summary: {summary_result is not None}. Extracted Raw Snippets: {len(raw_snippets_text)}")

            processed_segments = []

            print(f"Processing {len(raw_snippets_text)} raw snippets...")
            most_right = 0
            chunks_indexes = []
            curr_chunk = 0
            """
                if i <= most_right:
                    index_hint = chunking_section[curr]
                else:
                    curr += 1
                    most_right = chunking_section[curr][1] if curr < len(chunking_section) else 0
                    index_hint = chunking_section[curr]
                print(f"Processing snippet {i+1} with index hint: {index_hint}")
            """
            if not is_one_p:
                chunks_indexes = [(chunk.start_index, chunk.end_index) for chunk in chunked_data if chunk.chunk]
                most_right = chunks_indexes[curr_chunk][1] if chunks_indexes else 0 
            for i, snippet_text in enumerate(raw_snippets_text):
                index_hint = None
                if not is_one_p:
                    if i <= most_right:
                        index_hint = verbatim_result_to_chunk_mapping.get(chunks_indexes[curr_chunk], None)
                    else:
                        curr_chunk += 1
                        most_right = chunks_indexes[curr_chunk][1] if curr_chunk < len(chunks_indexes) else 0
                        index_hint = verbatim_result_to_chunk_mapping.get(chunks_indexes[curr_chunk], None)
                processed_section = _process_segment(snippet_text,index_hint if most_right > 0 else None, original_data,
                    min_snippet_word_count)
                    
                if processed_section:
                    processed_segments.append(processed_section)
            try:
                master_cache = {
                    "summary": summary_result,
                    "chunk_files": chunk_files,
                    "total_evidence_count": len(raw_snippets_text),
                    "timestamp": str(datetime.datetime.now())
                }
                
                with open(master_cache_path, "w", encoding="utf-8") as f:
                    json.dump(master_cache, f, ensure_ascii=False, indent=2)
                # logger.info(f"Saved master cache index to {master_cache_path}")
            except Exception as e:
                print(f"Error saving master cache file {master_cache_path}: {e}")
                # logger.error(f"Error saving master cache file {master_cache_path}: {str(e)}")
        except Exception as e:
            # logger.error(f"Error during LLM interaction: {str(e)}")
            import traceback
            # logger.error(traceback.format_exc())
            print(f"Error during LLM interaction: {e}")
            summary_result = None
            processed_segments = []

    valid_segments = [
        seg for seg in processed_segments
        if isinstance(seg, dict) and isinstance(seg.get("timestamp"), tuple) and len(seg["timestamp"]) == 2 and isinstance(seg.get("text"), str)
    ]
    if len(valid_segments) != len(processed_segments):
        print(f"Warning: Filtered out {len(processed_segments) - len(valid_segments)} invalid segment structures before returning.")
        processed_segments = valid_segments


    return summary_result, processed_segments


# --- Example Call Setup ---
# Define file paths (ensure these are correct)
original_srt_filepath = r"D:\DATA300\AudioBookSum\data\AtomicHabit\transcribed (2).srt"
notebook_dir = os.path.dirname(os.path.abspath("__file__")) if "__file__" in locals() else os.getcwd() # Get notebook dir
cache_dir = os.path.join(notebook_dir, "cache")
os.makedirs(cache_dir, exist_ok=True) # Ensure cache directory exists

srt_basename = os.path.basename(original_srt_filepath)
srt_name_no_ext = os.path.splitext(srt_basename)[0]
cache_filename = f"{srt_name_no_ext}_summary_evidence_cache.json"
cache_filepath = os.path.join(cache_dir, cache_filename)

# Initialize LLM (ensure API key is set in .env or environment)
try:
    # Make sure GOOGLE_API_KEY is set in your environment or .env file
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.7) # Use a known valid model
    print("LLM Initialized.")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("Ensure GOOGLE_API_KEY is set and the model name is correct.")
    llm = None

processed_srt_str = process_srt(original_srt_filepath)

# --- Data Preparation ---
# 1. Process SRT to add timestamps (assuming processed_srt_str is from cell 2a36cdbf)
# processed_srt_str should be List[SRTItem]
if 'processed_srt_str' not in locals() or not isinstance(processed_srt_str, list):
     print("Error: 'processed_srt_str' (List[SRTItem]) not found or is not a list. Please run the cell '2a36cdbf'.")
     processed_srt_str = [] # Avoid crashing later calls

# 2. Parse the ORIGINAL SRT file to get data for matching
#    (This is crucial for _process_segment to find correct timestamps)
original_parsed_srt_data = process_srt(original_srt_filepath)
original_parsed_srt_data.sort(key=lambda x: x.index)
if not original_parsed_srt_data:
    print("Warning: Could not parse original SRT data. Snippet matching might fail.")

# 3. Define chapter patterns (assuming they are defined in cell c2211bc8)
if 'chapter_pattern_text' not in locals() or 'chapter_pattern_int' not in locals():
    print("Error: Chapter patterns not defined. Please run cell 'c2211bc8'.")
    # Define dummy patterns to avoid crashing, but this won't work correctly
    chapter_pattern_text = re.compile(r"dummy_chapter_text_pattern_never_match")
    chapter_pattern_int = re.compile(r"dummy_chapter_int_pattern_never_match")

# 4. Define timestamp pattern (defined earlier in this cell)
if 'timestamp_pattern' not in locals():
     print("Error: timestamp_pattern not defined.")
     timestamp_pattern = re.compile(r"dummy_ts_pattern") # Avoid crash

# 5. Chunk the processed SRT data by chapter
chapter_chunks: List[ChunkedSRTItem] = []
if processed_srt_str and 'chapter_pattern_text' in locals() and 'chapter_pattern_int' in locals() and 'timestamp_pattern' in locals():
    try:
        # Ensure the chunking function is available (defined in cell c8ce6d60)
        if 'chunking_srt_by_chapter' in globals():
             chapter_chunks = chunking_srt_by_chapter(
                 processed_srt_str,
                 chapter_pattern_text,
                 chapter_pattern_int,
                 timestamp_pattern
             )
        else:
            print("Error: chunking_srt_by_chapter function not found.")
    except Exception as e:
        print(f"Error during chunking: {e}")
else:
    print("Skipping chunking due to missing processed SRT data or chapter patterns.")
total_length = 0
for i in original_parsed_srt_data:
    total_length += (len(i.text.split(' ')))
# print(total_length)
# --- Run the main function ---
if llm and chapter_chunks and original_parsed_srt_data:
    print("\n--- Calling identify_interesting_points_structured ---")
    summary, evidence = identify_interesting_points_structured(
        chunked_data=chapter_chunks,
        llm=llm,
        output_schema=SummaryAndEvidence,
        original_data=original_parsed_srt_data, # Pass the original parsed data
        cache_file_path= cache_filepath, # Path to cache file
        min_snippet_word_count=5,
        is_one_p=total_length <= 8192 
    )
    print("\n--- Summary ---")
    print(summary if summary else "No summary generated.")
    print("\n--- Processed Evidence ---")
    if evidence:
        for i, ev in enumerate(evidence):
            idx = ev.get('matched_index', 'N/A')
            ts = ev.get('timestamp', ('N/A', 'N/A'))
            score = ev.get('match_score', 0)
            text = ev.get('text', '')
            print(f"{i+1}. Index: {idx}, TS: {ts}, Score: {score:.2f}, Text: {text[:100]}...")
    else:
        print("No evidence snippets processed.")
else:
    print("\n--- Execution Skipped ---")
    if not llm: print("Reason: LLM not initialized.")
    if not chapter_chunks: print("Reason: Chapter chunks are empty or not generated.")
    if not original_parsed_srt_data: print("Reason: Original parsed SRT data is missing.")


# %%
# Merging Intervals within 1s proximity range
intervals = [(1,5),(6,9),(11,15),(16,20),(21,25)]
pointer = 0
curr_interval = intervals[pointer]
merged_intervals = []
for i in range(1, len(intervals)):
    if intervals[i][0] - curr_interval[1] <= 1:  # Check if the start of the current interval is within 1s of the end of the last merged interval
        curr_interval = (curr_interval[0], intervals[i][1])
        continue
    merged_intervals.append(curr_interval)
    curr_interval = intervals[i]
merged_intervals.append(curr_interval)  # Add the last interval after the loop
print(merged_intervals)  # Output the merged intervals

# %%
# In cell with id 'e149cdf7'

import json
import re
from pydub import AudioSegment
# ...existing code...
import traceback # Import traceback for detailed error logging
import os # Ensure os is imported

class SnippetItem:
    """Represents a single snippet with its text and timestamp."""
    def __init__(self, text: str, timestamp: Tuple[int, int]):
        self.text = text
        self.timestamp = timestamp

    def __repr__(self):
        return f"SnippetItem(text={self.text[:30]}..., timestamp={self.timestamp})"

# --- Configuration ---
# Use the correct master cache file name based on the previous cell's output
srt_basename = os.path.basename(original_srt_filepath) # Assumes original_srt_filepath is defined in the previous cell
srt_name_no_ext = os.path.splitext(srt_basename)[0]
master_cache_filename = f"{srt_name_no_ext}_summary_evidence_cache_master.json"
master_cache_file_path = os.path.join(cache_dir, master_cache_filename) # Assumes cache_dir is defined
chunks_dir = os.path.join(cache_dir, f"{srt_name_no_ext}_summary_evidence_cache_chunks") # Directory containing individual chunk JSONs

audio_file_path = r"D:\DATA300\AudioBookSum\data\AtomicHabit\AtomicHabitsAnEasyProvenWaytoBuildGoodHabitsBreakBadOnes.flac" # Ensure this path is correct
base_output_dir = r"d:\DATA300\AudioBookSum\output_snippets_by_chunk" # Base directory to save chunk snippet folders

# Regex pattern for timestamp (ensure it's defined or copied from the previous cell)
timestamp_re_str = r'\s*\[\s*(\d{2}:\d{2}:\d{2},\d{3})\s*,\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\]\s*'
timestamp_pattern = re.compile(timestamp_re_str)

def srt_timestamp_to_ms(timestamp_str):
    """Converts an SRT timestamp string (HH:MM:SS,ms) to milliseconds."""
    try:
        h, m, s_ms = timestamp_str.split(':')
        s, ms = s_ms.split(',')
        # Add a small buffer (e.g., 100ms) to the end time to avoid cutting off audio slightly early
        # Keep start time precise
        total_ms = int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
        return total_ms
    except ValueError as e:
        print(f"Error converting timestamp '{timestamp_str}': {e}")
        raise # Re-raise the error to be caught later

def save_merged_snippet(interval_ms, text, audio_segment, output_dir, snippet_index):
    """Saves the merged audio snippet and its corresponding transcript."""
    global total_snippets_saved # Use global counter

    start_ms, end_ms = interval_ms
    # Add buffer to end time for audio cutting only
    buffered_end_ms = end_ms + 100
    print(f"  Saving merged snippet: {text[:50]}... from {start_ms}ms to {end_ms}ms")

    try:
        # --- Cut Audio ---
        snippet = audio_segment[start_ms:buffered_end_ms]

        # --- Create Filenames ---
        # Clean text for filename preview (use start time for uniqueness)
        clean_text_preview = re.sub(r'[\\/*?:\"<>|,.]', '', text)[:30]
        safe_filename_part = re.sub(r'\s+', '_', clean_text_preview)
        base_filename = f"merged_snippet_{snippet_index:03d}_{start_ms}_{safe_filename_part}"
        audio_output_filename = os.path.join(output_dir, f"{base_filename}.mp3")
        text_output_filename = os.path.join(output_dir, f"{base_filename}.txt")

        # --- Save Audio ---
        snippet.export(audio_output_filename, format="mp3")
        print(f"    Audio saved to: {audio_output_filename}")

        # --- Save Transcript ---
        with open(text_output_filename, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"    Transcript saved to: {text_output_filename}")

        total_snippets_saved += 1
        return True # Indicate success

    except Exception as e:
        print(f"    Error saving merged snippet starting at {start_ms}ms: {e}")
        print(traceback.format_exc())
        return False # Indicate failure

# Create base output directory if it doesn't exist
os.makedirs(base_output_dir, exist_ok=True)

# --- Load Master Cache ---
chunk_files = []
try:
    print(f"Loading master cache index from: {master_cache_file_path}")
    with open(master_cache_file_path, 'r', encoding='utf-8') as f:
        master_data = json.load(f)
    chunk_files = master_data.get("chunk_files", [])
    if not chunk_files:
        print(f"Warning: No 'chunk_files' listed in {master_cache_file_path}")
    else:
        print(f"Found {len(chunk_files)} chunk files listed in master cache.")
except FileNotFoundError:
    print(f"Error: Master cache file not found at {master_cache_file_path}")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {master_cache_file_path}")
except Exception as e:
    print(f"An unexpected error occurred while reading master cache: {e}")
    print(traceback.format_exc())

# --- Load Audio File ---
audio = None
if chunk_files:
    try:
        print(f"Loading audio file: {audio_file_path}...")
        audio = AudioSegment.from_file(audio_file_path, format="flac")
        print("Audio file loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_file_path}")
    except Exception as e:
        print(f"Error loading audio file with pydub: {e}")
        print("Ensure ffmpeg is installed and in your system's PATH.")
        print(traceback.format_exc())

total_snippets_processed = 0
total_snippets_saved = 0 

if audio and chunk_files:
    print(f"\n--- Processing {len(chunk_files)} Chunk Files ---")
    for chunk_filename in chunk_files:
        chunk_file_path = os.path.join(chunks_dir, chunk_filename)
        chunk_output_dir_name = os.path.splitext(chunk_filename)[0] # e.g., "chunk_000"
        chunk_output_dir = os.path.join(base_output_dir, chunk_output_dir_name)
        os.makedirs(chunk_output_dir, exist_ok=True)

        print(f"\nProcessing chunk file: {chunk_filename} -> Outputting to: {chunk_output_dir}")

        evidence_list_raw = []
        try:
            with open(chunk_file_path, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            evidence_list_raw = chunk_data.get("evidence", [])
            if not evidence_list_raw:
                print(f"  No 'evidence' found in {chunk_filename}. Skipping.")
                continue
            print(f"  Found {len(evidence_list_raw)} raw evidence items in this chunk.")

        except FileNotFoundError:
            print(f"  Error: Chunk file not found at {chunk_file_path}. Skipping.")
            continue
        except json.JSONDecodeError:
            print(f"  Error: Could not decode JSON from {chunk_file_path}. Skipping.")
            continue
        except Exception as e:
            print(f"  An unexpected error occurred while reading {chunk_filename}: {e}")
            print(traceback.format_exc())
            continue

        # --- Process Evidence, Merge Consecutive, and Cut Snippets ---
        chunk_snippets_saved_count = 0
        current_merged_interval = None
        current_merged_text = ""
        merge_proximity_ms = 1000 

        for i, item in enumerate(evidence_list_raw):
            total_snippets_processed += 1
            verbatim_text = item.get("verbatim_text", "")
            if not verbatim_text.strip():
                print(f"  Skipping empty evidence item {i+1} in {chunk_filename}.")
                continue

            match = timestamp_pattern.match(verbatim_text)
            if not match:
                print(f"  Skipping evidence item {i+1}: Could not extract timestamp from text: {verbatim_text[:50]}...")
                continue

            start_ts_str, end_ts_str = match.groups()
            text_content = verbatim_text[match.end():].strip()

            try:
                start_ms = srt_timestamp_to_ms(start_ts_str)
                end_ms = srt_timestamp_to_ms(end_ts_str)

                # Basic sanity check
                if start_ms >= end_ms:
                    print(f"  Skipping snippet {i+1}: Start time ({start_ms}ms) is not before end time ({end_ms}ms). Text: {text_content[:50]}...")
                    continue

                # --- Merging Logic ---
                if current_merged_interval is None:
                    # Start a new merged interval
                    current_merged_interval = (start_ms, end_ms)
                    current_merged_text = text_content
                elif start_ms - current_merged_interval[1] <= merge_proximity_ms:
                    # Merge with the previous interval: extend end time and append text
                    current_merged_interval = (current_merged_interval[0], max(current_merged_interval[1], end_ms)) # Take the later end time
                    current_merged_text += " " + text_content
                else:
                    # Gap detected: Save the PREVIOUS merged interval
                    if save_merged_snippet(current_merged_interval, current_merged_text, audio, chunk_output_dir, chunk_snippets_saved_count + 1):
                         chunk_snippets_saved_count += 1
                    # Start a NEW merged interval with the current item
                    current_merged_interval = (start_ms, end_ms)
                    current_merged_text = text_content

            except ValueError as e: # Catch timestamp conversion errors
                print(f"  Skipping snippet {i+1}: Error converting timestamp - {e}. Text: {verbatim_text[:50]}...")
                # If error occurs, reset current merge state to avoid issues
                current_merged_interval = None
                current_merged_text = ""
                continue
            except Exception as e: # Catch other unexpected errors during processing
                print(f"  Skipping snippet {i+1}: Unexpected error processing item - {e}. Text: {verbatim_text[:50]}...")
                print(traceback.format_exc())
                current_merged_interval = None
                current_merged_text = ""
                continue

        # --- Save the LAST merged interval after the loop ---
        if current_merged_interval is not None:
            if save_merged_snippet(current_merged_interval, current_merged_text, audio, chunk_output_dir, chunk_snippets_saved_count + 1):
                 chunk_snippets_saved_count += 1

        print(f"  Finished processing chunk {chunk_filename}. Saved {chunk_snippets_saved_count} merged snippets.")


    print(f"\n--- Overall Summary ---")
    print(f"Processed {total_snippets_processed} raw evidence items across all chunks.")
    print(f"Successfully extracted and saved {total_snippets_saved} merged audio snippets and transcripts to respective folders in {base_output_dir}")

else:
    print("\n--- Execution Skipped ---")
    if not audio:
        print("Reason: Audio file could not be loaded.")
    if not chunk_files:
         print("Reason: No chunk files found or loaded from the master cache.")


# %% [markdown]
# Now we have the snippets, what do we do?
# 1. Load it onto cloud databases for future processes
# 2. Generate an overarching audio summarizing everything

# %% [markdown]
# 

# %%



