"""
Transform module for the book pipeline.
Handles document structure extraction, chunking, and hierarchy building.
"""

import os
import re
import logging
import json
import copy
import pandas as pd
from collections import deque, Counter, defaultdict
from typing import List, Dict, Any, Tuple, Optional, Union
import time
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .utils import Node, MedianFinder
from .constants import DUMMY_NODE_HEIGHT, PAGE_PATTERN, DOCUMENT_ROOT_TITLE, BLOCK_BREAK_MARKER
from .exceptions import StructureExtractionError, TextChunkingError

# Set up logging
logger = logging.getLogger(__name__)


def calculate_child_spans(elements: List[Dict[str, Any]]) -> List[int]:
    """
    Calculates the 'span' for each element, representing how many subsequent,
    smaller-height elements fall under it before encountering an element of
    equal or greater height. This is used to determine parent-child relationships.

    Args:
        elements: List of text elements with text_height_median property
        
    Returns:
        List of span counts for each element
    """
    if not elements:
        return []

    logger.info("Calculating hierarchy markers (child spans) based on text height")
    
    # Add a dummy element at the end with a very large height
    try:
        dummy_element = elements[-1].copy() if elements else {}
        dummy_element["text_height_median"] = DUMMY_NODE_HEIGHT
    except KeyError:
        logger.warning("Last element missing 'text_height_median'. Using default for dummy node.")
        dummy_element = {"text_height_median": DUMMY_NODE_HEIGHT}

    processing_list = elements + [dummy_element]
    num_processing = len(processing_list)
    # Stores how many elements immediately following element `i` are 'under' it
    child_span_counts = [0] * num_processing
    stack = []  # Stores indices of elements waiting for their scope boundary

    for idx, current_element in enumerate(processing_list):
        current_height = current_element.get("text_height_median", 0)

        # While stack is not empty AND current element's height is >= stack top element's height:
        # The current element marks the end of the scope for the element at the top of the stack.
        while stack and current_height >= processing_list[stack[-1]].get("text_height_median", 0):
            parent_index = stack.pop()
            # Calculate the span: number of elements between parent and current element
            child_span_counts[parent_index] = idx - parent_index - 1

        # Push the index of the current element onto the stack
        stack.append(idx)

    logger.info("Finished calculating child spans")
    return child_span_counts[:-1]  # Remove the dummy element's span


def build_hierarchy(elements: List[Dict[str, Any]], child_span_counts: List[int]) -> Tuple[List[Node], List[Node]]:
    """
    Builds a tree structure using Node objects based on the elements and their
    calculated child spans. Identifies root nodes (potential chapters/sections).

    Args:
        elements: List of text elements
        child_span_counts: List of child span counts for each element
        
    Returns:
        Tuple of (all nodes, root nodes)
    """
    if not elements or len(elements) != len(child_span_counts):
        logger.warning("Mismatch between elements and child spans, or list is empty")
        return [], []

    logger.info("Building hierarchy tree")
    nodes = [Node(el) for el in elements]
    num_nodes = len(nodes)

    for i in range(num_nodes):
        span = child_span_counts[i]
        if span > 0:
            # This node `i` is a parent. Add its direct children.
            current_child_idx = i + 1
            while current_child_idx < min(i + 1 + span, num_nodes):
                # Add node at `current_child_idx` as a child of node `i`
                nodes[i].add_child(nodes[current_child_idx])
                grandchild_span = child_span_counts[current_child_idx]
                current_child_idx += (grandchild_span + 1)  # Move past the child and its descendants

    # Identify root nodes: typically nodes that have children (parents)
    root_nodes = [node for i, node in enumerate(nodes) if child_span_counts[i] > 0]

    if not root_nodes and nodes:
        logger.warning("No parent nodes identified based on height/span. Assuming flat structure.")
        # Check if all spans are zero
        if sum(child_span_counts) == 0:
            root_nodes = nodes
        else:
            logger.warning("Ambiguous root node condition. Defaulting to all nodes as roots.")
            root_nodes = nodes
    elif not nodes:
        logger.warning("No text elements to form a hierarchy")
        root_nodes = []
    else:
        # Create an overarching single root node to ensure tree structure is maintained
        root_node = Node({
            "content": DOCUMENT_ROOT_TITLE, 
            "occupy_space": pymupdf.Rect(0, 0, 0, 0), 
            "text_height_median": 0
        })
        root_node.children = root_nodes
        logger.info(f"Identified {len(root_nodes)} potential root nodes (chapters/sections)")
        return nodes, [root_node]  # Return all nodes and the unified root node

    return nodes, root_nodes


def generate_json_structure(root_nodes: List[Node]) -> List[Dict[str, Any]]:
    """
    Generates a nested list of dictionaries (JSON structure) from the root nodes
    using Breadth-First Search (BFS) to traverse the hierarchy.
    
    Args:
        root_nodes: List of root nodes
        
    Returns:
        List of dictionaries representing the document structure
    """
    if not root_nodes:
        return []

    logger.info("Generating JSON output")
    output_json_list = []
    # Keep track of nodes already added to the JSON globally
    visited_nodes = set()

    def bfs_to_dict(start_node):
        """Performs BFS from a node and builds the nested dictionary for its branch."""
        # Check if this node was already processed as part of another root's tree
        if start_node in visited_nodes:
            return None

        # Use a queue for BFS: stores tuples of (node_to_process, dict_to_populate_for_that_node)
        root_dict = {}
        queue = deque([(start_node, root_dict)])
        # Local visited set to prevent cycles within this specific BFS call
        local_visited = {start_node}

        while queue:
            current_node, current_data_dict = queue.popleft()

            # Mark node as globally visited after processing its data
            visited_nodes.add(current_node)

            # Populate the dictionary for the current node
            content_text = current_node.content.get("content", "")
            page_num = current_node.content.get("page_num", 0)
            current_data_dict.update({
                "name": f"[Page:{page_num}] {content_text.strip()}",
                "children": []
            })

            child_list = current_data_dict["children"]
            for child_node in current_node.children:
                # Process child only if not visited globally or within this BFS call
                if child_node not in visited_nodes and child_node not in local_visited:
                    local_visited.add(child_node)
                    child_data = {}
                    child_list.append(child_data)
                    queue.append((child_node, child_data))

        return root_dict

    # Iterate through the identified root nodes
    for root in root_nodes:
        branch_dict = bfs_to_dict(root)
        if branch_dict:  # Only add if the BFS generated a structure
            output_json_list.append(branch_dict)

    logger.info(f"Finished generating JSON structure with {len(output_json_list)} top-level items")
    return output_json_list


def get_depth_mapping(data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> List[int]:
    """
    Performs a Breadth-First Search (BFS) on the nested data structure
    and counts the number of nodes at each depth level.

    Args:
        data: The nested data structure (from JSON)
        
    Returns:
        A list where the index represents the depth level and the value is the number of nodes
    """
    if not data:
        return []

    depth_counts = []
    queue = deque([(data, 0)])  # Initialize queue with the root node and depth 0

    while queue:
        current_node, depth = queue.popleft()
        current_node = current_node[0] if isinstance(current_node, list) else current_node
        
        # Ensure the depth list is long enough
        while len(depth_counts) <= depth:
            depth_counts.append(0)
        
        # Increment the count for the current depth
        depth_counts[depth] += 1
        
        # Add children to the queue if they exist
        if isinstance(current_node, dict) and 'children' in current_node:
            name = current_node.get('name', 'Unnamed Node')
            logger.debug(f"Processing node at depth {depth}: {name[:50]}...")
            children = current_node['children']
            for child in children:
                queue.append((child, depth + 1))
    
    return depth_counts


def collect_all_text(node: Dict[str, Any], page_number: int = -1) -> str:
    """
    Recursively collects 'name' fields from a node and all its descendants.
    
    Args:
        node: Node dictionary from the JSON structure
        page_number: Page number to check for removing redundant page annotations
        
    Returns:
        Concatenated text from the node and its descendants
    """
    if not isinstance(node, dict):
        return ""
        
    current_name = str(node.get("name", "")).strip()
    parent_match = re.match(PAGE_PATTERN, current_name)
    parent_page_number = int(parent_match.group(1)) if parent_match else -1
    
    text_parts = []
    if parent_page_number == page_number:
        # Remove the page_number from current_name
        current_name = current_name[parent_match.end():].strip() if parent_match else current_name
        
    if current_name:
        text_parts.append(current_name)

    # Recursively collect text from children
    if "children" in node and isinstance(node["children"], list):
        for child in node["children"]:
            child_text = collect_all_text(child, parent_page_number)
            if child_text:
                text_parts.append(child_text)

    # Join collected parts with a space
    return " ".join(text_parts).strip()


def clone_and_modify_recursive(node: Dict[str, Any], current_depth: int, target_merge_start_level: int) -> Optional[Dict[str, Any]]:
    """
    Recursive helper function to clone the structure.
    If a node is at target_merge_start_level, it collects all text
    from its children and their descendants into its own 'name' field,
    discarding the children.
    
    Args:
        node: Node dictionary from the JSON structure
        current_depth: Current depth in the tree
        target_merge_start_level: Level at which to start merging
        
    Returns:
        Cloned and potentially modified node
    """
    if not isinstance(node, dict):
        return None

    cloned_node = {
        "name": node.get("name", ""),
        "children": []
    }

    target_parent_level = target_merge_start_level

    if current_depth == target_parent_level:
        original_name = str(node.get("name", "")).strip()
        concatenated_text = ""
        match = re.match(PAGE_PATTERN, original_name)
        page_number = int(match.group(1)) if match else -1
        
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                descendant_text = collect_all_text(child, page_number)
                if descendant_text:
                    concatenated_text += descendant_text + " "
                    
        merged_name = concatenated_text.strip()
        merged_name_match = re.match(PAGE_PATTERN, merged_name)
        merged_name_page = int(merged_name_match.group(1)) if merged_name_match else -1
        
        if merged_name_page == page_number:
            merged_name = merged_name[merged_name_match.end():].strip() if merged_name_match else merged_name
            
        if original_name and merged_name:
            cloned_node["name"] = original_name + " " + merged_name
        elif merged_name:
            cloned_node["name"] = merged_name
            
        cloned_node["children"] = []
    
    elif current_depth < target_parent_level:
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                # Recurse: process child at next depth, same target level
                cloned_child = clone_and_modify_recursive(child, current_depth + 1, target_merge_start_level)
                if cloned_child:  # Add valid cloned children
                    cloned_node["children"].append(cloned_child)
                    
    if len(cloned_node.get("name", "")):
        return cloned_node
    else:
        return None


def process_structure_merge_after(data: Union[List[Dict[str, Any]], Dict[str, Any]], merge_start_level: int) -> Union[List[Dict[str, Any]], Dict[str, Any], None]:
    """
    Clones the nested data structure and merges the 'name' fields
    of all nodes at and below 'merge_start_level' into their respective
    ancestor node at 'merge_start_level - 1'.

    Args:
        data: The nested data structure (list of dicts or a single dict)
        merge_start_level: The depth level (0-indexed) from which merging should begin
        
    Returns:
        The cloned and modified data structure
    """
    if merge_start_level <= 0:
        logger.warning("merge_start_level must be 1 or greater. Returning original structure.")
        return copy.deepcopy(data)

    target_parent_level = merge_start_level
    logger.info(f"Merging content from level {merge_start_level} and deeper into nodes at level {target_parent_level}")

    if isinstance(data, list):
        # If the input is a list of root nodes (depth 0)
        new_data_list = []
        for root_node in data:
            cloned_root = clone_and_modify_recursive(root_node, 0, merge_start_level)
            if cloned_root:
                new_data_list.append(cloned_root)
        
        return new_data_list
    elif isinstance(data, dict):
        cloned_root = clone_and_modify_recursive(data, 0, merge_start_level)
        return cloned_root
    else:
        logger.error("Input data must be a list or a dictionary")
        return None


def chunk_text_v2(json_data: Union[List[Dict[str, Any]], Dict[str, Any]], chunk_size: int = 50000, chunk_overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Splits text blocks into smaller chunks for processing.
    Uses RecursiveTextSplitter from LangChain for efficient splitting.
    
    Args:
        json_data: The nested JSON document structure
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List of text chunks
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunked_data = []
        
        # Get the depth mapping to identify content layers
        depth_map = get_depth_mapping(json_data)
        logger.info(f"Nodes per depth level: {depth_map}")
        new_structure = None
        
        if depth_map and len(depth_map) > 1:
            try:
                # Find the maximum count starting from index 1
                max_nodes = max(depth_map[1:])
                # Find the first index (level) corresponding to that max count, starting from index 1
                content_level_index = depth_map.index(max_nodes, 1) - 1
                logger.info(f"Potential content layer: Level {content_level_index} with {depth_map[content_level_index]} nodes")

                # Process the structure, merging content at the identified level
                logger.info(f"Merging content starting from level: {content_level_index}")
                new_structure = process_structure_merge_after(json_data, content_level_index)

                # Optional: save the structure for debugging
                if new_structure is not None:
                    logger.info("New structure generated")
                    count = 0
                    if isinstance(new_structure, list):
                        count = len(new_structure)
                    elif isinstance(new_structure, dict):
                        count = 1 if (new_structure.get("name") or new_structure.get("children")) else 0
                    
                    logger.info(f"Generated new structure with {count} top-level item(s)")
                else:
                    logger.warning("Generated structure is empty or None")

            except ValueError:
                logger.warning("Could not find a content level after the root level")
                logger.warning("No merging performed")
                new_structure = copy.deepcopy(json_data)

        elif depth_map and len(depth_map) == 1:
            logger.warning(f"Only one level (Level 0) found with {depth_map[0]} nodes. Cannot determine content level.")
            new_structure = copy.deepcopy(json_data)
        else:
            logger.warning("Could not generate depth map or depth map is empty")
            new_structure = None
            
        # Using the new structure, chunk the text
        if new_structure is None or not isinstance(new_structure, (list, dict)):
            logger.warning("No valid structure to chunk")
            return []
        
        # Extract content from the processed structure
        queue = deque([new_structure])
        content = []
        
        while queue:
            current_node = queue.popleft()
            if isinstance(current_node, list) and len(current_node) > 0:
                current_node = current_node[0]
                
            if isinstance(current_node, dict):
                if 'children' in current_node and not current_node['children']:
                    content_text = current_node.get('name', '')
                    content_length = len(content_text.split())
                    content.append({
                        'text': content_text,
                        'length': content_length
                    })
                    
                if 'children' in current_node and isinstance(current_node['children'], list):
                    for child in current_node['children']:
                        queue.append(child)
        
        # Convert to DataFrame for analysis
        if not content:
            logger.warning("No content extracted from structure")
            return []
            
        content_df = pd.DataFrame(content)
        
        # Find a threshold for "short" text based on length distribution
        threshold = content_df['length'].quantile(0.75)
        
        # Chunk the text, merging short sections
        chunked_data = []
        save = ""
        
        for _, row in content_df.iterrows():
            if save:
                text = save + "\n" + row['text']
                save = ""
            else:
                text = row['text']
                
            length = row['length']
            if length < threshold:
                save += "\n" + text.strip()
                continue
            else:
                chunked_block = text_splitter.split_text(text)
                
            if not chunked_block:
                logger.warning(f"Empty chunk generated for text: {text[:100]}...")
                continue
                
            chunked_data.extend(chunked_block)
            
        logger.info(f"Generated {len(chunked_data)} chunks")
        return chunked_data
        
    except Exception as e:
        logger.error(f"Error in chunk_text_v2: {str(e)}")
        raise TextChunkingError(f"Failed to chunk text: {str(e)}")


def process_pdf_to_structured_json(pdf_path: str, output_json_path: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Main orchestrator function for document structure extraction:
    1. Checks input path
    2. Extracts and prepares text elements
    3. Identifies and filters watermarks
    4. Sorts elements globally for hierarchy processing
    5. Calculates child spans based on height
    6. Builds the node hierarchy
    7. Generates the final JSON structure
    8. Optionally saves the JSON file
    
    Args:
        pdf_path: Path to the PDF file
        output_json_path: Optional path to save the structured JSON
        
    Returns:
        Tuple of (extracted elements, structured JSON)
    """
    try:
        from .utils import validate_pdf_path
        from .extract import extract_and_prepare_elements, identify_and_filter_watermarks
        
        start_time = time.time()
        logger.info(f"Starting processing for: {pdf_path}")

        # Validate input path
        validate_pdf_path(pdf_path)
        
        if output_json_path is None:
            output_json_path = os.path.splitext(pdf_path)[0] + "_structured.json"

        # 1. Extract and Prepare Elements
        elements = extract_and_prepare_elements(pdf_path)
        if not elements:
            logger.error("No text elements extracted or processed. Aborting.")
            raise StructureExtractionError("No text elements found in the PDF.")

        # 2. Identify and Filter Watermarks
        elements, _ = identify_and_filter_watermarks(elements)
        if not elements:
            logger.error("No content remaining after watermark removal. Aborting.")
            raise StructureExtractionError("No content remaining after watermark removal.")

        # 3. Sort All Elements Globally for Hierarchy Building
        logger.info("Sorting all elements for hierarchical processing")
        elements.sort(key=lambda x: (
            x.get("page_num", 0),
            x.get("occupy_space").y0 if x.get("occupy_space") else 0,
            x.get("occupy_space").x0 if x.get("occupy_space") else 0
        ))
        logger.info("Sorting complete")

        # 4. Calculate Hierarchy Markers (Child Spans)
        child_span_counts = calculate_child_spans(elements)

        # 5. Build Hierarchy Tree
        all_nodes, root_nodes = build_hierarchy(elements, child_span_counts)
        if not root_nodes:
            logger.error("No hierarchical structure could be determined. Aborting JSON generation.")
            raise StructureExtractionError("Failed to extract document hierarchy.")

        # 6. Generate JSON Structure
        output_data = generate_json_structure(root_nodes)

        # 7. Save the JSON File
        if output_json_path:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=4)
            logger.info(f"Saved structured JSON to: {output_json_path}")

        end_time = time.time()
        logger.info(f"Processing finished in {end_time - start_time:.2f} seconds")
        return elements, output_data
        
    except Exception as e:
        logger.error(f"Error in process_pdf_to_structured_json: {str(e)}")
        raise StructureExtractionError(f"Failed to process PDF to structured JSON: {str(e)}")
        
        
def get_time() -> float:
    """
    Helper function to measure execution time.
    
    Returns:
        Current time in seconds
    """
    return time.time()