"""
Basic tests for the book pipeline.
"""

import os
import pytest
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.utils import MedianFinder, Node, validate_pdf_path
from src.models import TextBlock, SummaryAndEvidence, EvidenceSnippet
from src.exceptions import ValidationError


def test_median_finder():
    """Test MedianFinder class."""
    finder = MedianFinder()
    finder.addNum(1)
    assert finder.findMedian() == 1.0
    
    finder.addNum(3)
    assert finder.findMedian() == 2.0
    
    finder.addNum(2)
    assert finder.findMedian() == 2.0


def test_node_class():
    """Test Node class."""
    content = {"content": "Test content", "page_num": 0}
    node = Node(content)
    assert node.content == content
    assert len(node.children) == 0
    
    child_content = {"content": "Child content", "page_num": 1}
    child_node = Node(child_content)
    node.add_child(child_node)
    
    assert len(node.children) == 1
    assert node.children[0].content == child_content


def test_validate_pdf_path():
    """Test validate_pdf_path function."""
    # Should raise ValidationError for non-existent path
    with pytest.raises(ValidationError):
        validate_pdf_path("non_existent_file.pdf")
    
    # Should raise ValidationError for empty path
    with pytest.raises(ValidationError):
        validate_pdf_path("")
    
    # Create a temporary file for testing
    temp_file = "temp_test_file.txt"
    with open(temp_file, 'w') as f:
        f.write("test")
    
    # Should work for existing file
    assert validate_pdf_path(temp_file) is True
    
    # Clean up
    os.remove(temp_file)


def test_evidence_snippet_model():
    """Test EvidenceSnippet model."""
    # Test valid creation
    snippet = EvidenceSnippet(verbatim_text="This is a test snippet")
    assert snippet.verbatim_text == "This is a test snippet"


def test_summary_and_evidence_model():
    """Test SummaryAndEvidence model."""
    # Test valid creation
    model = SummaryAndEvidence(
        summary="Test summary",
        evidence=[
            EvidenceSnippet(verbatim_text="Evidence 1"),
            EvidenceSnippet(verbatim_text="Evidence 2")
        ]
    )
    
    assert model.summary == "Test summary"
    assert len(model.evidence) == 2
    assert model.evidence[0].verbatim_text == "Evidence 1"
    assert model.evidence[1].verbatim_text == "Evidence 2"


def test_text_block_model():
    """Test TextBlock model."""
    # Test valid creation
    block = TextBlock(
        page=1,
        text="Sample text",
        bbox=(0, 0, 100, 20)
    )
    
    assert block.page == 1
    assert block.text == "Sample text"
    assert block.bbox == (0, 0, 100, 20)
    
    # Test default values
    assert block.block_no is None
    assert block.block_type is None