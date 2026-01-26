#!/usr/bin/env python3
"""
Batch Writer Tool - Ensures files are written in batches of maximum 150 lines
This tool enforces the strict 150-line limit per write operation.
"""

import re
from typing import List, Tuple


class BatchWriter:
    """Handles splitting and writing content in batches of maximum 150 lines."""
    
    MAX_LINES_PER_BATCH = 150
    MIN_BATCH_SIZE = 50  # Minimum lines for a batch to avoid too many small batches
    
    @staticmethod
    def count_lines(content: str) -> int:
        """Count the number of lines in content."""
        return len(content.split('\n'))
    
    @staticmethod
    def find_logical_split_point(lines: List[str], start_idx: int, max_lines: int = MAX_LINES_PER_BATCH) -> int:
        """
        Find the best place to split content at logical boundaries.
        
        Args:
            lines: List of content lines
            start_idx: Starting index for the batch
            max_lines: Maximum lines for this batch
            
        Returns:
            Index where the split should occur
        """
        total_lines = len(lines)
        end_idx = min(start_idx + max_lines, total_lines)
        
        # If this is the last batch, return total_lines
        if end_idx >= total_lines:
            return total_lines
        
        # Look backward for logical boundaries (empty lines, end of blocks)
        # Search in the last 30% of the batch
        search_start = max(start_idx + max_lines // 2, start_idx)
        
        for i in range(end_idx, search_start, -1):
            if i >= total_lines:
                continue
            
            line = lines[i].strip()
            
            # Good split points:
            # 1. Empty lines
            # 2. End of function/class definition (line ending with ':')
            # 3. End of import blocks
            # 4. After comment blocks
            if line == '':
                return i + 1
            elif line.endswith(':') and not line.startswith('#'):
                return i + 1
            elif line.startswith('import ') or line.startswith('from '):
                # Look ahead to see if this is end of import block
                if i + 1 < total_lines and not lines[i + 1].strip().startswith(('import', 'from')):
                    return i + 1
        
        # Fallback: use max_lines
        return end_idx
    
    @staticmethod
    def analyze_batches(content: str) -> Tuple[int, List[Tuple[int, int]]]:
        """
        Analyze content and determine batch boundaries.
        
        Args:
            content: The content to split
            
        Returns:
            Tuple of (total_lines, list of (start_idx, end_idx) tuples)
        """
        lines = content.split('\n')
        total_lines = len(lines)
        batches = []
        
        current_idx = 0
        while current_idx < total_lines:
            split_idx = BatchWriter.find_logical_split_point(lines, current_idx)
            batches.append((current_idx, split_idx))
            current_idx = split_idx
        
        return total_lines, batches
    
    @staticmethod
    def split_content(content: str) -> List[str]:
        """
        Split content into batches of maximum 150 lines.
        
        Args:
            content: The content to split
            
        Returns:
            List of content batches
        """
        lines = content.split('\n')
        total_lines, batches = BatchWriter.analyze_batches(content)
        
        if total_lines <= BatchWriter.MAX_LINES_PER_BATCH:
            return [content]
        
        result = []
        for start_idx, end_idx in batches:
            batch_lines = lines[start_idx:end_idx]
            result.append('\n'.join(batch_lines))
        
        return result
    
    @staticmethod
    def format_batch_summary(content: str) -> str:
        """
        Generate a summary of the batch writing plan.
        
        Args:
            content: The content to analyze
            
        Returns:
            Formatted summary string
        """
        total_lines, batches = BatchWriter.analyze_batches(content)
        batch_count = len(batches)
        
        summary = []
        summary.append(f"📊 Content Analysis:")
        summary.append(f"   Total lines: {total_lines}")
        summary.append(f"   Batches needed: {batch_count}")
        summary.append(f"")
        summary.append(f"📦 Batch Breakdown:")
        
        for i, (start_idx, end_idx) in enumerate(batches, 1):
            line_count = end_idx - start_idx
            summary.append(f"   Batch {i}: Lines {start_idx + 1}-{end_idx} ({line_count} lines)")
        
        return '\n'.join(summary)
    
    @staticmethod
    def validate_integrity(original_content: str, written_content: str) -> bool:
        """
        Validate that written content matches original content.
        
        Args:
            original_content: The original content that was to be written
            written_content: The content read back from file
            
        Returns:
            True if content matches, False otherwise
        """
        original_lines = original_content.split('\n')
        written_lines = written_content.split('\n')
        
        if len(original_lines) != len(written_lines):
            return False
        
        # Compare line by line
        for orig, written in zip(original_lines, written_lines):
            if orig != written:
                return False
        
        return True
    
    @staticmethod
    def get_edit_context(content: str, context_lines: int = 10) -> str:
        """
        Get the last N lines of content for use as edit context.
        
        Args:
            content: The content to get context from
            context_lines: Number of lines to include
            
        Returns:
            Last N lines of content
        """
        lines = content.split('\n')
        if len(lines) <= context_lines:
            return content
        return '\n'.join(lines[-context_lines:])


# Convenience functions for common operations

def check_write_size(content: str) -> Tuple[bool, int]:
    """
    Check if content exceeds the 150-line limit.
    
    Args:
        content: Content to check
        
    Returns:
        Tuple of (exceeds_limit, line_count)
    """
    line_count = BatchWriter.count_lines(content)
    exceeds_limit = line_count > BatchWriter.MAX_LINES_PER_BATCH
    return exceeds_limit, line_count


def plan_file_write(content: str) -> str:
    """
    Plan how a file should be written in batches.
    
    Args:
        content: Content to write
        
    Returns:
        Formatted plan showing batch breakdown
    """
    return BatchWriter.format_batch_summary(content)


if __name__ == "__main__":
    # Example usage and testing
    test_content = "\n".join([f"# Line {i}" for i in range(1, 501)])
    
    print("=== Batch Writer Tool Test ===\n")
    print(plan_file_write(test_content))
    print("\n=== Validating Integrity ===")
    batches = BatchWriter.split_content(test_content)
    reconstructed = "\n".join(batches)
    is_valid = BatchWriter.validate_integrity(test_content, reconstructed)
    print(f"Integrity check: {'✅ PASSED' if is_valid else '❌ FAILED'}")
