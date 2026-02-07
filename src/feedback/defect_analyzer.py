"""Defect analyzer for parsing and preprocessing defect information."""

import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.feedback.data_models import DefectInfo
from src.utils import get_logger


class DefectAnalyzer:
    """Analyzer for parsing and preprocessing fuzzer defects."""
    
    def __init__(self):
        """Initialize defect analyzer."""
        self.logger = get_logger()
        
        # Defect type patterns
        self.defect_patterns = {
            'crash': r'(crash|segfault|sigsegv|sigabrt)',
            'assertion': r'(assertion|assert|failed)',
            'timeout': r'(timeout|hang|deadlock)',
            'memory_leak': r'(memory leak|memleak|leak)',
            'buffer_overflow': r'(buffer overflow|stack overflow|heap overflow)',
            'null_pointer': r'(null pointer|nullptr|segmentation fault)',
        }
        
        # Severity keywords
        self.severity_keywords = {
            'critical': ['crash', 'segfault', 'buffer overflow', 'remote code execution'],
            'high': ['assertion', 'memory leak', 'null pointer'],
            'medium': ['timeout', 'hang', 'warning'],
            'low': ['info', 'notice']
        }
    
    def parse_defect(
        self,
        test_case: str,
        error_message: str,
        stack_trace: Optional[str] = None,
        defect_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DefectInfo:
        """Parse defect information into structured format.
        
        Args:
            test_case: Test case that triggered the defect
            error_message: Error message from fuzzer
            stack_trace: Stack trace if available
            defect_id: Optional defect ID (auto-generated if not provided)
            metadata: Additional metadata
            
        Returns:
            DefectInfo object with parsed information
        """
        # Auto-generate defect ID if not provided
        if defect_id is None:
            import hashlib
            content = f"{test_case}{error_message}"
            defect_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        # Identify defect type
        defect_type = self.identify_defect_type(error_message, stack_trace)
        
        # Determine severity
        severity = self.determine_severity(error_message, defect_type)
        
        # Create DefectInfo
        defect_info = DefectInfo(
            defect_id=defect_id,
            defect_type=defect_type,
            test_case=test_case,
            error_message=error_message,
            stack_trace=stack_trace,
            severity=severity,
            metadata=metadata or {}
        )
        
        self.logger.info(f"Parsed defect: {defect_id} (type={defect_type}, severity={severity})")
        
        return defect_info
    
    def identify_defect_type(
        self,
        error_message: str,
        stack_trace: Optional[str] = None
    ) -> str:
        """Identify the type of defect based on error message and stack trace.
        
        Args:
            error_message: Error message
            stack_trace: Stack trace if available
            
        Returns:
            Defect type string
        """
        text = (error_message + " " + (stack_trace or "")).lower()
        
        # Check each pattern
        for defect_type, pattern in self.defect_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                return defect_type
        
        # Default to generic error
        return "unknown_error"
    
    def determine_severity(self, error_message: str, defect_type: str) -> str:
        """Determine severity level of the defect.
        
        Args:
            error_message: Error message
            defect_type: Type of defect
            
        Returns:
            Severity level (critical, high, medium, low)
        """
        text = error_message.lower()
        
        # Check severity keywords
        for severity, keywords in self.severity_keywords.items():
            for keyword in keywords:
                if keyword in text or keyword == defect_type:
                    return severity
        
        # Default to medium
        return "medium"
    
    def extract_context(
        self,
        document: str,
        line_number: int,
        context_lines: int = 5
    ) -> str:
        """Extract context around a specific line in the document.
        
        Args:
            document: Document content
            line_number: Target line number
            context_lines: Number of lines before and after to include
            
        Returns:
            Context string
        """
        lines = document.split('\n')
        
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        context = '\n'.join(lines[start:end])
        return context
    
    def extract_relevant_tokens(
        self,
        tokens: List[Any],
        error_message: str,
        top_n: int = 10
    ) -> List[Any]:
        """Extract tokens that are potentially relevant to the defect.
        
        Args:
            tokens: List of Token objects
            error_message: Error message
            top_n: Number of top tokens to return
            
        Returns:
            List of relevant tokens
        """
        # Extract keywords from error message
        keywords = self._extract_keywords(error_message)
        
        # Score tokens by relevance
        scored_tokens = []
        for token in tokens:
            score = self._calculate_relevance(token, keywords)
            scored_tokens.append((token, score))
        
        # Sort by score and return top N
        scored_tokens.sort(key=lambda x: x[1], reverse=True)
        return [token for token, score in scored_tokens[:top_n]]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text.
        
        Args:
            text: Input text
            
        Returns:
            List of keywords
        """
        # Simple keyword extraction (can be improved with NLP)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        
        return keywords
    
    def _calculate_relevance(self, token: Any, keywords: List[str]) -> float:
        """Calculate relevance score for a token.
        
        Args:
            token: Token object
            keywords: List of keywords
            
        Returns:
            Relevance score
        """
        token_text = token.text.lower()
        score = 0.0
        
        # Check if token text contains any keywords
        for keyword in keywords:
            if keyword in token_text:
                score += 1.0
        
        # Boost score for security-related tokens
        security_keywords = ['auth', 'valid', 'check', 'verify', 'secure']
        for keyword in security_keywords:
            if keyword in token_text:
                score += 0.5
        
        return score
    
    def summarize_defect(self, defect_info: DefectInfo) -> str:
        """Create a human-readable summary of the defect.
        
        Args:
            defect_info: DefectInfo object
            
        Returns:
            Summary string
        """
        summary = f"""
Defect Summary:
--------------
ID: {defect_info.defect_id}
Type: {defect_info.defect_type}
Severity: {defect_info.severity}
Timestamp: {defect_info.timestamp}

Error Message:
{defect_info.error_message}

Test Case:
{defect_info.test_case[:200]}...
"""
        
        if defect_info.stack_trace:
            summary += f"\nStack Trace:\n{defect_info.stack_trace[:500]}..."
        
        return summary
