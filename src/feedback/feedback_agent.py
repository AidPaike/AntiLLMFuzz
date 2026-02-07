"""Feedback agent using LLM to analyze defects and identify critical locations."""

import json
from typing import List, Dict, Any, Optional

from src.feedback.data_models import DefectInfo, FeedbackAnalysis, CriticalLocation
from src.data_models import Token
from src.utils import get_llm_client, get_logger


class FeedbackAgent:
    """LLM-powered agent for analyzing defects and identifying critical locations."""
    
    # Prompt template for defect analysis
    ANALYSIS_PROMPT = """You are an expert security researcher analyzing fuzzer defects to identify which parts of documentation are most likely responsible for causing the defect.

## Task
Analyze the defect information and identify critical locations in the documentation that should be targeted for perturbation.

## Defect Information

**Defect ID**: {defect_id}
**Defect Type**: {defect_type}
**Severity**: {severity}

**Error Message**:
```
{error_message}
```

**Test Case** (that triggered the defect):
```
{test_case}
```

{stack_trace_section}

## Documentation Context

**Original Document** (first 1000 chars):
```
{document_preview}
```

## Available Tokens

{tokens_info}

## Your Analysis

Please analyze this defect and provide:

1. **Root Cause**: What likely caused this defect? (1-2 sentences)
2. **Critical Locations**: Which tokens/lines are most relevant? (top 3-5)
3. **Reasoning**: Why are these locations critical? (2-3 sentences)
4. **Confidence**: How confident are you in this analysis? (0.0-1.0)
5. **Suggestions**: What perturbation strategies should be applied?

## Output Format

Respond with ONLY a valid JSON object (no markdown, no code blocks):

{{
  "root_cause": "Brief explanation of what caused the defect",
  "critical_locations": [
    {{
      "line_number": 10,
      "token_text": "authentication",
      "token_type": "noun",
      "relevance_score": 0.95,
      "reason": "This token describes a security-critical concept that the fuzzer misinterpreted",
      "suggested_strategies": ["tokenization_drift", "lexical_disguise"]
    }}
  ],
  "reasoning": "Detailed step-by-step reasoning about why these locations are critical",
  "confidence": 0.85,
  "suggestions": [
    "Focus on security-related tokens",
    "Apply stronger perturbations to validation logic"
  ]
}}
"""
    
    def __init__(self, llm_client=None, temperature: float = 0.3, max_tokens: int = 2000):
        """Initialize feedback agent.
        
        Args:
            llm_client: LLM client instance (creates new one if not provided)
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens for LLM response
        """
        self.llm_client = llm_client or get_llm_client()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = get_logger()
    
    def analyze_defect(
        self,
        defect_info: DefectInfo,
        document: str,
        tokens: List[Token],
        max_tokens_to_show: int = 20
    ) -> FeedbackAnalysis:
        """Analyze a defect using LLM to identify critical locations.
        
        Args:
            defect_info: Information about the defect
            document: Original document content
            tokens: List of extracted tokens
            max_tokens_to_show: Maximum number of tokens to show in prompt
            
        Returns:
            FeedbackAnalysis with LLM's analysis results
        """
        self.logger.info(f"Analyzing defect {defect_info.defect_id} with LLM...")
        
        # Prepare prompt
        prompt = self._prepare_prompt(defect_info, document, tokens, max_tokens_to_show)
        
        try:
            # Call LLM
            response = self.llm_client.simple_completion(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse response
            analysis = self._parse_response(response, defect_info.defect_id)
            
            self.logger.info(
                f"Analysis complete: {len(analysis.critical_locations)} critical locations identified "
                f"(confidence={analysis.confidence:.2f})"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(defect_info, tokens)
    
    def _prepare_prompt(
        self,
        defect_info: DefectInfo,
        document: str,
        tokens: List[Token],
        max_tokens_to_show: int
    ) -> str:
        """Prepare the prompt for LLM analysis.
        
        Args:
            defect_info: Defect information
            document: Document content
            tokens: List of tokens
            max_tokens_to_show: Maximum tokens to include
            
        Returns:
            Formatted prompt string
        """
        # Prepare stack trace section
        stack_trace_section = ""
        if defect_info.stack_trace:
            stack_trace_section = f"""
**Stack Trace**:
```
{defect_info.stack_trace[:500]}...
```
"""
        
        # Prepare document preview
        document_preview = document[:1000]
        if len(document) > 1000:
            document_preview += "\n... (truncated)"
        
        # Prepare tokens info
        top_tokens = sorted(tokens, key=lambda t: t.priority_score, reverse=True)[:max_tokens_to_show]
        tokens_info = "Top tokens by priority:\n"
        for i, token in enumerate(top_tokens, 1):
            tokens_info += f"{i}. Line {token.line}: '{token.text}' ({token.token_type}, priority={token.priority_score:.1f})\n"
        
        # Format prompt
        prompt = self.ANALYSIS_PROMPT.format(
            defect_id=defect_info.defect_id,
            defect_type=defect_info.defect_type,
            severity=defect_info.severity,
            error_message=defect_info.error_message,
            test_case=defect_info.test_case[:500],  # Truncate long test cases
            stack_trace_section=stack_trace_section,
            document_preview=document_preview,
            tokens_info=tokens_info
        )
        
        return prompt
    
    def _parse_response(self, response: str, defect_id: str) -> FeedbackAnalysis:
        """Parse LLM response into FeedbackAnalysis object.
        
        Args:
            response: LLM response string
            defect_id: Defect ID
            
        Returns:
            FeedbackAnalysis object
        """
        try:
            # Try to extract JSON from response
            # Sometimes LLM wraps JSON in markdown code blocks
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith('```'):
                lines = response.split('\n')
                response = '\n'.join(lines[1:-1])  # Remove first and last lines
            
            # Parse JSON
            data = json.loads(response)
            
            # Create CriticalLocation objects
            critical_locations = []
            for loc_data in data.get('critical_locations', []):
                location = CriticalLocation(
                    line_number=loc_data.get('line_number', 0),
                    token_text=loc_data.get('token_text', ''),
                    token_type=loc_data.get('token_type', 'unknown'),
                    relevance_score=loc_data.get('relevance_score', 0.5),
                    reason=loc_data.get('reason', ''),
                    suggested_strategies=loc_data.get('suggested_strategies', [])
                )
                critical_locations.append(location)
            
            # Create FeedbackAnalysis
            analysis = FeedbackAnalysis(
                defect_id=defect_id,
                root_cause=data.get('root_cause', 'Unknown'),
                critical_locations=critical_locations,
                reasoning=data.get('reasoning', ''),
                confidence=data.get('confidence', 0.5),
                suggestions=data.get('suggestions', [])
            )
            
            return analysis
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            self.logger.debug(f"Response: {response[:500]}")
            raise ValueError(f"Invalid JSON response from LLM: {e}")
    
    def _create_fallback_analysis(
        self,
        defect_info: DefectInfo,
        tokens: List[Token]
    ) -> FeedbackAnalysis:
        """Create a fallback analysis when LLM fails.
        
        Args:
            defect_info: Defect information
            tokens: List of tokens
            
        Returns:
            Basic FeedbackAnalysis based on heuristics
        """
        self.logger.warning("Creating fallback analysis (LLM failed)")
        
        # Use top priority tokens as critical locations
        top_tokens = sorted(tokens, key=lambda t: t.priority_score, reverse=True)[:5]
        
        critical_locations = []
        for token in top_tokens:
            location = CriticalLocation(
                line_number=token.line,
                token_text=token.text,
                token_type=token.token_type,
                relevance_score=min(token.priority_score / 10.0, 1.0),
                reason="High priority token (fallback heuristic)",
                suggested_strategies=["tokenization_drift", "lexical_disguise"]
            )
            critical_locations.append(location)
        
        return FeedbackAnalysis(
            defect_id=defect_info.defect_id,
            root_cause="Unable to analyze (LLM unavailable)",
            critical_locations=critical_locations,
            reasoning="Fallback analysis based on token priorities",
            confidence=0.3,
            suggestions=["Use high-priority tokens", "Apply standard perturbations"]
        )
    
    def batch_analyze(
        self,
        defects: List[DefectInfo],
        document: str,
        tokens: List[Token]
    ) -> List[FeedbackAnalysis]:
        """Analyze multiple defects in batch.
        
        Args:
            defects: List of defect information
            document: Document content
            tokens: List of tokens
            
        Returns:
            List of FeedbackAnalysis results
        """
        analyses = []
        
        for defect in defects:
            try:
                analysis = self.analyze_defect(defect, document, tokens)
                analyses.append(analysis)
            except Exception as e:
                self.logger.error(f"Failed to analyze defect {defect.defect_id}: {e}")
                # Add fallback analysis
                analyses.append(self._create_fallback_analysis(defect, tokens))
        
        return analyses
