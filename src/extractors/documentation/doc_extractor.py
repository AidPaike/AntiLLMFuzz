"""Documentation token extraction using spaCy NLP."""

import spacy
from typing import List
import logging

from src.data_models import Token
from src.extractors.base_extractor import BaseTokenExtractor
from src.utils import read_file

logger = logging.getLogger(__name__)


class DocumentationTokenExtractor(BaseTokenExtractor):
    """Extracts semantic tokens from documentation files (Markdown/JavaDoc/Text).
    
    Uses spaCy for NLP processing to extract:
    - Nouns (technical terms)
    - Verbs (actions)
    - Technical phrases (noun chunks)
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the documentation extractor.
        
        Args:
            model_name: spaCy model name to use
            
        Raises:
            RuntimeError: If spaCy model is not installed
        """
        super().__init__(
            language="documentation",
            supported_extensions=['.md', '.txt', '.rst', '.adoc', '.javadoc']
        )
        
        self.model_name = model_name
        
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            error_msg = (
                f"spaCy model '{model_name}' not found. "
                f"Please install it with: python -m spacy download {model_name}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def extract_tokens(self, file_path: str) -> List[Token]:
        """Extract all tokens from documentation file.
        
        Args:
            file_path: Path to the documentation file
            
        Returns:
            List of Token objects
        """
        # Validate file
        if not self.validate_file(file_path):
            logger.error(f"File validation failed: {file_path}")
            return []
        
        # Read content
        content = read_file(file_path)
        if content is None:
            logger.error(f"Failed to read file: {file_path}")
            return []
        
        # Parse and extract
        tokens = self.parse_content(content, file_path)
        
        logger.info(f"Extracted {len(tokens)} tokens from {file_path}")
        return tokens
    
    def parse_content(self, content: str, source_file: str) -> List[Token]:
        """Parse content and extract tokens using spaCy.
        
        Args:
            content: File content as string
            source_file: Source file path for token metadata
            
        Returns:
            List of Token objects
        """
        # Process with spaCy
        doc = self.nlp(content)
        
        tokens = []
        
        # Extract different token types
        tokens.extend(self._extract_nouns(doc, source_file))
        tokens.extend(self._extract_verbs(doc, source_file))
        tokens.extend(self._extract_technical_phrases(doc, source_file))
        
        logger.debug(f"Parsed {len(tokens)} tokens from content")
        return tokens
    
    def _extract_nouns(self, doc, source_file: str) -> List[Token]:
        """Extract nouns from tagged text.
        
        Args:
            doc: spaCy Doc object
            source_file: Source file path
            
        Returns:
            List of Token objects for nouns
        """
        nouns = []
        for token in doc:
            if token.pos_ == "NOUN" and not token.is_stop:
                nouns.append(Token(
                    text=token.text,
                    line=token.i,  # Token index as line number
                    column=token.idx,
                    token_type="noun",
                    source_file=source_file
                ))
        
        logger.debug(f"Extracted {len(nouns)} nouns")
        return nouns
    
    def _extract_verbs(self, doc, source_file: str) -> List[Token]:
        """Extract verbs from tagged text.
        
        Args:
            doc: spaCy Doc object
            source_file: Source file path
            
        Returns:
            List of Token objects for verbs
        """
        verbs = []
        for token in doc:
            if token.pos_ == "VERB" and not token.is_stop:
                verbs.append(Token(
                    text=token.text,
                    line=token.i,
                    column=token.idx,
                    token_type="verb",
                    source_file=source_file
                ))
        
        logger.debug(f"Extracted {len(verbs)} verbs")
        return verbs
    
    def _extract_technical_phrases(self, doc, source_file: str) -> List[Token]:
        """Extract technical phrases based on noun chunks.
        
        Args:
            doc: spaCy Doc object
            source_file: Source file path
            
        Returns:
            List of Token objects for technical phrases
        """
        phrases = []
        for chunk in doc.noun_chunks:
            # Only include multi-word phrases
            if len(chunk.text.split()) > 1:
                phrases.append(Token(
                    text=chunk.text,
                    line=chunk.start,
                    column=chunk.start_char,
                    token_type="phrase",
                    source_file=source_file
                ))
        
        logger.debug(f"Extracted {len(phrases)} technical phrases")
        return phrases
    
    def extract_specific_pos(self, content: str, source_file: str, 
                           pos_tags: List[str]) -> List[Token]:
        """Extract tokens with specific POS tags.
        
        This is a utility method for extracting custom token types.
        
        Args:
            content: File content
            source_file: Source file path
            pos_tags: List of POS tags to extract (e.g., ['ADJ', 'ADV'])
            
        Returns:
            List of Token objects
        """
        doc = self.nlp(content)
        tokens = []
        
        for token in doc:
            if token.pos_ in pos_tags and not token.is_stop:
                tokens.append(Token(
                    text=token.text,
                    line=token.i,
                    column=token.idx,
                    token_type=token.pos_.lower(),
                    source_file=source_file
                ))
        
        logger.debug(f"Extracted {len(tokens)} tokens with POS tags {pos_tags}")
        return tokens
