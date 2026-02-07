"""Java token extraction using javalang AST parser."""

from typing import List
import logging

try:
    import javalang
    JAVALANG_AVAILABLE = True
except ImportError:
    JAVALANG_AVAILABLE = False
    javalang = None

from src.data_models import Token
from src.extractors.base_extractor import BaseTokenExtractor
from src.utils import read_file

logger = logging.getLogger(__name__)


class JavaTokenExtractor(BaseTokenExtractor):
    """Extracts semantic tokens from Java source code.
    
    Uses javalang to parse Java AST and extract:
    - Function/method names
    - Variable names
    - Class names
    - String and numeric literals
    - Conditional expressions
    """
    
    def __init__(self):
        """Initialize the Java extractor.
        
        Raises:
            RuntimeError: If javalang is not installed
        """
        super().__init__(
            language="java",
            supported_extensions=['.java']
        )
        
        if not JAVALANG_AVAILABLE:
            error_msg = (
                "javalang library not found. "
                "Please install it with: pip install javalang"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("Java token extractor initialized")
    
    def extract_tokens(self, file_path: str) -> List[Token]:
        """Extract all tokens from Java source file.
        
        Args:
            file_path: Path to the Java source file
            
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
        """Parse Java content and extract tokens using AST.
        
        Args:
            content: Java source code as string
            source_file: Source file path for token metadata
            
        Returns:
            List of Token objects
        """
        try:
            # Parse Java code into AST
            tree = javalang.parse.parse(content)
            
            tokens = []
            
            # Extract different token types
            tokens.extend(self._extract_methods(tree, source_file))
            tokens.extend(self._extract_variables(tree, source_file))
            tokens.extend(self._extract_classes(tree, source_file))
            tokens.extend(self._extract_literals(tree, source_file))
            
            logger.debug(f"Parsed {len(tokens)} tokens from Java content")
            return tokens
            
        except javalang.parser.JavaSyntaxError as e:
            logger.error(f"Java syntax error in {source_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing Java file {source_file}: {e}")
            return []
    
    def _extract_methods(self, tree, source_file: str) -> List[Token]:
        """Extract method names from AST.
        
        Args:
            tree: javalang AST tree
            source_file: Source file path
            
        Returns:
            List of Token objects for methods
        """
        methods = []
        
        for path, node in tree.filter(javalang.tree.MethodDeclaration):
            methods.append(Token(
                text=node.name,
                line=node.position.line if node.position else 0,
                column=node.position.column if node.position else 0,
                token_type="function",
                source_file=source_file
            ))
        
        logger.debug(f"Extracted {len(methods)} methods")
        return methods
    
    def _extract_variables(self, tree, source_file: str) -> List[Token]:
        """Extract variable names from AST.
        
        Args:
            tree: javalang AST tree
            source_file: Source file path
            
        Returns:
            List of Token objects for variables
        """
        variables = []
        
        # Extract field declarations (class variables)
        for path, node in tree.filter(javalang.tree.FieldDeclaration):
            for declarator in node.declarators:
                variables.append(Token(
                    text=declarator.name,
                    line=node.position.line if node.position else 0,
                    column=node.position.column if node.position else 0,
                    token_type="variable",
                    source_file=source_file
                ))
        
        # Extract local variable declarations
        for path, node in tree.filter(javalang.tree.LocalVariableDeclaration):
            for declarator in node.declarators:
                variables.append(Token(
                    text=declarator.name,
                    line=node.position.line if node.position else 0,
                    column=node.position.column if node.position else 0,
                    token_type="variable",
                    source_file=source_file
                ))
        
        logger.debug(f"Extracted {len(variables)} variables")
        return variables
    
    def _extract_classes(self, tree, source_file: str) -> List[Token]:
        """Extract class names from AST.
        
        Args:
            tree: javalang AST tree
            source_file: Source file path
            
        Returns:
            List of Token objects for classes
        """
        classes = []
        
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            classes.append(Token(
                text=node.name,
                line=node.position.line if node.position else 0,
                column=node.position.column if node.position else 0,
                token_type="class",
                source_file=source_file
            ))
        
        logger.debug(f"Extracted {len(classes)} classes")
        return classes
    
    def _extract_literals(self, tree, source_file: str) -> List[Token]:
        """Extract string and numeric literals from AST.
        
        Args:
            tree: javalang AST tree
            source_file: Source file path
            
        Returns:
            List of Token objects for literals
        """
        literals = []
        
        # Extract string literals
        for path, node in tree.filter(javalang.tree.Literal):
            if node.value and isinstance(node.value, str):
                # Only include non-trivial strings (length > 3)
                if len(node.value) > 3:
                    literals.append(Token(
                        text=node.value,
                        line=node.position.line if node.position else 0,
                        column=node.position.column if node.position else 0,
                        token_type="literal",
                        source_file=source_file
                    ))
        
        logger.debug(f"Extracted {len(literals)} literals")
        return literals
    
    def extract_conditionals(self, content: str, source_file: str) -> List[Token]:
        """Extract conditional expressions from Java code.
        
        This is a specialized method for extracting if/while/for conditions.
        
        Args:
            content: Java source code
            source_file: Source file path
            
        Returns:
            List of Token objects for conditionals
        """
        try:
            tree = javalang.parse.parse(content)
            conditionals = []
            
            # Extract if statements
            for path, node in tree.filter(javalang.tree.IfStatement):
                if node.condition:
                    conditionals.append(Token(
                        text=str(node.condition),
                        line=node.position.line if node.position else 0,
                        column=node.position.column if node.position else 0,
                        token_type="conditional",
                        source_file=source_file
                    ))
            
            # Extract while statements
            for path, node in tree.filter(javalang.tree.WhileStatement):
                if node.condition:
                    conditionals.append(Token(
                        text=str(node.condition),
                        line=node.position.line if node.position else 0,
                        column=node.position.column if node.position else 0,
                        token_type="conditional",
                        source_file=source_file
                    ))
            
            logger.debug(f"Extracted {len(conditionals)} conditionals")
            return conditionals
            
        except Exception as e:
            logger.error(f"Error extracting conditionals: {e}")
            return []
