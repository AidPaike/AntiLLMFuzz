"""Python token extraction using AST parser."""

import ast
from typing import List
import logging

from src.data_models import Token
from src.extractors.base_extractor import BaseTokenExtractor
from src.utils import read_file

logger = logging.getLogger(__name__)


class PythonTokenExtractor(BaseTokenExtractor):
    """Extracts semantic tokens from Python source code.
    
    Uses Python's built-in ast module to parse and extract:
    - Function names
    - Class names
    - Variable names
    - String literals
    - Import statements
    """
    
    def __init__(self):
        """Initialize the Python extractor."""
        super().__init__(
            language="python",
            supported_extensions=['.py', '.pyw']
        )
        logger.info("Python token extractor initialized")
    
    def extract_tokens(self, file_path: str) -> List[Token]:
        """Extract all tokens from Python source file.
        
        Args:
            file_path: Path to the Python source file
            
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
        """Parse Python content and extract tokens using AST.
        
        Args:
            content: Python source code as string
            source_file: Source file path for token metadata
            
        Returns:
            List of Token objects
        """
        try:
            # Parse Python code into AST
            tree = ast.parse(content, filename=source_file)
            
            tokens = []
            
            # Extract different token types
            tokens.extend(self._extract_functions(tree, source_file))
            tokens.extend(self._extract_classes(tree, source_file))
            tokens.extend(self._extract_variables(tree, source_file))
            tokens.extend(self._extract_imports(tree, source_file))
            tokens.extend(self._extract_literals(tree, source_file))
            
            logger.debug(f"Parsed {len(tokens)} tokens from Python content")
            return tokens
            
        except SyntaxError as e:
            logger.error(f"Python syntax error in {source_file}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing Python file {source_file}: {e}")
            return []
    
    def _extract_functions(self, tree: ast.AST, source_file: str) -> List[Token]:
        """Extract function names from AST.
        
        Args:
            tree: Python AST tree
            source_file: Source file path
            
        Returns:
            List of Token objects for functions
        """
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(Token(
                    text=node.name,
                    line=node.lineno,
                    column=node.col_offset,
                    token_type="function",
                    source_file=source_file
                ))
        
        logger.debug(f"Extracted {len(functions)} functions")
        return functions
    
    def _extract_classes(self, tree: ast.AST, source_file: str) -> List[Token]:
        """Extract class names from AST.
        
        Args:
            tree: Python AST tree
            source_file: Source file path
            
        Returns:
            List of Token objects for classes
        """
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(Token(
                    text=node.name,
                    line=node.lineno,
                    column=node.col_offset,
                    token_type="class",
                    source_file=source_file
                ))
        
        logger.debug(f"Extracted {len(classes)} classes")
        return classes
    
    def _extract_variables(self, tree: ast.AST, source_file: str) -> List[Token]:
        """Extract variable names from AST.
        
        Args:
            tree: Python AST tree
            source_file: Source file path
            
        Returns:
            List of Token objects for variables
        """
        variables = []
        
        for node in ast.walk(tree):
            # Extract assignment targets
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(Token(
                            text=target.id,
                            line=target.lineno,
                            column=target.col_offset,
                            token_type="variable",
                            source_file=source_file
                        ))
        
        logger.debug(f"Extracted {len(variables)} variables")
        return variables
    
    def _extract_imports(self, tree: ast.AST, source_file: str) -> List[Token]:
        """Extract import statements from AST.
        
        Args:
            tree: Python AST tree
            source_file: Source file path
            
        Returns:
            List of Token objects for imports
        """
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(Token(
                        text=alias.name,
                        line=node.lineno,
                        column=node.col_offset,
                        token_type="import",
                        source_file=source_file
                    ))
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(Token(
                        text=node.module,
                        line=node.lineno,
                        column=node.col_offset,
                        token_type="import",
                        source_file=source_file
                    ))
        
        logger.debug(f"Extracted {len(imports)} imports")
        return imports
    
    def _extract_literals(self, tree: ast.AST, source_file: str) -> List[Token]:
        """Extract string literals from AST.
        
        Args:
            tree: Python AST tree
            source_file: Source file path
            
        Returns:
            List of Token objects for literals
        """
        literals = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant):
                # Only include string literals with length > 3
                if isinstance(node.value, str) and len(node.value) > 3:
                    literals.append(Token(
                        text=node.value,
                        line=node.lineno,
                        column=node.col_offset,
                        token_type="literal",
                        source_file=source_file
                    ))
        
        logger.debug(f"Extracted {len(literals)} literals")
        return literals
    
    def extract_decorators(self, content: str, source_file: str) -> List[Token]:
        """Extract decorator names from Python code.
        
        This is a specialized method for extracting decorators.
        
        Args:
            content: Python source code
            source_file: Source file path
            
        Returns:
            List of Token objects for decorators
        """
        try:
            tree = ast.parse(content, filename=source_file)
            decorators = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    for decorator in node.decorator_list:
                        if isinstance(decorator, ast.Name):
                            decorators.append(Token(
                                text=decorator.id,
                                line=decorator.lineno,
                                column=decorator.col_offset,
                                token_type="decorator",
                                source_file=source_file
                            ))
            
            logger.debug(f"Extracted {len(decorators)} decorators")
            return decorators
            
        except Exception as e:
            logger.error(f"Error extracting decorators: {e}")
            return []
