"""Document processor for extracting API specifications from documentation."""

import re
import html
from typing import List, Dict, Any, Optional, Tuple
from src.fuzzer.data_models import DocumentStructure, APISpec, ParameterSpec
from src.utils.logger import get_logger


class DocumentProcessor:
    """Processes documentation to extract API specifications and structure."""
    
    def __init__(self):
        """Initialize document processor."""
        self.logger = get_logger("DocumentProcessor")
        
        # Enhanced regex patterns for comprehensive API extraction
        self.function_patterns = {
            'python': re.compile(
                r'(?:def|async\s+def)\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:',
                re.IGNORECASE | re.MULTILINE
            ),
            'javascript': re.compile(
                r'(?:function\s+(\w+)|(\w+)\s*=\s*(?:function|\([^)]*\)\s*=>))\s*\(([^)]*)\)',
                re.IGNORECASE | re.MULTILINE
            ),
            'java': re.compile(
                r'(?:public|private|protected)?\s*(?:static)?\s*(\w+)\s+(\w+)\s*\(([^)]*)\)',
                re.IGNORECASE | re.MULTILINE
            ),
            'markdown': re.compile(
                r'#+\s*(\w+)\s*\(([^)]*)\)(?:\s*->\s*(\w+))?',
                re.IGNORECASE | re.MULTILINE
            ),
            'generic': re.compile(
                r'(?:def|function|class|interface|method)\s+(\w+)\s*\([^)]*\)',
                re.IGNORECASE | re.MULTILINE
            )
        }
        
        # Enhanced parameter patterns for different languages
        self.parameter_patterns = {
            'python': re.compile(
                r'(\w+)\s*:\s*([^,=\)]+)(?:\s*=\s*([^,\)]+))?',
                re.IGNORECASE
            ),
            'javascript': re.compile(
                r'(\w+)(?:\s*:\s*([^,=\)]+))?(?:\s*=\s*([^,\)]+))?',
                re.IGNORECASE
            ),
            'java': re.compile(
                r'([^,\s]+)\s+(\w+)(?:\s*=\s*([^,\)]+))?',
                re.IGNORECASE
            ),
            'generic': re.compile(
                r'(\w+)\s*(?::\s*(\w+))?(?:\s*=\s*([^,\)]+))?',
                re.IGNORECASE
            )
        }
        
        # API section detection patterns
        self.api_section_patterns = [
            re.compile(r'#+\s*(API|Functions?|Methods?|Endpoints?)\s*\n', re.IGNORECASE | re.MULTILINE),
            re.compile(r'#+\s*(Reference|Documentation|Usage)\s*\n', re.IGNORECASE | re.MULTILINE),
            re.compile(r'#+\s*(Interface|Operations|Commands)\s*\n', re.IGNORECASE | re.MULTILINE)
        ]
        
        # Parameter constraint patterns
        self.constraint_patterns = {
            'range': re.compile(r'(?:between|from)\s+(\d+(?:\.\d+)?)\s+(?:to|and)\s+(\d+(?:\.\d+)?)', re.IGNORECASE),
            'min_max': re.compile(r'(?:min|minimum):\s*(\d+(?:\.\d+)?)|(?:max|maximum):\s*(\d+(?:\.\d+)?)', re.IGNORECASE),
            'length': re.compile(r'(?:length|size)\s*(?:of\s*)?(?:between\s+)?(\d+)(?:\s*(?:to|-)\s*(\d+))?', re.IGNORECASE),
            'format': re.compile(r'format:\s*([^\n,]+)', re.IGNORECASE),
            'enum': re.compile(r'(?:one of|values?):\s*([^\n]+)', re.IGNORECASE)
        }
        
        # Example extraction patterns
        self.example_patterns = [
            re.compile(r'(?:example|sample|usage):\s*\n(.*?)(?=\n\n|\n#|$)', re.IGNORECASE | re.DOTALL),
            re.compile(r'```(?:\w+)?\n(.*?)\n```', re.DOTALL),
            re.compile(r'`([^`]+)`', re.IGNORECASE)
        ]
        
        # Enhanced type mapping with more comprehensive coverage
        self.type_mapping = {
            'str': 'string', 'string': 'string', 'text': 'string',
            'int': 'integer', 'integer': 'integer', 'number': 'number',
            'float': 'number', 'double': 'number', 'decimal': 'number',
            'bool': 'boolean', 'boolean': 'boolean',
            'list': 'array', 'array': 'array', 'collection': 'array',
            'dict': 'object', 'object': 'object', 'map': 'object',
            'file': 'file', 'binary': 'binary', 'blob': 'binary',
            'date': 'date', 'datetime': 'datetime', 'timestamp': 'datetime',
            'url': 'url', 'uri': 'url', 'email': 'email'
        }
    
    def parse_document(self, content: str, format_hint: Optional[str] = None) -> DocumentStructure:
        """Parse document content into structured format.
        
        Args:
            content: Raw document content
            format_hint: Optional hint about document format ('markdown', 'html', 'text')
            
        Returns:
            DocumentStructure with parsed information
        """
        self.logger.info("Parsing document structure")
        
        # Detect document format if not provided
        if format_hint is None:
            format_hint = self._detect_format(content)
        
        self.logger.info(f"Processing document as {format_hint} format")
        
        # Preprocess content based on format
        processed_content = self._preprocess_content(content, format_hint)
        
        structure = DocumentStructure()
        structure.metadata['format'] = format_hint
        structure.metadata['original_length'] = len(content)
        structure.metadata['processed_length'] = len(processed_content)
        
        # Extract title
        structure.title = self._extract_title(processed_content, format_hint)
        
        # Extract sections with enhanced parsing
        structure.sections = self._extract_sections(processed_content, format_hint)
        
        # Extract code blocks with language detection
        structure.code_blocks = self._extract_code_blocks(processed_content, format_hint)
        
        # Extract examples with multiple patterns
        structure.examples = self._extract_examples(processed_content)
        
        # Extract API specifications using enhanced methods
        structure.api_specs = self.extract_api_specs(structure)
        
        self.logger.info(f"Parsed document: {len(structure.sections)} sections, "
                        f"{len(structure.api_specs)} APIs, {len(structure.code_blocks)} code blocks, "
                        f"{len(structure.examples)} examples")
        
        return structure
    
    def _detect_format(self, content: str) -> str:
        """Detect document format based on content analysis.
        
        Args:
            content: Document content to analyze
            
        Returns:
            Detected format: 'markdown', 'html', or 'text'
        """
        content_lower = content.lower()
        
        # Check for HTML tags
        html_indicators = ['<html', '<body', '<div', '<p>', '<h1', '<h2', '<pre>', '<code>']
        if any(indicator in content_lower for indicator in html_indicators):
            return 'html'
        
        # Check for Markdown indicators
        markdown_indicators = ['# ', '## ', '### ', '```', '* ', '- ', '[', '](']
        markdown_score = sum(1 for indicator in markdown_indicators if indicator in content)
        
        if markdown_score >= 3:
            return 'markdown'
        
        return 'text'
    
    def _preprocess_content(self, content: str, format_type: str) -> str:
        """Preprocess content based on format type.
        
        Args:
            content: Raw content
            format_type: Document format type
            
        Returns:
            Preprocessed content
        """
        if format_type == 'html':
            # Remove HTML tags and decode entities
            content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<[^>]+>', ' ', content)
            content = html.unescape(content)
        
        # Normalize whitespace
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\r', '\n', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content.strip()
    
    def extract_api_specs(self, structure: DocumentStructure) -> List[APISpec]:
        """Extract API specifications from document structure with enhanced processing.
        
        Args:
            structure: Parsed document structure
            
        Returns:
            List of validated APISpec objects
        """
        self.logger.info("Extracting API specifications")
        
        api_specs = []
        
        # Process each section for API information
        for i, section in enumerate(structure.sections):
            self.logger.debug(f"Processing section {i}: {section.get('title', 'Untitled')}")
            section_apis = self._extract_apis_from_section(section)
            api_specs.extend(section_apis)
        
        # Process code blocks for function definitions
        for i, code_block in enumerate(structure.code_blocks):
            self.logger.debug(f"Processing code block {i}: {code_block.get('language', 'unknown')} language")
            code_apis = self._extract_apis_from_code(code_block)
            api_specs.extend(code_apis)
        
        # Enhanced deduplication and merging
        unique_apis = self._deduplicate_and_merge_apis(api_specs)
        
        # Validate and enhance API specifications
        validated_apis = []
        for api in unique_apis:
            enhanced_api = self._enhance_api_specification(api, structure)
            validation_result = enhanced_api.validate()
            
            if validation_result.is_valid:
                validated_apis.append(enhanced_api)
                if validation_result.warnings:
                    self.logger.warning(f"API {api.name} validation warnings: {validation_result.warnings}")
            else:
                self.logger.error(f"API {api.name} validation failed: {validation_result.errors}")
                # Still include invalid APIs but mark them
                enhanced_api.constraints.append("validation_failed")
                validated_apis.append(enhanced_api)
        
        self.logger.info(f"Extracted {len(validated_apis)} API specifications "
                        f"({len([a for a in validated_apis if a.validate().is_valid])} valid)")
        
        return validated_apis
    
    def _deduplicate_and_merge_apis(self, api_specs: List[APISpec]) -> List[APISpec]:
        """Deduplicate and merge API specifications intelligently.
        
        Args:
            api_specs: List of API specifications
            
        Returns:
            List of deduplicated and merged APIs
        """
        unique_apis = {}
        
        for api in api_specs:
            key = api.name.lower()  # Case-insensitive matching
            
            if key not in unique_apis:
                unique_apis[key] = api
            else:
                # Merge information from duplicate APIs
                existing = unique_apis[key]
                
                # Merge descriptions (prefer longer, more detailed ones)
                if len(api.description) > len(existing.description):
                    existing.description = api.description
                elif not existing.description and api.description:
                    existing.description = api.description
                
                # Merge parameters (avoid duplicates by name)
                existing_param_names = {p.name.lower() for p in existing.parameters}
                for param in api.parameters:
                    if param.name.lower() not in existing_param_names:
                        existing.parameters.append(param)
                    else:
                        # Enhance existing parameter with additional info
                        existing_param = next(p for p in existing.parameters 
                                            if p.name.lower() == param.name.lower())
                        if not existing_param.description and param.description:
                            existing_param.description = param.description
                        existing_param.constraints.extend(param.constraints)
                        existing_param.examples.extend(param.examples)
                
                # Merge examples and constraints
                existing.examples.extend(api.examples)
                existing.constraints.extend(api.constraints)
                
                # Prefer more specific return type
                if api.return_type != "Any" and existing.return_type == "Any":
                    existing.return_type = api.return_type
                
                # Prefer HTTP endpoint information
                if api.endpoint and not existing.endpoint:
                    existing.endpoint = api.endpoint
                    existing.method = api.method
        
        return list(unique_apis.values())
    
    def _enhance_api_specification(self, api: APISpec, structure: DocumentStructure) -> APISpec:
        """Enhance API specification with additional inferred information.
        
        Args:
            api: API specification to enhance
            structure: Document structure for context
            
        Returns:
            Enhanced API specification
        """
        # Deduplicate constraints and examples (handle unhashable types)
        api.constraints = list(dict.fromkeys(api.constraints))  # Remove duplicates while preserving order
        api.examples = self._deduplicate_examples(api.examples)
        
        # Infer additional constraints from parameter types
        for param in api.parameters:
            param.constraints = list(dict.fromkeys(param.constraints))
            param.examples = self._deduplicate_examples(param.examples)
            
            # Add type-specific constraints
            if param.type == 'email' and 'format: email' not in param.constraints:
                param.constraints.append('format: email')
            elif param.type == 'url' and 'format: url' not in param.constraints:
                param.constraints.append('format: url')
            elif param.type == 'integer' and not any('minimum' in c for c in param.constraints):
                if 'id' in param.name.lower():
                    param.constraints.append('minimum: 1')
                elif 'count' in param.name.lower() or 'size' in param.name.lower():
                    param.constraints.append('minimum: 0')
        
        # Infer API category from name and description
        api_name_lower = api.name.lower()
        description_lower = api.description.lower()
        
        if any(word in api_name_lower for word in ['get', 'fetch', 'retrieve', 'find', 'search']):
            if 'method: GET' not in api.constraints:
                api.constraints.append('method: GET')
                api.method = 'GET'
        elif any(word in api_name_lower for word in ['create', 'add', 'insert', 'post']):
            if 'method: POST' not in api.constraints:
                api.constraints.append('method: POST')
                api.method = 'POST'
        elif any(word in api_name_lower for word in ['update', 'modify', 'edit', 'put']):
            if 'method: PUT' not in api.constraints:
                api.constraints.append('method: PUT')
                api.method = 'PUT'
        elif any(word in api_name_lower for word in ['delete', 'remove', 'destroy']):
            if 'method: DELETE' not in api.constraints:
                api.constraints.append('method: DELETE')
                api.method = 'DELETE'
        
        # Infer security requirements
        if any(word in description_lower for word in ['auth', 'login', 'token', 'credential']):
            api.constraints.append('requires_authentication: true')
        
        if any(word in description_lower for word in ['admin', 'privileged', 'restricted']):
            api.constraints.append('requires_authorization: true')
        
        # Generate endpoint if not present
        if not api.endpoint:
            api.endpoint = self._generate_endpoint_from_name(api.name)
        
        return api
    
    def _generate_endpoint_from_name(self, api_name: str) -> str:
        """Generate a REST endpoint from API name.
        
        Args:
            api_name: API function name
            
        Returns:
            Generated endpoint path
        """
        # Convert camelCase or snake_case to kebab-case
        name = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', api_name).lower()
        name = name.replace('_', '-')
        
        # Remove common prefixes
        prefixes = ['get-', 'post-', 'put-', 'delete-', 'create-', 'update-', 'fetch-']
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        
        # Ensure it starts with /
        if not name.startswith('/'):
            name = '/' + name
        
        return name
    
    def validate_extracted_specs(self, api_specs: List[APISpec]) -> Dict[str, Any]:
        """Validate extracted API specifications and provide summary.
        
        Args:
            api_specs: List of API specifications to validate
            
        Returns:
            Validation summary dictionary
        """
        summary = {
            'total_apis': len(api_specs),
            'valid_apis': 0,
            'invalid_apis': 0,
            'apis_with_warnings': 0,
            'total_parameters': 0,
            'apis_with_examples': 0,
            'apis_with_constraints': 0,
            'validation_errors': [],
            'validation_warnings': []
        }
        
        for api in api_specs:
            validation_result = api.validate()
            
            if validation_result.is_valid:
                summary['valid_apis'] += 1
            else:
                summary['invalid_apis'] += 1
                summary['validation_errors'].extend([
                    f"{api.name}: {error}" for error in validation_result.errors
                ])
            
            if validation_result.warnings:
                summary['apis_with_warnings'] += 1
                summary['validation_warnings'].extend([
                    f"{api.name}: {warning}" for warning in validation_result.warnings
                ])
            
            summary['total_parameters'] += len(api.parameters)
            
            if api.examples:
                summary['apis_with_examples'] += 1
            
            if api.constraints:
                summary['apis_with_constraints'] += 1
        
        return summary
    
    def identify_functions(self, content: str, language_hint: Optional[str] = None) -> List[Tuple[str, str, str]]:
        """Identify function signatures in content with enhanced parsing.
        
        Args:
            content: Content to search
            language_hint: Optional language hint for better parsing
            
        Returns:
            List of tuples (function_name, full_signature, detected_language)
        """
        functions = []
        
        # Try language-specific patterns first
        if language_hint:
            pattern = self.function_patterns.get(language_hint)
            if pattern:
                matches = pattern.finditer(content)
                for match in matches:
                    func_name = self._extract_function_name_from_match(match, language_hint)
                    if func_name:
                        functions.append((func_name, match.group(0), language_hint))
        
        # Try all patterns if no language hint or no matches found
        if not functions:
            for lang, pattern in self.function_patterns.items():
                matches = pattern.finditer(content)
                for match in matches:
                    func_name = self._extract_function_name_from_match(match, lang)
                    if func_name:
                        # Check if we already found this function
                        if not any(f[0] == func_name for f in functions):
                            functions.append((func_name, match.group(0), lang))
        
        self.logger.debug(f"Identified {len(functions)} functions: {[f[0] for f in functions]}")
        return functions
    
    def _extract_function_name_from_match(self, match, language: str) -> Optional[str]:
        """Extract function name from regex match based on language.
        
        Args:
            match: Regex match object
            language: Programming language
            
        Returns:
            Function name or None if not found
        """
        groups = match.groups()
        
        if language == 'python':
            return groups[0] if groups[0] else None
        elif language == 'javascript':
            return groups[0] if groups[0] else groups[1]
        elif language == 'java':
            return groups[1] if len(groups) > 1 else groups[0]
        elif language == 'markdown':
            return groups[0] if groups[0] else None
        else:  # generic
            return groups[0] if groups and groups[0] else None
    
    def extract_parameters(self, function_signature: str, language_hint: Optional[str] = None, 
                          context: Optional[str] = None) -> List[ParameterSpec]:
        """Extract parameters from function signature with enhanced type and constraint inference.
        
        Args:
            function_signature: Function signature string
            language_hint: Optional language hint for better parsing
            context: Optional surrounding context for constraint inference
            
        Returns:
            List of ParameterSpec objects
        """
        parameters = []
        
        # Extract parameter section from signature
        param_match = re.search(r'\(([^)]*)\)', function_signature)
        if not param_match:
            return parameters
        
        param_text = param_match.group(1).strip()
        if not param_text:
            return parameters
        
        # Choose appropriate pattern based on language
        pattern = self.parameter_patterns.get(language_hint or 'generic', 
                                            self.parameter_patterns['generic'])
        
        # Split parameters by comma, handling nested structures
        param_parts = self._split_parameters(param_text)
        
        for param_part in param_parts:
            param_part = param_part.strip()
            if not param_part:
                continue
                
            param_spec = self._parse_single_parameter(param_part, pattern, language_hint, context)
            if param_spec:
                parameters.append(param_spec)
        
        self.logger.debug(f"Extracted {len(parameters)} parameters from signature")
        return parameters
    
    def _split_parameters(self, param_text: str) -> List[str]:
        """Split parameter text by commas, handling nested structures.
        
        Args:
            param_text: Parameter text to split
            
        Returns:
            List of individual parameter strings
        """
        parameters = []
        current_param = ""
        paren_depth = 0
        bracket_depth = 0
        
        for char in param_text:
            if char == ',' and paren_depth == 0 and bracket_depth == 0:
                if current_param.strip():
                    parameters.append(current_param.strip())
                current_param = ""
            else:
                if char == '(':
                    paren_depth += 1
                elif char == ')':
                    paren_depth -= 1
                elif char == '[':
                    bracket_depth += 1
                elif char == ']':
                    bracket_depth -= 1
                current_param += char
        
        if current_param.strip():
            parameters.append(current_param.strip())
        
        return parameters
    
    def _parse_single_parameter(self, param_text: str, pattern: re.Pattern, 
                               language_hint: Optional[str], context: Optional[str]) -> Optional[ParameterSpec]:
        """Parse a single parameter string into ParameterSpec.
        
        Args:
            param_text: Single parameter text
            pattern: Regex pattern for parsing
            language_hint: Language hint
            context: Surrounding context
            
        Returns:
            ParameterSpec object or None if parsing fails
        """
        match = pattern.search(param_text)
        if not match:
            # Fallback: treat entire text as parameter name
            name = param_text.split()[0] if param_text.split() else param_text
            return ParameterSpec(
                name=name,
                type="Any",
                description=f"Parameter {name}",
                required=True
            )
        
        groups = match.groups()
        
        # Extract name and type based on language
        if language_hint == 'java':
            param_type = groups[0] if groups[0] else "Object"
            name = groups[1] if len(groups) > 1 and groups[1] else "param"
            default_value = groups[2] if len(groups) > 2 else None
        else:
            name = groups[0] if groups[0] else "param"
            param_type = groups[1] if len(groups) > 1 and groups[1] else "Any"
            default_value = groups[2] if len(groups) > 2 else None
        
        # Normalize type
        normalized_type = self.type_mapping.get(param_type.lower(), param_type)
        
        # Infer constraints from context
        constraints = self._infer_parameter_constraints(name, param_type, context or "")
        
        # Generate examples based on type
        examples = self._generate_parameter_examples(normalized_type, constraints)
        
        # Enhanced description
        description = self._generate_parameter_description(name, normalized_type, constraints, context)
        
        return ParameterSpec(
            name=name,
            type=normalized_type,
            description=description,
            required=default_value is None,
            constraints=constraints,
            examples=examples,
            default_value=default_value
        )
    
    def _infer_parameter_constraints(self, param_name: str, param_type: str, context: str) -> List[str]:
        """Infer parameter constraints from context and naming.
        
        Args:
            param_name: Parameter name
            param_type: Parameter type
            context: Surrounding context
            
        Returns:
            List of constraint strings
        """
        constraints = []
        
        # Check for explicit constraints in context
        for constraint_type, pattern in self.constraint_patterns.items():
            matches = pattern.finditer(context)
            for match in matches:
                if constraint_type == 'range':
                    min_val, max_val = match.groups()
                    constraints.append(f"range: {min_val} to {max_val}")
                elif constraint_type == 'min_max':
                    min_val, max_val = match.groups()
                    if min_val:
                        constraints.append(f"minimum: {min_val}")
                    if max_val:
                        constraints.append(f"maximum: {max_val}")
                elif constraint_type == 'length':
                    min_len = match.group(1)
                    max_len = match.group(2) if match.group(2) else None
                    if max_len:
                        constraints.append(f"length: {min_len} to {max_len}")
                    else:
                        constraints.append(f"length: {min_len}")
                elif constraint_type == 'format':
                    format_spec = match.group(1).strip()
                    constraints.append(f"format: {format_spec}")
                elif constraint_type == 'enum':
                    values = match.group(1).strip()
                    constraints.append(f"enum: {values}")
        
        # Infer constraints from parameter name
        name_lower = param_name.lower()
        if 'email' in name_lower:
            constraints.append("format: email")
        elif 'url' in name_lower or 'uri' in name_lower:
            constraints.append("format: url")
        elif 'phone' in name_lower:
            constraints.append("format: phone")
        elif 'id' in name_lower:
            constraints.append("format: identifier")
        elif 'count' in name_lower or 'size' in name_lower:
            constraints.append("minimum: 0")
        elif 'password' in name_lower:
            constraints.append("minimum length: 8")
        
        return constraints
    
    def _generate_parameter_examples(self, param_type: str, constraints: List[str]) -> List[Any]:
        """Generate example values for parameter based on type and constraints.
        
        Args:
            param_type: Parameter type
            constraints: List of constraints
            
        Returns:
            List of example values
        """
        examples = []
        
        # Basic examples by type
        type_examples = {
            'string': ['example', 'test_value', ''],
            'integer': [0, 1, 42, -1],
            'number': [0.0, 1.5, 3.14, -2.5],
            'boolean': [True, False],
            'array': [[], ['item1', 'item2'], [1, 2, 3]],
            'object': [{}, {'key': 'value'}, {'id': 1, 'name': 'test'}],
            'email': ['user@example.com', 'test.email@domain.org'],
            'url': ['https://example.com', 'http://api.service.com/endpoint'],
            'date': ['2023-01-01', '2023-12-31'],
            'datetime': ['2023-01-01T00:00:00Z', '2023-12-31T23:59:59Z']
        }
        
        base_examples = type_examples.get(param_type, ['example_value'])
        examples.extend(base_examples[:3])  # Limit to 3 base examples
        
        # Add constraint-specific examples
        for constraint in constraints:
            if constraint.startswith('enum:'):
                enum_values = constraint[5:].strip().split(',')
                examples.extend([val.strip() for val in enum_values[:2]])
            elif constraint.startswith('range:'):
                # Extract range and add boundary examples
                range_match = re.search(r'(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)', constraint)
                if range_match:
                    min_val, max_val = range_match.groups()
                    if '.' in min_val or '.' in max_val:
                        examples.extend([float(min_val), float(max_val)])
                    else:
                        examples.extend([int(min_val), int(max_val)])
        
        return examples[:5]  # Limit total examples
    
    def _generate_parameter_description(self, name: str, param_type: str, 
                                      constraints: List[str], context: Optional[str]) -> str:
        """Generate enhanced parameter description.
        
        Args:
            name: Parameter name
            param_type: Parameter type
            constraints: List of constraints
            context: Surrounding context
            
        Returns:
            Enhanced description string
        """
        # Start with basic description
        description = f"Parameter {name} of type {param_type}"
        
        # Add constraint information
        if constraints:
            constraint_desc = ", ".join(constraints)
            description += f" with constraints: {constraint_desc}"
        
        # Try to extract description from context
        if context:
            # Look for parameter description in context
            desc_patterns = [
                rf'{re.escape(name)}\s*[-:]\s*([^.\n]+)',
                rf'@param\s+{re.escape(name)}\s+([^.\n]+)',
                rf'{re.escape(name)}\s*\([^)]*\)\s*[-:]\s*([^.\n]+)'
            ]
            
            for pattern in desc_patterns:
                match = re.search(pattern, context, re.IGNORECASE)
                if match:
                    context_desc = match.group(1).strip()
                    if len(context_desc) > 10:  # Only use substantial descriptions
                        description = context_desc
                        break
        
        return description
    
    def _extract_title(self, content: str, format_type: str) -> str:
        """Extract document title based on format.
        
        Args:
            content: Document content
            format_type: Document format type
            
        Returns:
            Extracted title
        """
        if format_type == 'html':
            # Look for HTML title tag
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
            if title_match:
                return title_match.group(1).strip()
            
            # Look for H1 tag
            h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', content, re.IGNORECASE)
            if h1_match:
                return h1_match.group(1).strip()
        
        elif format_type == 'markdown':
            # Look for markdown heading
            title_match = re.search(r'^#+\s*(.+)$', content, re.MULTILINE)
            if title_match:
                return title_match.group(1).strip()
        
        # Fallback: look for title in first few lines
        lines = content.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('<'):
                # Skip very short lines that are likely not titles
                if len(line) > 5:
                    return line
        
        return "Untitled Document"
    
    def _extract_sections(self, content: str, format_type: str) -> List[Dict[str, Any]]:
        """Extract document sections based on format.
        
        Args:
            content: Document content
            format_type: Document format type
            
        Returns:
            List of section dictionaries
        """
        sections = []
        
        if format_type == 'html':
            sections = self._extract_html_sections(content)
        elif format_type == 'markdown':
            sections = self._extract_markdown_sections(content)
        else:
            sections = self._extract_text_sections(content)
        
        # Add metadata to sections
        for section in sections:
            section['format'] = format_type
            section['word_count'] = len(section['content'].split())
            section['is_api_section'] = self._is_api_section(section['title'])
        
        return sections
    
    def _extract_markdown_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from markdown content."""
        sections = []
        
        # Split by headings
        section_pattern = re.compile(r'^(#+)\s*(.+)$', re.MULTILINE)
        matches = list(section_pattern.finditer(content))
        
        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.end()
            
            # Find end position (next heading of same or higher level)
            end_pos = len(content)
            for j in range(i + 1, len(matches)):
                next_match = matches[j]
                next_level = len(next_match.group(1))
                if next_level <= level:
                    end_pos = next_match.start()
                    break
            
            section_content = content[start_pos:end_pos].strip()
            
            sections.append({
                'level': level,
                'title': title,
                'content': section_content
            })
        
        return sections
    
    def _extract_html_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from HTML content."""
        sections = []
        
        # Look for heading tags
        heading_pattern = re.compile(r'<h([1-6])[^>]*>([^<]+)</h[1-6]>', re.IGNORECASE)
        matches = list(heading_pattern.finditer(content))
        
        for i, match in enumerate(matches):
            level = int(match.group(1))
            title = match.group(2).strip()
            start_pos = match.end()
            
            # Find end position (next heading of same or higher level)
            end_pos = len(content)
            for j in range(i + 1, len(matches)):
                next_match = matches[j]
                next_level = int(next_match.group(1))
                if next_level <= level:
                    end_pos = next_match.start()
                    break
            
            section_content = content[start_pos:end_pos].strip()
            
            sections.append({
                'level': level,
                'title': title,
                'content': section_content
            })
        
        return sections
    
    def _extract_text_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract sections from plain text content."""
        sections = []
        
        # Split by double newlines or lines that look like headings
        paragraphs = re.split(r'\n\s*\n', content)
        
        current_section = None
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if this looks like a heading (short line, possibly uppercase)
            lines = paragraph.split('\n')
            first_line = lines[0].strip()
            
            if (len(lines) == 1 and len(first_line) < 80 and 
                (first_line.isupper() or first_line.endswith(':'))):
                # This looks like a heading
                if current_section:
                    sections.append(current_section)
                
                current_section = {
                    'level': 1,
                    'title': first_line.rstrip(':'),
                    'content': ''
                }
            else:
                # This is content
                if current_section:
                    current_section['content'] += paragraph + '\n\n'
                else:
                    # Create a default section for content without heading
                    sections.append({
                        'level': 1,
                        'title': 'Content',
                        'content': paragraph
                    })
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _extract_code_blocks(self, content: str, format_type: str) -> List[Dict[str, Any]]:
        """Extract code blocks from content with language detection.
        
        Args:
            content: Document content
            format_type: Document format type
            
        Returns:
            List of code block dictionaries with metadata
        """
        code_blocks = []
        
        if format_type == 'html':
            code_blocks.extend(self._extract_html_code_blocks(content))
        elif format_type == 'markdown':
            code_blocks.extend(self._extract_markdown_code_blocks(content))
        else:
            code_blocks.extend(self._extract_text_code_blocks(content))
        
        # Add metadata and detect language for each block
        for i, block in enumerate(code_blocks):
            if isinstance(block, str):
                # Convert string to dict format
                code_blocks[i] = {
                    'code': block,
                    'language': self._detect_code_language(block),
                    'line_count': len(block.split('\n'))
                }
            elif isinstance(block, dict) and 'language' not in block:
                block['language'] = self._detect_code_language(block.get('code', ''))
                block['line_count'] = len(block.get('code', '').split('\n'))
        
        return code_blocks
    
    def _extract_markdown_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        
        # Fenced code blocks with language specification
        fenced_pattern = re.compile(r'```(\w+)?\n(.*?)\n```', re.DOTALL)
        for match in fenced_pattern.finditer(content):
            language = match.group(1) or 'unknown'
            code = match.group(2)
            code_blocks.append({
                'code': code,
                'language': language,
                'type': 'fenced'
            })
        
        # Indented code blocks
        indent_pattern = re.compile(r'^    (.+)$', re.MULTILINE)
        indent_matches = indent_pattern.findall(content)
        for code in indent_matches:
            code_blocks.append({
                'code': code,
                'language': 'unknown',
                'type': 'indented'
            })
        
        # Inline code
        inline_pattern = re.compile(r'`([^`]+)`')
        for match in inline_pattern.finditer(content):
            code = match.group(1)
            if len(code) > 10:  # Only include substantial inline code
                code_blocks.append({
                    'code': code,
                    'language': 'unknown',
                    'type': 'inline'
                })
        
        return code_blocks
    
    def _extract_html_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from HTML content."""
        code_blocks = []
        
        # <pre><code> blocks
        pre_code_pattern = re.compile(r'<pre[^>]*><code[^>]*>(.*?)</code></pre>', re.DOTALL | re.IGNORECASE)
        for match in pre_code_pattern.finditer(content):
            code = html.unescape(match.group(1))
            code_blocks.append({
                'code': code,
                'language': 'unknown',
                'type': 'pre_code'
            })
        
        # <code> blocks
        code_pattern = re.compile(r'<code[^>]*>(.*?)</code>', re.DOTALL | re.IGNORECASE)
        for match in code_pattern.finditer(content):
            code = html.unescape(match.group(1))
            if len(code) > 10:  # Only include substantial code
                code_blocks.append({
                    'code': code,
                    'language': 'unknown',
                    'type': 'code'
                })
        
        return code_blocks
    
    def _extract_text_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from plain text content."""
        code_blocks = []
        
        # Look for indented blocks that might be code
        lines = content.split('\n')
        current_block = []
        in_code_block = False
        
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                # This line is indented, might be code
                current_block.append(line.lstrip())
                in_code_block = True
            else:
                if in_code_block and current_block:
                    # End of code block
                    code = '\n'.join(current_block)
                    if len(code.strip()) > 20:  # Only include substantial blocks
                        code_blocks.append({
                            'code': code,
                            'language': 'unknown',
                            'type': 'indented'
                        })
                    current_block = []
                in_code_block = False
        
        # Handle final block
        if current_block:
            code = '\n'.join(current_block)
            if len(code.strip()) > 20:
                code_blocks.append({
                    'code': code,
                    'language': 'unknown',
                    'type': 'indented'
                })
        
        return code_blocks
    
    def _detect_code_language(self, code: str) -> str:
        """Detect programming language from code content.
        
        Args:
            code: Code content to analyze
            
        Returns:
            Detected language or 'unknown'
        """
        code_lower = code.lower()
        
        # Language indicators
        if any(keyword in code_lower for keyword in ['def ', 'import ', 'from ', 'class ', '__init__']):
            return 'python'
        elif any(keyword in code_lower for keyword in ['function ', 'var ', 'let ', 'const ', '=>']):
            return 'javascript'
        elif any(keyword in code_lower for keyword in ['public class', 'private ', 'public static']):
            return 'java'
        elif any(keyword in code_lower for keyword in ['#include', 'int main', 'printf']):
            return 'c'
        elif any(keyword in code_lower for keyword in ['<?php', 'echo ', '$']):
            return 'php'
        elif any(keyword in code_lower for keyword in ['SELECT', 'FROM', 'WHERE', 'INSERT']):
            return 'sql'
        elif any(keyword in code_lower for keyword in ['<html', '<body', '<div']):
            return 'html'
        elif any(keyword in code_lower for keyword in ['{', '}', '":', '[']):
            return 'json'
        
        return 'unknown'
    
    def _extract_examples(self, content: str) -> List[str]:
        """Extract examples from content."""
        examples = []
        
        # Look for example sections
        example_pattern = re.compile(
            r'(?:example|usage|sample):\s*\n(.*?)(?=\n\n|\n#|$)',
            re.IGNORECASE | re.DOTALL
        )
        
        matches = example_pattern.findall(content)
        examples.extend(matches)
        
        return [ex.strip() for ex in examples if ex.strip()]
    
    def _extract_apis_from_section(self, section: Dict[str, Any]) -> List[APISpec]:
        """Extract APIs from a document section with enhanced parsing."""
        apis = []
        content = section['content']
        title = section['title']
        
        # Skip non-API sections
        if not section.get('is_api_section', False):
            return apis
        
        # Check if this section title itself contains a function signature
        title_functions = self.identify_functions(title, 'markdown')
        if title_functions:
            for func_name, signature, language in title_functions:
                api = self._create_api_from_section_title(func_name, signature, content, section)
                if api:
                    apis.append(api)
        
        # Look for function definitions in section content
        functions = self.identify_functions(content)
        
        for func_name, signature, language in functions:
            # Find function description with better context
            description = self._extract_function_description(content, func_name, signature)
            
            # Extract parameters with language hint and context
            parameters = self.extract_parameters(signature, language, content)
            
            # Extract return type if available
            return_type = self._extract_return_type(signature, language)
            
            # Extract examples specific to this function
            examples = self._extract_function_examples(content, func_name)
            
            # Extract constraints from documentation
            constraints = self._extract_function_constraints(content, func_name)
            
            # Determine HTTP method and endpoint if this is a REST API
            method, endpoint = self._extract_http_info(content, func_name)
            
            api = APISpec(
                name=func_name,
                description=description,
                parameters=parameters,
                return_type=return_type,
                examples=examples,
                constraints=constraints,
                method=method,
                endpoint=endpoint
            )
            apis.append(api)
        
        return apis
    
    def _create_api_from_section_title(self, func_name: str, signature: str, 
                                     content: str, section: Dict[str, Any]) -> Optional[APISpec]:
        """Create API specification from section title that contains function signature.
        
        Args:
            func_name: Function name
            signature: Function signature from title
            content: Section content
            section: Section metadata
            
        Returns:
            APISpec object or None
        """
        # Extract parameters from signature
        parameters = self.extract_parameters(signature, 'markdown', content)
        
        # Extract return type
        return_type = self._extract_return_type(signature, 'markdown')
        
        # Use section content as description source
        description = self._extract_description_from_content(content, func_name)
        
        # Extract examples from content
        examples = self._extract_function_examples(content, func_name)
        
        # Extract constraints from content
        constraints = self._extract_function_constraints(content, func_name)
        
        # Determine HTTP method and endpoint
        method, endpoint = self._extract_http_info(content, func_name)
        
        api = APISpec(
            name=func_name,
            description=description,
            parameters=parameters,
            return_type=return_type,
            examples=examples,
            constraints=constraints,
            method=method,
            endpoint=endpoint
        )
        
        return api
    
    def _extract_description_from_content(self, content: str, func_name: str) -> str:
        """Extract description from section content.
        
        Args:
            content: Section content
            func_name: Function name
            
        Returns:
            Extracted description
        """
        # Look for the first substantial paragraph
        paragraphs = content.split('\n\n')
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if (paragraph and 
                not paragraph.startswith('Parameters:') and 
                not paragraph.startswith('Example:') and
                not paragraph.startswith('```') and
                len(paragraph) > 10):
                return paragraph
        
        return f"Function {func_name}"
    
    def _extract_apis_from_code(self, code_block: Dict[str, Any]) -> List[APISpec]:
        """Extract APIs from code block with enhanced parsing."""
        apis = []
        
        code = code_block.get('code', '')
        language = code_block.get('language', 'unknown')
        
        if not code:
            return apis
        
        # Look for function definitions with language-specific parsing
        functions = self.identify_functions(code, language)
        
        for func_name, signature, detected_lang in functions:
            # Find full function definition
            func_def = self._extract_full_function_definition(code, func_name, detected_lang)
            
            # Extract parameters with language context
            parameters = self.extract_parameters(signature, detected_lang, func_def)
            
            # Extract docstring/comments as description
            description = self._extract_docstring(func_def, detected_lang)
            
            # Extract return type
            return_type = self._extract_return_type(signature, detected_lang)
            
            # Extract examples from comments
            examples = self._extract_code_examples(func_def)
            
            api = APISpec(
                name=func_name,
                description=description or f"Function {func_name}",
                parameters=parameters,
                return_type=return_type,
                examples=examples
            )
            apis.append(api)
        
        return apis
    
    def _is_api_section(self, title: str) -> bool:
        """Check if section title indicates API content."""
        api_keywords = [
            'api', 'function', 'method', 'endpoint', 'interface',
            'reference', 'documentation', 'usage', 'management',
            'service', 'operations', 'commands', 'calls'
        ]
        
        title_lower = title.lower()
        return any(keyword in title_lower for keyword in api_keywords)
    
    def _extract_function_description(self, content: str, func_name: str) -> str:
        """Extract description for a function from content."""
        # Look for description before function name
        pattern = rf'([^.\n]*{re.escape(func_name)}[^.\n]*\.)'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if match:
            return match.group(1).strip()
        
        return f"Function {func_name}"
    
    def _extract_return_type(self, signature: str, language: str) -> str:
        """Extract return type from function signature.
        
        Args:
            signature: Function signature
            language: Programming language
            
        Returns:
            Return type or 'Any' if not found
        """
        if language == 'python':
            # Look for -> return_type
            return_match = re.search(r'->\s*([^:]+)', signature)
            if return_match:
                return return_match.group(1).strip()
        elif language == 'java':
            # Return type is before function name
            parts = signature.split()
            for i, part in enumerate(parts):
                if '(' in part:  # Found function name with parameters
                    if i > 0:
                        return parts[i-1]
                    break
        elif language == 'javascript':
            # TypeScript style return type
            return_match = re.search(r':\s*([^{]+)', signature)
            if return_match:
                return return_match.group(1).strip()
        
        return 'Any'
    
    def _extract_function_description(self, content: str, func_name: str, signature: str) -> str:
        """Extract enhanced function description from content.
        
        Args:
            content: Content to search
            func_name: Function name
            signature: Function signature
            
        Returns:
            Function description
        """
        # Look for description patterns around the function
        patterns = [
            # Description before function
            rf'([^.\n]*{re.escape(func_name)}[^.\n]*\.)',
            # JSDoc style
            rf'/\*\*\s*(.*?)\s*\*/\s*{re.escape(signature)}',
            # Python docstring style in documentation
            rf'{re.escape(func_name)}.*?\n\s*"""(.*?)"""',
            # Markdown description
            rf'#+\s*{re.escape(func_name)}\s*\n([^#]+)',
            # Simple description line
            rf'{re.escape(func_name)}\s*[-:]\s*([^\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if match:
                description = match.group(1).strip()
                if len(description) > 10:  # Only use substantial descriptions
                    return description
        
        return f"Function {func_name}"
    
    def _extract_function_examples(self, content: str, func_name: str) -> List[str]:
        """Extract examples specific to a function.
        
        Args:
            content: Content to search
            func_name: Function name
            
        Returns:
            List of example strings
        """
        examples = []
        
        # Look for examples near the function name
        example_patterns = [
            rf'{re.escape(func_name)}\s*\([^)]*\)',  # Function calls
            rf'example.*?{re.escape(func_name)}.*?\n(.*?)(?=\n\n|\n#|$)',  # Example sections
            rf'```.*?{re.escape(func_name)}.*?\n(.*?)\n```'  # Code examples
        ]
        
        for pattern in example_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                example = match.group(0) if len(match.groups()) == 0 else match.group(1)
                if example and len(example.strip()) > 5:
                    examples.append(example.strip())
        
        return examples[:3]  # Limit to 3 examples
    
    def _extract_function_constraints(self, content: str, func_name: str) -> List[str]:
        """Extract constraints specific to a function.
        
        Args:
            content: Content to search
            func_name: Function name
            
        Returns:
            List of constraint strings
        """
        constraints = []
        
        # Look for constraint patterns near the function
        func_section = self._extract_function_section(content, func_name)
        
        for constraint_type, pattern in self.constraint_patterns.items():
            matches = pattern.finditer(func_section)
            for match in matches:
                constraint_text = match.group(0)
                constraints.append(constraint_text)
        
        return constraints
    
    def _extract_function_section(self, content: str, func_name: str) -> str:
        """Extract the section of content related to a specific function.
        
        Args:
            content: Full content
            func_name: Function name
            
        Returns:
            Section of content related to the function
        """
        # Find the function and extract surrounding context
        func_pos = content.lower().find(func_name.lower())
        if func_pos == -1:
            return ""
        
        # Extract context around the function (500 chars before and after)
        start = max(0, func_pos - 500)
        end = min(len(content), func_pos + len(func_name) + 500)
        
        return content[start:end]
    
    def _extract_http_info(self, content: str, func_name: str) -> Tuple[str, Optional[str]]:
        """Extract HTTP method and endpoint information.
        
        Args:
            content: Content to search
            func_name: Function name
            
        Returns:
            Tuple of (method, endpoint)
        """
        method = "POST"  # Default
        endpoint = None
        
        func_section = self._extract_function_section(content, func_name)
        
        # Look for HTTP method indicators
        method_patterns = {
            'GET': r'(?:GET|get)\s+',
            'POST': r'(?:POST|post)\s+',
            'PUT': r'(?:PUT|put)\s+',
            'DELETE': r'(?:DELETE|delete)\s+',
            'PATCH': r'(?:PATCH|patch)\s+'
        }
        
        for http_method, pattern in method_patterns.items():
            if re.search(pattern, func_section, re.IGNORECASE):
                method = http_method
                break
        
        # Look for endpoint patterns
        endpoint_patterns = [
            r'(?:endpoint|url|path):\s*([^\s\n]+)',
            r'(?:GET|POST|PUT|DELETE|PATCH)\s+([^\s\n]+)',
            r'/[a-zA-Z0-9/_-]+',  # URL-like patterns
        ]
        
        for pattern in endpoint_patterns:
            match = re.search(pattern, func_section, re.IGNORECASE)
            if match:
                endpoint = match.group(1) if len(match.groups()) > 0 else match.group(0)
                break
        
        return method, endpoint
    
    def _extract_full_function_definition(self, code: str, func_name: str, language: str) -> str:
        """Extract the full function definition from code.
        
        Args:
            code: Code content
            func_name: Function name
            language: Programming language
            
        Returns:
            Full function definition
        """
        if language == 'python':
            pattern = rf'(?:def|async\s+def)\s+{re.escape(func_name)}\s*\([^)]*\).*?(?=\n(?:def|class|async\s+def|\Z))'
        elif language == 'javascript':
            pattern = rf'(?:function\s+{re.escape(func_name)}|{re.escape(func_name)}\s*=\s*(?:function|\([^)]*\)\s*=>)).*?(?=\n(?:function|\w+\s*=|\Z))'
        elif language == 'java':
            pattern = rf'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+{re.escape(func_name)}\s*\([^)]*\).*?(?=\n\s*(?:public|private|protected|\Z))'
        else:
            # Generic pattern
            pattern = rf'(?:def|function)\s+{re.escape(func_name)}\s*\([^)]*\).*?(?=\n(?:def|function|\Z))'
        
        match = re.search(pattern, code, re.DOTALL | re.IGNORECASE)
        return match.group(0) if match else ""
    
    def _extract_docstring(self, func_def: str, language: str = 'python') -> str:
        """Extract docstring/comments from function definition.
        
        Args:
            func_def: Function definition
            language: Programming language
            
        Returns:
            Extracted documentation
        """
        if language == 'python':
            # Python docstring patterns
            patterns = [
                re.compile(r'"""(.*?)"""', re.DOTALL),
                re.compile(r"'''(.*?)'''", re.DOTALL),
                re.compile(r'"([^"]+)"'),
                re.compile(r"'([^']+)'")
            ]
        elif language == 'javascript':
            # JavaScript JSDoc and comments
            patterns = [
                re.compile(r'/\*\*(.*?)\*/', re.DOTALL),
                re.compile(r'/\*(.*?)\*/', re.DOTALL),
                re.compile(r'//\s*(.+)')
            ]
        elif language == 'java':
            # Java Javadoc and comments
            patterns = [
                re.compile(r'/\*\*(.*?)\*/', re.DOTALL),
                re.compile(r'/\*(.*?)\*/', re.DOTALL),
                re.compile(r'//\s*(.+)')
            ]
        else:
            # Generic comment patterns
            patterns = [
                re.compile(r'/\*(.*?)\*/', re.DOTALL),
                re.compile(r'#\s*(.+)'),
                re.compile(r'//\s*(.+)')
            ]
        
        for pattern in patterns:
            match = pattern.search(func_def)
            if match:
                docstring = match.group(1).strip()
                if len(docstring) > 10:  # Only use substantial documentation
                    return docstring
        
        return ""
    
    def _extract_code_examples(self, func_def: str) -> List[str]:
        """Extract examples from code comments and docstrings.
        
        Args:
            func_def: Function definition
            
        Returns:
            List of example strings
        """
        examples = []
        
        # Look for example patterns in comments
        example_patterns = [
            r'example:\s*(.*?)(?=\n|$)',
            r'usage:\s*(.*?)(?=\n|$)',
            r'sample:\s*(.*?)(?=\n|$)',
            r'>>>\s*(.*?)(?=\n|$)'  # Python doctest style
        ]
        
        for pattern in example_patterns:
            matches = re.finditer(pattern, func_def, re.IGNORECASE)
            for match in matches:
                example = match.group(1).strip()
                if example and len(example) > 5:
                    examples.append(example)
        
        return examples[:3]  # Limit to 3 examples
    
    def create_comprehensive_api_spec(self, name: str, signature: str, context: str, 
                                    language_hint: Optional[str] = None) -> APISpec:
        """Create a comprehensive API specification from minimal information.
        
        Args:
            name: API function name
            signature: Function signature
            context: Surrounding context for inference
            language_hint: Optional language hint
            
        Returns:
            Comprehensive APISpec object
        """
        # Extract parameters with full context
        parameters = self.extract_parameters(signature, language_hint, context)
        
        # Extract return type
        return_type = self._extract_return_type(signature, language_hint)
        
        # Generate comprehensive description
        description = self._generate_comprehensive_description(name, signature, context, parameters)
        
        # Extract examples
        examples = self._extract_function_examples(context, name)
        
        # Extract constraints
        constraints = self._extract_function_constraints(context, name)
        
        # Determine HTTP method and endpoint
        method, endpoint = self._extract_http_info(context, name)
        
        # Create and enhance the API spec
        api_spec = APISpec(
            name=name,
            description=description,
            parameters=parameters,
            return_type=return_type,
            examples=examples,
            constraints=constraints,
            method=method,
            endpoint=endpoint
        )
        
        # Apply enhancements
        enhanced_spec = self._enhance_api_specification(api_spec, DocumentStructure())
        
        return enhanced_spec
    
    def _generate_comprehensive_description(self, name: str, signature: str, context: str, 
                                          parameters: List[ParameterSpec]) -> str:
        """Generate a comprehensive description for an API.
        
        Args:
            name: API name
            signature: Function signature
            context: Context information
            parameters: List of parameters
            
        Returns:
            Comprehensive description
        """
        # Try to extract existing description first
        existing_desc = self._extract_function_description(context, name, signature)
        if existing_desc and existing_desc != f"Function {name}" and len(existing_desc) > 20:
            return existing_desc
        
        # Generate description based on name and parameters
        description_parts = []
        
        # Analyze function name for action
        name_lower = name.lower()
        if any(word in name_lower for word in ['get', 'fetch', 'retrieve', 'find']):
            action = "retrieves"
        elif any(word in name_lower for word in ['create', 'add', 'insert']):
            action = "creates"
        elif any(word in name_lower for word in ['update', 'modify', 'edit']):
            action = "updates"
        elif any(word in name_lower for word in ['delete', 'remove', 'destroy']):
            action = "deletes"
        elif any(word in name_lower for word in ['list', 'search']):
            action = "lists"
        else:
            action = "processes"
        
        # Extract object from name
        object_name = self._extract_object_from_name(name)
        
        # Build description
        if object_name:
            description_parts.append(f"This function {action} {object_name}")
        else:
            description_parts.append(f"This function {action} data")
        
        # Add parameter information
        if parameters:
            required_params = [p for p in parameters if p.required]
            if required_params:
                param_names = [p.name for p in required_params[:3]]  # Limit to first 3
                if len(required_params) > 3:
                    param_str = f"{', '.join(param_names)}, and {len(required_params) - 3} more parameters"
                else:
                    param_str = ', '.join(param_names)
                description_parts.append(f"Requires: {param_str}")
        
        return '. '.join(description_parts) + '.'
    
    def _extract_object_from_name(self, name: str) -> Optional[str]:
        """Extract the object/entity name from a function name.
        
        Args:
            name: Function name
            
        Returns:
            Extracted object name or None
        """
        # Remove common prefixes and suffixes
        clean_name = name
        prefixes = ['get', 'set', 'create', 'update', 'delete', 'fetch', 'find', 'search', 'list']
        suffixes = ['data', 'info', 'details', 'list', 'item', 'items']
        
        name_lower = clean_name.lower()
        for prefix in prefixes:
            if name_lower.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break
        
        for suffix in suffixes:
            if name_lower.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        # Convert camelCase to readable format
        clean_name = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', clean_name).lower()
        clean_name = clean_name.replace('_', ' ').strip()
        
        return clean_name if clean_name and len(clean_name) > 1 else None
    
    def extract_api_specs_from_text(self, text: str, format_hint: Optional[str] = None) -> List[APISpec]:
        """Extract API specifications directly from text content.
        
        Args:
            text: Text content to process
            format_hint: Optional format hint
            
        Returns:
            List of extracted API specifications
        """
        # Parse the document first
        structure = self.parse_document(text, format_hint)
        
        # Extract API specifications
        api_specs = self.extract_api_specs(structure)
        
        return api_specs
    
    def get_extraction_statistics(self, structure: DocumentStructure, api_specs: List[APISpec]) -> Dict[str, Any]:
        """Get statistics about the extraction process.
        
        Args:
            structure: Parsed document structure
            api_specs: Extracted API specifications
            
        Returns:
            Statistics dictionary
        """
        stats = {
            'document': {
                'format': structure.metadata.get('format', 'unknown'),
                'title': structure.title,
                'sections': len(structure.sections),
                'code_blocks': len(structure.code_blocks),
                'examples': len(structure.examples),
                'total_words': sum(section.get('word_count', 0) for section in structure.sections)
            },
            'extraction': {
                'total_apis': len(api_specs),
                'apis_with_parameters': len([api for api in api_specs if api.parameters]),
                'apis_with_examples': len([api for api in api_specs if api.examples]),
                'apis_with_constraints': len([api for api in api_specs if api.constraints]),
                'total_parameters': sum(len(api.parameters) for api in api_specs),
                'languages_detected': list(set(
                    block.get('language', 'unknown') for block in structure.code_blocks
                )),
                'http_methods': list(set(api.method for api in api_specs if api.method))
            },
            'validation': self.validate_extracted_specs(api_specs)
        }
        
        return stats
    
    def _deduplicate_examples(self, examples: List[Any]) -> List[Any]:
        """Deduplicate examples list, handling unhashable types.
        
        Args:
            examples: List of examples that may contain unhashable types
            
        Returns:
            Deduplicated list of examples
        """
        if not examples:
            return examples
        
        seen = []
        result = []
        
        for example in examples:
            try:
                # Try to use the example as a dictionary key (hashable check)
                if example not in seen:
                    seen.append(example)
                    result.append(example)
            except TypeError:
                # Handle unhashable types by converting to string for comparison
                example_str = str(example)
                if example_str not in [str(s) for s in seen]:
                    seen.append(example)
                    result.append(example)
        
        return result