"""LLM-based test case generator for fuzzing."""

import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from src.fuzzer.data_models import APISpec, ParameterSpec, TestCase, TestType, ValidationResult, FuzzerConfig, ExecutionResult
from src.fuzzer.performance_optimizations import LLMResponseCache
from src.fuzzer.adaptive_generation import AdaptiveGenerator, AdaptationStrategy
from src.utils.llm_client import LLMClient
from src.utils.logger import get_logger


class LLMTestGenerator:
    """Generates test cases using LLM based on API specifications."""
    
    def __init__(self, config: FuzzerConfig, seed_manager=None):
        """Initialize test generator.
        
        Args:
            config: Fuzzer configuration
            seed_manager: Random seed manager for deterministic behavior
        """
        self.config = config
        self.logger = get_logger("LLMTestGenerator")
        self.seed_manager = seed_manager
        
        # Initialize LLM client
        self.llm_client = LLMClient(
            model=config.llm_model,
            timeout=config.llm_timeout,
            api_key=config.llm_api_key
        )

        self.summary_client: Optional[LLMClient] = None
        if config.summary_enabled:
            self.summary_client = LLMClient(
                endpoint=config.summary_endpoint,
                model=config.summary_model,
                timeout=config.summary_timeout,
                api_key=config.summary_api_key,
            )
            if not self.summary_client.test_connection():
                self.logger.warning("Summary model health check failed")
        
        # Initialize performance optimizations
        self.response_cache = LLMResponseCache(
            max_size=getattr(config, 'cache_size', 1000),
            ttl_hours=getattr(config, 'cache_ttl_hours', 24)
        )
        
        # Rate limiting state
        self._last_request_time = 0.0
        self._min_request_interval = 1.0  # Minimum 1 second between requests
        self._request_count = 0
        self._rate_limit_window_start = time.time()
        self._max_requests_per_minute = 50  # Conservative rate limit
        
        # Performance tracking
        self.last_generation_time = 0.0
        self.total_requests = 0
        self.cache_hits = 0
        
        # Initialize adaptive generation
        self.adaptive_generator = AdaptiveGenerator(
            learning_rate=getattr(config, 'adaptive_learning_rate', 0.1),
            exploration_rate=getattr(config, 'adaptive_exploration_rate', 0.2),
            pattern_memory_size=getattr(config, 'adaptive_pattern_memory', 1000)
        )
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            random.seed(config.random_seed)
        
        # Prompt templates
        self._prompt_templates = self._initialize_prompt_templates()
        self._document_prompt_template = self._initialize_document_prompt()
        self._summary_prompt_template = (
            "Summarize the following technical documentation for fuzzing input generation. "
            "Focus on syntax, constraints, and typical usage. Keep it concise.\n\n{document}"
        )

    def _initialize_document_prompt(self) -> str:
        return (
            "{doc_block}\n\n"
            "{trigger}\n"
            "{input_hint}\n"
            "Generate a complete compilable Java program. Requirements:\n"
            "- Must include a public class and a main method.\n"
            "- Prefer simple, minimal programs without advanced features like generics, annotations, complex inheritance, or concurrency.\n"
            "- Focus on basic syntax, control structures, and standard library usage.\n"
            "- If using pattern matching, use `case T t when condition ->` (no `&&`).\n"
            "- Keep the program self-contained and compilable with JDK 23 preview.\n"
            "Return a single Java code block only.\n"
        )

    def summarize_document(self, document_content: str) -> str:
        fallback = document_content.strip()[:800]
        if not self.summary_client:
            return fallback

        prompt = self._summary_prompt_template.format(document=fallback)
        try:
            summary = self.summary_client.simple_completion(
                prompt=prompt,
                temperature=self.config.summary_temperature,
                max_tokens=self.config.summary_max_tokens,
            )
            summary = summary.strip()
            return summary if summary else fallback
        except Exception as exc:
            self.logger.warning(f"Summary generation failed, using raw document: {exc}")
            return fallback

    def generate_document_test_cases(self, document_content: str, count: int) -> List[TestCase]:
        if self.config.document_generation_case_file:
            return self._load_document_cases_from_file(
                self.config.document_generation_case_file,
                count,
            )
        summary = self.summarize_document(document_content)
        doc_block = summary.strip() or document_content.strip()
        test_cases: List[TestCase] = []
        attempts = 0
        max_attempts = max(count * 3, 10)
        while len(test_cases) < count and attempts < max_attempts:
            prompt = self._document_prompt_template.format(
                count=1,
                doc_block=doc_block,
                trigger=self.config.document_prompt_trigger,
                input_hint=self.config.document_prompt_hint,
                language=self.config.document_generation_language,
            )
            if (
                self.llm_client.endpoint.endswith("/v1/completions")
                or self.llm_client.endpoint.endswith("/completions")
                or self.llm_client.endpoint.endswith("/api/generate")
            ):
                prompt = f"{prompt}\n```java\n"
            response = self._make_llm_request(
                prompt,
                system_message=(
                    "You must output a single Java code block fenced with ```java. "
                    "Do not include any other text."
                ),
                stop=["```"],
            )
            parsed = self._parse_document_response(response, 1)
            if not parsed and self.llm_client.endpoint.endswith("/api/generate"):
                fallback = self._ensure_compilable_source(self._fix_java_syntax(response))
                if fallback and "public class" in fallback:
                    parsed = [
                        TestCase(
                            id="document_case_1",
                            api_name="document",
                            parameters={"java_source": fallback},
                            test_type=TestType.NORMAL,
                            expected_result="compile",
                            generation_prompt="document_generation_direct",
                        )
                    ]
            if parsed:
                parsed[0].id = f"document_case_{len(test_cases)+1}"
                test_cases.append(parsed[0])
            attempts += 1
        return test_cases

    def _load_document_cases_from_file(self, file_path: str, count: int) -> List[TestCase]:
        path = Path(file_path)
        if not path.exists():
            self.logger.warning(f"Document case file not found: {file_path}")
            return []

        content = path.read_text(encoding="utf-8")
        parts = [part.strip() for part in content.split("// case ") if part.strip()]

        test_cases: List[TestCase] = []
        for part in parts:
            lines = part.splitlines()
            if not lines:
                continue
            if lines[0].strip().isdigit():
                java_source = "\n".join(lines[1:]).strip()
            else:
                java_source = part.strip()
            java_source = self._ensure_compilable_source(self._fix_java_syntax(java_source))
            import_lines = []
            for line in java_source.splitlines():
                if line.strip().startswith("import ") or line.strip().startswith("package "):
                    import_lines.append(line)
                elif "public class" in line:
                    break
            extracted_classes = self._extract_java_classes(java_source)
            if not extracted_classes:
                continue
            for extracted in extracted_classes:
                extracted = self._ensure_compilable_source(self._fix_java_syntax(extracted))
                if import_lines and "import " not in extracted:
                    extracted = "\n".join(import_lines) + "\n\n" + extracted
                test_cases.append(
                    TestCase(
                        id=f"document_case_{len(test_cases)+1}",
                        api_name="document",
                        parameters={"java_source": extracted},
                        test_type=TestType.NORMAL,
                        expected_result="compile",
                        generation_prompt="document_generation_file",
                    )
                )
                if len(test_cases) >= count:
                    break
            if len(test_cases) >= count:
                break

        self.logger.info(f"Loaded {len(test_cases)} document cases from file")
        return test_cases
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize prompt templates for different test case types."""
        return {
            'normal': """Generate {count} valid test cases for the following API:

API Name: {api_name}
Description: {description}

Parameters:
{parameters}

{examples_section}

Generate {count} diverse, valid test cases that:
1. Follow the API specification exactly
2. Use realistic parameter values
3. Cover different usage scenarios
4. Include both simple and complex cases

Format as JSON array:
[
  {{
    "parameters": {{"param1": "value1", "param2": "value2"}},
    "expected_result": "success",
    "description": "Brief description of test case"
  }}
]""",
            
            'security': """Generate security test cases for the following API:

API Name: {api_name}
Description: {description}

Parameters:
{parameters}

Generate test cases that attempt to exploit potential vulnerabilities:

1. SQL Injection: Try SQL injection payloads in string parameters
2. XSS: Cross-site scripting attempts in user input fields
3. Buffer Overflow: Extremely long strings to test buffer limits
4. Authentication Bypass: Attempts to bypass authentication
5. Authorization Escalation: Attempts to access unauthorized resources
6. Input Validation Bypass: Malicious inputs to bypass validation

Format as JSON array with security-focused test cases:
[
  {{
    "parameters": {{"param": "malicious_payload"}},
    "expected_result": "error_or_vulnerability",
    "attack_type": "sql_injection",
    "description": "SQL injection attempt"
  }}
]""",
            
            'edge': """Generate edge case test cases for the following API:

API Name: {api_name}
Description: {description}

Parameters:
{parameters}

Generate test cases that test boundary conditions and edge cases:

1. Empty values (empty strings, null, undefined)
2. Boundary values (min/max integers, very long strings)
3. Special characters and Unicode
4. Invalid data types
5. Missing required parameters
6. Extra unexpected parameters

Format as JSON array:
[
  {{
    "parameters": {{"param": "edge_case_value"}},
    "expected_result": "success_or_error",
    "edge_case_type": "boundary_value",
    "description": "Description of edge case"
  }}
]""",
            
            'malformed': """Generate malformed test cases for the following API:

API Name: {api_name}
Description: {description}

Parameters:
{parameters}

Generate test cases with intentionally malformed inputs:

1. Wrong data types (strings for numbers, numbers for strings)
2. Invalid JSON structures
3. Corrupted parameter names
4. Invalid parameter combinations
5. Syntax errors in string values

Format as JSON array:
[
  {{
    "parameters": {{"param": "malformed_value"}},
    "expected_result": "error",
    "malformation_type": "wrong_type",
    "description": "Description of malformation"
  }}
]"""
        }
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting for LLM API calls."""
        current_time = time.time()
        
        # Reset request count if window has passed
        if current_time - self._rate_limit_window_start >= 60.0:
            self._request_count = 0
            self._rate_limit_window_start = current_time
        
        # Check if we've exceeded rate limit
        if self._request_count >= self._max_requests_per_minute:
            wait_time = 60.0 - (current_time - self._rate_limit_window_start)
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self._request_count = 0
                self._rate_limit_window_start = time.time()
        
        # Enforce minimum interval between requests
        time_since_last = current_time - self._last_request_time
        if time_since_last < self._min_request_interval:
            wait_time = self._min_request_interval - time_since_last
            self.logger.debug(f"Enforcing minimum interval, waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        
        self._last_request_time = time.time()
        self._request_count += 1
    
    def _make_llm_request(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_message: Optional[str] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Make an LLM request with caching, rate limiting and error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Temperature override (uses config default if None)
            max_tokens: Max tokens override (uses config default if None)
            
        Returns:
            LLM response text
            
        Raises:
            Exception: If LLM request fails after retries
        """
        # Use config defaults if not specified
        temp = temperature if temperature is not None else self.config.llm_temperature
        tokens = max_tokens if max_tokens is not None else self.config.llm_max_tokens
        
        # Check cache first
        cached_response = self.response_cache.get(
            prompt=prompt,
            model=self.config.llm_model,
            temperature=temp,
            max_tokens=tokens
        )
        
        if cached_response is not None:
            self.cache_hits += 1
            self.logger.debug("Using cached LLM response")
            return cached_response
        
        # Cache miss - make actual request
        self._enforce_rate_limit()
        
        start_time = time.time()
        try:
            response = self.llm_client.simple_completion(
                prompt=prompt,
                system_message=system_message,
                temperature=temp,
                max_tokens=tokens,
                stop=stop,
            )
            
            # Cache the response
            self.response_cache.put(
                prompt=prompt,
                model=self.config.llm_model,
                temperature=temp,
                max_tokens=tokens,
                response=response
            )
            
            # Update performance tracking
            self.total_requests += 1
            self.last_generation_time = time.time() - start_time
            
            return response
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            raise
    
    def generate_test_cases(self, api_specs: List[APISpec], count: int) -> List[TestCase]:
        """Generate test cases for API specifications.
        
        Args:
            api_specs: List of API specifications
            count: Total number of test cases to generate
            
        Returns:
            List of generated test cases
        """
        self.logger.info(f"Generating {count} test cases for {len(api_specs)} APIs")
        
        if not api_specs:
            self.logger.warning("No API specifications provided")
            return []
        
        test_cases = []
        cases_per_api = max(1, count // len(api_specs))
        
        for api_spec in api_specs:
            api_cases = self._generate_cases_for_api(api_spec, cases_per_api)
            test_cases.extend(api_cases)
        
        # Trim to exact count if needed
        if len(test_cases) > count:
            test_cases = test_cases[:count]
        
        self.logger.info(f"Generated {len(test_cases)} test cases")
        return test_cases
    
    def generate_edge_cases(self, api_spec: APISpec, count: int) -> List[TestCase]:
        """Edge case generator for boundary values.
        
        This generator creates test cases that test boundary conditions,
        edge values, and corner cases for each parameter type. It includes
        empty values, null values, boundary numbers, and special characters.
        
        Args:
            api_spec: API specification
            count: Number of edge cases to generate
            
        Returns:
            List of edge case test cases
        """
        self.logger.debug(f"Generating {count} edge cases for {api_spec.name}")
        
        edge_cases = []
        
        # Try LLM generation first for more creative edge cases
        llm_count = min(count, count // 3)  # Use LLM for 1/3 of cases
        if llm_count > 0:
            try:
                prompt = self._build_prompt_from_template('edge', api_spec, llm_count)
                response = self._make_llm_request(prompt)
                llm_cases = self._parse_llm_response(response, api_spec, TestType.EDGE)
                if llm_cases:
                    edge_cases.extend(llm_cases[:llm_count])
                    self.logger.debug(f"Generated {len(llm_cases)} edge cases via LLM")
            except Exception as e:
                self.logger.warning(f"LLM edge case generation failed: {e}")
        
        # Generate systematic edge cases for each parameter type
        systematic_cases = []
        for param in api_spec.parameters:
            if param.type == 'string':
                systematic_cases.extend(self._create_string_edge_cases(api_spec, param))
            elif param.type in ['integer', 'number']:
                systematic_cases.extend(self._create_numeric_edge_cases(api_spec, param))
            elif param.type == 'array':
                systematic_cases.extend(self._create_array_edge_cases(api_spec, param))
            elif param.type == 'boolean':
                systematic_cases.extend(self._create_boolean_edge_cases(api_spec, param))
            elif param.type == 'object':
                systematic_cases.extend(self._create_object_edge_cases(api_spec, param))
        
        # Add systematic cases to reach the target count
        remaining = max(0, count - len(edge_cases))
        edge_cases.extend(systematic_cases[:remaining])
        
        return edge_cases[:count]
    
    def generate_security_test_cases(self, api_spec: APISpec, count: int) -> List[TestCase]:
        """Security test generator for vulnerability detection.
        
        This generator creates test cases designed to detect security
        vulnerabilities including SQL injection, XSS, buffer overflows,
        authentication bypass, and other common attack vectors.
        
        Args:
            api_spec: API specification
            count: Number of security test cases to generate
            
        Returns:
            List of security test cases
        """
        self.logger.debug(f"Generating {count} security test cases for {api_spec.name}")
        
        test_cases = []
        
        # Try LLM generation first for sophisticated attack patterns
        llm_count = min(count, count // 2)  # Use LLM for half the cases
        if llm_count > 0:
            try:
                prompt = self._build_prompt_from_template('security', api_spec, llm_count)
                response = self._make_llm_request(prompt)
                llm_cases = self._parse_llm_response(response, api_spec, TestType.SECURITY)
                test_cases.extend(llm_cases[:llm_count])
                self.logger.debug(f"Generated {len(llm_cases)} security test cases via LLM")
            except Exception as e:
                self.logger.warning(f"LLM security test generation failed: {e}")
        
        # Generate systematic security tests for remaining count
        remaining_count = count - len(test_cases)
        if remaining_count > 0:
            systematic_cases = self._generate_comprehensive_security_tests(api_spec, remaining_count)
            test_cases.extend(systematic_cases)
        
        return test_cases[:count]
    
    def _generate_comprehensive_security_tests(self, api_spec: APISpec, count: int) -> List[TestCase]:
        """Generate comprehensive security test cases covering all major attack vectors."""
        test_cases = []
        
        # Define security test categories with their generators
        security_generators = [
            ('sql_injection', self._generate_sql_injection_tests),
            ('xss', self._generate_xss_tests),
            ('buffer_overflow', self._generate_buffer_overflow_tests),
            ('auth_bypass', self._generate_auth_bypass_tests),
            ('path_traversal', self._generate_path_traversal_tests),
            ('command_injection', self._generate_command_injection_tests),
            ('ldap_injection', self._generate_ldap_injection_tests),
            ('xml_injection', self._generate_xml_injection_tests)
        ]
        
        # Distribute test cases across different attack types
        cases_per_type = max(1, count // len(security_generators))
        
        for attack_type, generator_func in security_generators:
            if len(test_cases) >= count:
                break
            
            try:
                attack_cases = generator_func(api_spec)
                # Take only the needed number of cases for this attack type
                needed = min(cases_per_type, count - len(test_cases))
                test_cases.extend(attack_cases[:needed])
                self.logger.debug(f"Generated {len(attack_cases[:needed])} {attack_type} test cases")
            except Exception as e:
                self.logger.warning(f"Failed to generate {attack_type} tests: {e}")
        
        return test_cases[:count]
    

    
    def _generate_sql_injection_tests(self, api_spec: APISpec) -> List[TestCase]:
        """Generate SQL injection test cases."""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--",
            "'; INSERT INTO users VALUES ('hacker', 'password'); --"
        ]
        
        test_cases = []
        string_params = [p for p in api_spec.parameters if p.type == 'string']
        
        for i, payload in enumerate(sql_payloads):
            for param in string_params:
                parameters = {}
                
                # Apply SQL injection payload to string parameter
                parameters[param.name] = payload
                
                # Set default values for other parameters
                for other_param in api_spec.parameters:
                    if other_param.name != param.name:
                        parameters[other_param.name] = self._generate_default_value(other_param.type)
                
                test_case = TestCase(
                    id=f"{api_spec.name}_sql_injection_{param.name}_{i+1}",
                    api_name=api_spec.name,
                    parameters=parameters,
                    test_type=TestType.SECURITY,
                    expected_result="error",
                    generation_prompt=f"SQL injection test for parameter {param.name}"
                )
                
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_xss_tests(self, api_spec: APISpec) -> List[TestCase]:
        """Generate XSS test cases."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert(1)",
            "<img src=x onerror=alert(1)>",
            "<svg onload=alert(1)>",
            "';alert(String.fromCharCode(88,83,83))//';alert(String.fromCharCode(88,83,83))//\";alert(String.fromCharCode(88,83,83))//\";alert(String.fromCharCode(88,83,83))//--></SCRIPT>\">'><SCRIPT>alert(String.fromCharCode(88,83,83))</SCRIPT>"
        ]
        
        test_cases = []
        string_params = [p for p in api_spec.parameters if p.type == 'string']
        
        for i, payload in enumerate(xss_payloads):
            for param in string_params:
                parameters = {}
                
                parameters[param.name] = payload
                
                for other_param in api_spec.parameters:
                    if other_param.name != param.name:
                        parameters[other_param.name] = self._generate_default_value(other_param.type)
                
                test_case = TestCase(
                    id=f"{api_spec.name}_xss_{param.name}_{i+1}",
                    api_name=api_spec.name,
                    parameters=parameters,
                    test_type=TestType.SECURITY,
                    expected_result="error",
                    generation_prompt=f"XSS test for parameter {param.name}"
                )
                
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_buffer_overflow_tests(self, api_spec: APISpec) -> List[TestCase]:
        """Generate buffer overflow test cases."""
        test_cases = []
        string_params = [p for p in api_spec.parameters if p.type == 'string']
        
        # Different buffer sizes to test
        buffer_sizes = [1000, 10000, 100000]
        
        for size in buffer_sizes:
            for param in string_params:
                parameters = {}
                
                # Create very long string
                parameters[param.name] = "A" * size
                
                for other_param in api_spec.parameters:
                    if other_param.name != param.name:
                        parameters[other_param.name] = self._generate_default_value(other_param.type)
                
                test_case = TestCase(
                    id=f"{api_spec.name}_buffer_overflow_{param.name}_{size}",
                    api_name=api_spec.name,
                    parameters=parameters,
                    test_type=TestType.SECURITY,
                    expected_result="error",
                    generation_prompt=f"Buffer overflow test for parameter {param.name} with {size} characters"
                )
                
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_auth_bypass_tests(self, api_spec: APISpec) -> List[TestCase]:
        """Generate authentication bypass test cases."""
        auth_payloads = [
            {"admin": True},
            {"role": "admin"},
            {"user_id": 0},
            {"user_id": -1},
            {"token": ""},
            {"auth": "bypass"}
        ]
        
        test_cases = []
        
        for i, payload in enumerate(auth_payloads):
            parameters = {}
            
            # Add auth bypass payload
            parameters.update(payload)
            
            # Add normal parameters
            for param in api_spec.parameters:
                if param.name not in parameters:
                    parameters[param.name] = self._generate_default_value(param.type)
            
            test_case = TestCase(
                id=f"{api_spec.name}_auth_bypass_{i+1}",
                api_name=api_spec.name,
                parameters=parameters,
                test_type=TestType.SECURITY,
                expected_result="error",
                generation_prompt=f"Authentication bypass test with payload {payload}"
            )
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_path_traversal_tests(self, api_spec: APISpec) -> List[TestCase]:
        """Generate path traversal test cases."""
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "/etc/passwd%00.jpg"
        ]
        
        test_cases = []
        string_params = [p for p in api_spec.parameters if p.type == 'string']
        
        for i, payload in enumerate(path_payloads):
            for param in string_params:
                parameters = {}
                
                parameters[param.name] = payload
                
                for other_param in api_spec.parameters:
                    if other_param.name != param.name:
                        parameters[other_param.name] = self._generate_default_value(other_param.type)
                
                test_case = TestCase(
                    id=f"{api_spec.name}_path_traversal_{param.name}_{i+1}",
                    api_name=api_spec.name,
                    parameters=parameters,
                    test_type=TestType.SECURITY,
                    expected_result="error",
                    generation_prompt=f"Path traversal test for parameter {param.name}"
                )
                
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_command_injection_tests(self, api_spec: APISpec) -> List[TestCase]:
        """Generate command injection test cases."""
        command_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& whoami",
            "`id`",
            "$(uname -a)",
            "; rm -rf /",
            "| nc -l 4444"
        ]
        
        test_cases = []
        string_params = [p for p in api_spec.parameters if p.type == 'string']
        
        for i, payload in enumerate(command_payloads):
            for param in string_params:
                parameters = {}
                parameters[param.name] = payload
                
                for other_param in api_spec.parameters:
                    if other_param.name != param.name:
                        parameters[other_param.name] = self._generate_default_value(other_param.type)
                
                test_case = TestCase(
                    id=f"{api_spec.name}_command_injection_{param.name}_{i+1}",
                    api_name=api_spec.name,
                    parameters=parameters,
                    test_type=TestType.SECURITY,
                    expected_result="error",
                    generation_prompt=f"Command injection test for parameter {param.name}"
                )
                
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_ldap_injection_tests(self, api_spec: APISpec) -> List[TestCase]:
        """Generate LDAP injection test cases."""
        ldap_payloads = [
            "*)(uid=*",
            "*)(|(uid=*))",
            "admin)(&(password=*))",
            "*))%00",
            "*()|%26'",
            "*)(objectClass=*"
        ]
        
        test_cases = []
        string_params = [p for p in api_spec.parameters if p.type == 'string']
        
        for i, payload in enumerate(ldap_payloads):
            for param in string_params:
                parameters = {}
                parameters[param.name] = payload
                
                for other_param in api_spec.parameters:
                    if other_param.name != param.name:
                        parameters[other_param.name] = self._generate_default_value(other_param.type)
                
                test_case = TestCase(
                    id=f"{api_spec.name}_ldap_injection_{param.name}_{i+1}",
                    api_name=api_spec.name,
                    parameters=parameters,
                    test_type=TestType.SECURITY,
                    expected_result="error",
                    generation_prompt=f"LDAP injection test for parameter {param.name}"
                )
                
                test_cases.append(test_case)
        
        return test_cases
    
    def _generate_xml_injection_tests(self, api_spec: APISpec) -> List[TestCase]:
        """Generate XML injection test cases."""
        xml_payloads = [
            "<?xml version=\"1.0\"?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]><root>&test;</root>",
            "<script>alert('xss')</script>",
            "]]><script>alert('xss')</script>",
            "<?xml version=\"1.0\" encoding=\"ISO-8859-1\"?><!DOCTYPE foo [<!ELEMENT foo ANY ><!ENTITY xxe SYSTEM \"file:///etc/passwd\" >]><foo>&xxe;</foo>",
            "<![CDATA[<script>alert('xss')</script>]]>"
        ]
        
        test_cases = []
        string_params = [p for p in api_spec.parameters if p.type == 'string']
        
        for i, payload in enumerate(xml_payloads):
            for param in string_params:
                parameters = {}
                parameters[param.name] = payload
                
                for other_param in api_spec.parameters:
                    if other_param.name != param.name:
                        parameters[other_param.name] = self._generate_default_value(other_param.type)
                
                test_case = TestCase(
                    id=f"{api_spec.name}_xml_injection_{param.name}_{i+1}",
                    api_name=api_spec.name,
                    parameters=parameters,
                    test_type=TestType.SECURITY,
                    expected_result="error",
                    generation_prompt=f"XML injection test for parameter {param.name}"
                )
                
                test_cases.append(test_case)
        
        return test_cases
    
    def validate_test_case(self, test_case: TestCase, api_spec: Optional[APISpec] = None) -> ValidationResult:
        """Validate a generated test case with enhanced checks.
        
        Args:
            test_case: Test case to validate
            api_spec: Optional API specification for parameter validation
            
        Returns:
            ValidationResult with validation status
        """
        errors = []
        warnings = []
        
        # Basic validation
        if not test_case.id or not isinstance(test_case.id, str):
            errors.append("Invalid test case ID")
        
        if not test_case.api_name or not isinstance(test_case.api_name, str):
            errors.append("Invalid API name")
        
        if not isinstance(test_case.parameters, dict):
            errors.append("Parameters must be a dictionary")
        else:
            # Check for malformed JSON in parameters
            try:
                json.dumps(test_case.parameters)
            except (TypeError, ValueError) as e:
                errors.append(f"Invalid parameter format: {e}")
        
        # Validate against API spec if provided
        if api_spec and isinstance(test_case.parameters, dict):
            # Check required parameters
            required_params = {p.name for p in api_spec.parameters if p.required}
            provided_params = set(test_case.parameters.keys())
            
            missing_required = required_params - provided_params
            if missing_required:
                if test_case.test_type in [TestType.MALFORMED, TestType.EDGE]:
                    # Missing required params might be intentional for these test types
                    warnings.append(f"Missing required parameters: {missing_required}")
                else:
                    errors.append(f"Missing required parameters: {missing_required}")
            
            # Check for unknown parameters
            valid_params = {p.name for p in api_spec.parameters}
            unknown_params = provided_params - valid_params
            if unknown_params:
                if test_case.test_type == TestType.MALFORMED:
                    # Unknown params might be intentional for malformed tests
                    warnings.append(f"Unknown parameters: {unknown_params}")
                else:
                    warnings.append(f"Unknown parameters: {unknown_params}")
            
            # Validate parameter types (basic validation)
            for param_spec in api_spec.parameters:
                if param_spec.name in test_case.parameters:
                    value = test_case.parameters[param_spec.name]
                    type_valid = self._validate_parameter_type(value, param_spec.type)
                    if not type_valid and test_case.test_type not in [TestType.MALFORMED, TestType.EDGE]:
                        warnings.append(f"Parameter {param_spec.name} type mismatch")
        
        # Check test type
        if not isinstance(test_case.test_type, TestType):
            errors.append("Test type must be a valid TestType enum")
        
        # Check generation prompt
        if not test_case.generation_prompt or not test_case.generation_prompt.strip():
            warnings.append("Generation prompt is empty")
        
        is_valid = len(errors) == 0
        confidence = 1.0 - (len(errors) * 0.3 + len(warnings) * 0.1)
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            confidence=max(0.0, confidence)
        )
    
    def _validate_parameter_type(self, value: Any, expected_type: str) -> bool:
        """Validate parameter value against expected type.
        
        Args:
            value: Parameter value to validate
            expected_type: Expected parameter type
            
        Returns:
            True if type matches, False otherwise
        """
        type_mapping = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        expected_python_type = type_mapping.get(expected_type.lower())
        if expected_python_type is None:
            return True  # Unknown type, assume valid
        
        return isinstance(value, expected_python_type)
    
    def _generate_cases_for_api(self, api_spec: APISpec, count: int) -> List[TestCase]:
        """Generate test cases for a single API using all generation strategies."""
        test_cases = []
        
        # Calculate distribution of test types
        normal_count = int(count * self.config.normal_case_ratio)
        edge_count = int(count * self.config.edge_case_ratio)
        security_count = int(count * self.config.security_test_ratio)
        malformed_count = count - normal_count - edge_count - security_count
        
        self.logger.debug(f"Generating test cases for {api_spec.name}: "
                         f"normal={normal_count}, edge={edge_count}, "
                         f"security={security_count}, malformed={malformed_count}")
        
        # Generate normal cases using normal case generator
        if normal_count > 0:
            normal_cases = self.generate_normal_cases(api_spec, normal_count)
            test_cases.extend(normal_cases)
            self.logger.debug(f"Generated {len(normal_cases)} normal cases")
        
        # Generate edge cases using edge case generator
        if edge_count > 0:
            edge_cases = self.generate_edge_cases(api_spec, edge_count)
            test_cases.extend(edge_cases[:edge_count])
            self.logger.debug(f"Generated {len(edge_cases[:edge_count])} edge cases")
        
        # Generate security cases using security test generator
        if security_count > 0:
            security_cases = self.generate_security_test_cases(api_spec, security_count)
            test_cases.extend(security_cases[:security_count])
            self.logger.debug(f"Generated {len(security_cases[:security_count])} security cases")
        
        # Generate malformed cases using malformed input generator
        if malformed_count > 0:
            malformed_cases = self.generate_malformed_input_cases(api_spec, malformed_count)
            test_cases.extend(malformed_cases)
            self.logger.debug(f"Generated {len(malformed_cases)} malformed cases")
        
        return test_cases
    
    def generate_normal_cases(self, api_spec: APISpec, count: int) -> List[TestCase]:
        """Normal case generator following API documentation.
        
        This generator creates test cases that represent typical, valid usage
        patterns based on the API documentation. It follows the documented
        parameter types, constraints, and examples.
        
        Args:
            api_spec: API specification to generate test cases for
            count: Number of normal test cases to generate
            
        Returns:
            List of normal test cases
        """
        self.logger.debug(f"Generating {count} normal test cases for {api_spec.name}")
        
        test_cases = []
        
        # Try LLM generation first for more diverse cases
        llm_count = min(count, count // 2)  # Use LLM for half the cases
        if llm_count > 0:
            try:
                prompt = self._build_prompt_from_template('normal', api_spec, llm_count)
                response = self._make_llm_request(prompt)
                llm_cases = self._parse_llm_response(response, api_spec, TestType.NORMAL)
                test_cases.extend(llm_cases[:llm_count])
                self.logger.debug(f"Generated {len(llm_cases)} normal cases via LLM")
            except Exception as e:
                self.logger.warning(f"LLM normal case generation failed: {e}")
        
        # Generate systematic normal cases for remaining count
        remaining_count = count - len(test_cases)
        if remaining_count > 0:
            systematic_cases = self._generate_systematic_normal_cases(api_spec, remaining_count)
            test_cases.extend(systematic_cases)
        
        return test_cases[:count]
    
    def _generate_systematic_normal_cases(self, api_spec: APISpec, count: int) -> List[TestCase]:
        """Generate systematic normal test cases based on API specification."""
        test_cases = []
        
        for i in range(count):
            parameters = {}
            
            # Generate valid parameters based on specification
            for param in api_spec.parameters:
                if param.required or random.random() < 0.8:  # Include most optional params
                    if param.examples:
                        # Use examples from documentation when available
                        parameters[param.name] = random.choice(param.examples)
                    else:
                        # Generate appropriate values based on type
                        parameters[param.name] = self._generate_realistic_value(param)
            
            test_case = TestCase(
                id=f"{api_spec.name}_normal_systematic_{i+1}",
                api_name=api_spec.name,
                parameters=parameters,
                test_type=TestType.NORMAL,
                expected_result="success",
                generation_prompt=f"Systematic normal test case following API documentation"
            )
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_realistic_value(self, param: ParameterSpec) -> Any:
        """Generate realistic values based on parameter specification."""
        if param.type == 'string':
            if 'email' in param.name.lower():
                return f"user{random.randint(1, 1000)}@example.com"
            elif 'name' in param.name.lower():
                names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
                return random.choice(names)
            elif 'id' in param.name.lower():
                return f"id_{random.randint(1000, 9999)}"
            else:
                return f"test_value_{random.randint(1, 100)}"
        elif param.type == 'integer':
            if 'age' in param.name.lower():
                return random.randint(18, 80)
            elif 'count' in param.name.lower() or 'size' in param.name.lower():
                return random.randint(1, 100)
            else:
                return random.randint(1, 1000)
        elif param.type == 'number':
            return round(random.uniform(0.1, 100.0), 2)
        elif param.type == 'boolean':
            return random.choice([True, False])
        elif param.type == 'array':
            return [f"item_{i}" for i in range(random.randint(1, 5))]
        elif param.type == 'object':
            return {"key": "value", "nested": {"data": random.randint(1, 100)}}
        else:
            return self._generate_default_value(param.type)
    
    def generate_malformed_input_cases(self, api_spec: APISpec, count: int) -> List[TestCase]:
        """Malformed input generator for error testing.
        
        This generator creates test cases with intentionally malformed,
        invalid, or corrupted inputs to test error handling and input
        validation. It includes wrong data types, invalid syntax,
        missing required fields, and corrupted values.
        
        Args:
            api_spec: API specification
            count: Number of malformed test cases to generate
            
        Returns:
            List of malformed test cases
        """
        self.logger.debug(f"Generating {count} malformed input test cases for {api_spec.name}")
        
        test_cases = []
        
        # Try LLM generation first for creative malformed inputs
        llm_count = count // 3  # Use LLM for 1/3 of cases
        if llm_count > 0:
            try:
                prompt = self._build_prompt_from_template('malformed', api_spec, llm_count)
                response = self._make_llm_request(prompt)
                llm_cases = self._parse_llm_response(response, api_spec, TestType.MALFORMED)
                test_cases.extend(llm_cases[:llm_count])
                self.logger.debug(f"Generated {len(llm_cases)} malformed cases via LLM")
            except Exception as e:
                self.logger.warning(f"LLM malformed case generation failed: {e}")
        
        # Generate systematic malformed cases for remaining count
        remaining_count = count - len(test_cases)
        if remaining_count > 0:
            systematic_cases = self._generate_comprehensive_malformed_cases(api_spec, remaining_count)
            test_cases.extend(systematic_cases)
        
        return test_cases[:count]
    
    def _generate_comprehensive_malformed_cases(self, api_spec: APISpec, count: int) -> List[TestCase]:
        """Generate comprehensive malformed test cases covering all malformation types."""
        test_cases = []
        
        # Define malformation strategies
        malformation_strategies = [
            ('wrong_types', self._malform_wrong_types),
            ('missing_required', self._malform_missing_required),
            ('invalid_params', self._malform_invalid_params),
            ('corrupted_values', self._malform_corrupted_values),
            ('structure_errors', self._malform_structure_errors),
            ('encoding_issues', self._malform_encoding_issues),
            ('size_violations', self._malform_size_violations),
            ('format_violations', self._malform_format_violations)
        ]
        
        # Distribute test cases across different malformation types
        cases_per_type = max(1, count // len(malformation_strategies))
        
        for malform_type, strategy_func in malformation_strategies:
            if len(test_cases) >= count:
                break
            
            try:
                for i in range(cases_per_type):
                    if len(test_cases) >= count:
                        break
                    
                    malformed_params = strategy_func(api_spec)
                    
                    test_case = TestCase(
                        id=f"{api_spec.name}_malformed_{malform_type}_{i+1}",
                        api_name=api_spec.name,
                        parameters=malformed_params,
                        test_type=TestType.MALFORMED,
                        expected_result="error",
                        generation_prompt=f"Malformed test case using {malform_type} strategy"
                    )
                    
                    test_cases.append(test_case)
                    
                self.logger.debug(f"Generated {min(cases_per_type, count - len(test_cases) + cases_per_type)} {malform_type} malformed cases")
            except Exception as e:
                self.logger.warning(f"Failed to generate {malform_type} malformed cases: {e}")
        
        return test_cases[:count]
    

    def _malform_wrong_types(self, api_spec: APISpec) -> Dict[str, Any]:
        """Create malformed parameters with wrong types."""
        malformed_params = {}
        
        for param in api_spec.parameters:
            if random.random() < 0.8:  # 80% chance to include parameter
                malformed_params[param.name] = self._create_malformed_value(param.type)
        
        return malformed_params
    
    def _malform_missing_required(self, api_spec: APISpec) -> Dict[str, Any]:
        """Create malformed parameters by omitting required fields."""
        malformed_params = {}
        required_params = [p for p in api_spec.parameters if p.required]
        
        # Include some but not all required parameters
        if required_params:
            include_count = max(0, len(required_params) - 1)
            included_required = random.sample(required_params, include_count)
            
            for param in included_required:
                malformed_params[param.name] = self._generate_default_value(param.type)
        
        # Add optional parameters
        optional_params = [p for p in api_spec.parameters if not p.required]
        for param in optional_params:
            if random.random() < 0.5:
                malformed_params[param.name] = self._generate_default_value(param.type)
        
        return malformed_params
    
    def _malform_invalid_params(self, api_spec: APISpec) -> Dict[str, Any]:
        """Create malformed parameters with invalid parameter names."""
        malformed_params = {}
        
        # Add valid parameters
        for param in api_spec.parameters:
            if random.random() < 0.7:
                malformed_params[param.name] = self._generate_default_value(param.type)
        
        # Add invalid parameters
        invalid_param_names = [
            'invalid_param', '123invalid', 'param-with-dashes', 
            'param with spaces', 'UPPERCASE_PARAM', '!@#$%'
        ]
        
        for invalid_name in random.sample(invalid_param_names, random.randint(1, 3)):
            malformed_params[invalid_name] = "unexpected_value"
        
        return malformed_params
    
    def _malform_corrupted_values(self, api_spec: APISpec) -> Dict[str, Any]:
        """Create malformed parameters with corrupted values."""
        malformed_params = {}
        
        for param in api_spec.parameters:
            if random.random() < 0.8:
                # Create corrupted version of expected type
                if param.type == 'string':
                    malformed_params[param.name] = "\x00\x01\x02corrupted"
                elif param.type in ['integer', 'number']:
                    malformed_params[param.name] = "not_a_number"
                elif param.type == 'boolean':
                    malformed_params[param.name] = "maybe"
                elif param.type == 'array':
                    malformed_params[param.name] = "not_an_array"
                elif param.type == 'object':
                    malformed_params[param.name] = "not_an_object"
                else:
                    malformed_params[param.name] = None
        
        return malformed_params
    
    def _malform_structure_errors(self, api_spec: APISpec) -> Dict[str, Any]:
        """Create malformed parameters with structural errors."""
        # This creates parameters that might cause JSON parsing issues
        malformed_params = {}
        
        for param in api_spec.parameters:
            if random.random() < 0.6:
                if param.type == 'string':
                    # Strings with JSON-breaking characters
                    malformed_params[param.name] = '{"unclosed": "json'
                elif param.type == 'object':
                    # Invalid object structure
                    malformed_params[param.name] = {"circular": "reference"}
                else:
                    malformed_params[param.name] = self._generate_default_value(param.type)
        
        return malformed_params
    
    def _malform_encoding_issues(self, api_spec: APISpec) -> Dict[str, Any]:
        """Create malformed parameters with encoding issues."""
        malformed_params = {}
        
        for param in api_spec.parameters:
            if random.random() < 0.7:
                if param.type == 'string':
                    # Various encoding issues
                    encoding_issues = [
                        "\xff\xfe\x00\x00invalid_utf8",  # Invalid UTF-8
                        "test\x00null_byte",  # Null byte injection
                        "test\r\n\r\nHTTP/1.1 200 OK",  # HTTP response splitting
                        "\x1b[31mANSI_escape\x1b[0m",  # ANSI escape sequences
                        "test\u202e\u202dreversed",  # Unicode direction override
                    ]
                    malformed_params[param.name] = random.choice(encoding_issues)
                else:
                    malformed_params[param.name] = self._generate_default_value(param.type)
        
        return malformed_params
    
    def _malform_size_violations(self, api_spec: APISpec) -> Dict[str, Any]:
        """Create malformed parameters that violate size constraints."""
        malformed_params = {}
        
        for param in api_spec.parameters:
            if random.random() < 0.8:
                if param.type == 'string':
                    # Extremely long strings
                    size = random.choice([10000, 100000, 1000000])
                    malformed_params[param.name] = "A" * size
                elif param.type == 'array':
                    # Extremely large arrays
                    size = random.choice([1000, 10000])
                    malformed_params[param.name] = [f"item_{i}" for i in range(size)]
                elif param.type == 'object':
                    # Objects with many keys
                    size = random.choice([1000, 5000])
                    malformed_params[param.name] = {f"key_{i}": f"value_{i}" for i in range(size)}
                elif param.type in ['integer', 'number']:
                    # Numbers outside typical ranges
                    malformed_params[param.name] = random.choice([
                        999999999999999999999,  # Very large number
                        -999999999999999999999,  # Very large negative
                        float('inf'),  # Infinity
                        float('-inf'),  # Negative infinity
                        float('nan')  # Not a number
                    ])
                else:
                    malformed_params[param.name] = self._generate_default_value(param.type)
        
        return malformed_params
    
    def _malform_format_violations(self, api_spec: APISpec) -> Dict[str, Any]:
        """Create malformed parameters that violate expected formats."""
        malformed_params = {}
        
        for param in api_spec.parameters:
            if random.random() < 0.8:
                if param.type == 'string':
                    # Format violations based on parameter name
                    if 'email' in param.name.lower():
                        malformed_params[param.name] = random.choice([
                            "not_an_email",
                            "@invalid.com",
                            "user@",
                            "user@.com",
                            "user..name@domain.com"
                        ])
                    elif 'url' in param.name.lower() or 'uri' in param.name.lower():
                        malformed_params[param.name] = random.choice([
                            "not_a_url",
                            "http://",
                            "ftp://invalid",
                            "javascript:alert(1)"
                        ])
                    elif 'date' in param.name.lower():
                        malformed_params[param.name] = random.choice([
                            "not_a_date",
                            "2023-13-45",  # Invalid date
                            "32/25/2023",  # Invalid format
                            "2023-02-30"   # Non-existent date
                        ])
                    elif 'phone' in param.name.lower():
                        malformed_params[param.name] = random.choice([
                            "not_a_phone",
                            "123",
                            "+++invalid+++",
                            "phone_number"
                        ])
                    else:
                        malformed_params[param.name] = "format_violation_" + param.name
                else:
                    malformed_params[param.name] = self._generate_default_value(param.type)
        
        return malformed_params
    
    def _build_prompt_from_template(self, template_type: str, api_spec: APISpec, 
                                   count: Optional[int] = None) -> str:
        """Build prompt from template for specific test case type.
        
        Args:
            template_type: Type of template ('normal', 'security', 'edge', 'malformed')
            api_spec: API specification
            count: Number of test cases to generate (for applicable templates)
            
        Returns:
            Formatted prompt string
        """
        if template_type not in self._prompt_templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        template = self._prompt_templates[template_type]
        
        # Format parameters section
        parameters_text = ""
        for param in api_spec.parameters:
            parameters_text += f"- {param.name} ({param.type}): {param.description}"
            if param.required:
                parameters_text += " [REQUIRED]"
            if param.examples:
                parameters_text += f" Examples: {', '.join(map(str, param.examples))}"
            parameters_text += "\n"
        
        # Format examples section
        examples_section = ""
        if api_spec.examples:
            examples_section = "Existing Examples:\n"
            for example in api_spec.examples:
                examples_section += f"- {example}\n"
        
        # Format the template
        format_args: Dict[str, Any] = {
            'api_name': api_spec.name,
            'description': api_spec.description,
            'parameters': parameters_text,
            'examples_section': examples_section
        }
        
        if count is not None:
            format_args['count'] = str(count)
        
        return template.format(**format_args)
    

    
    def _parse_llm_response(self, response: str, api_spec: APISpec, test_type: TestType) -> List[TestCase]:
        """Parse LLM response into test cases with enhanced validation.
        
        Args:
            response: Raw LLM response text
            api_spec: API specification for validation
            test_type: Type of test cases being parsed
            
        Returns:
            List of parsed and validated test cases
        """
        test_cases = []
        
        try:
            # Try to extract JSON from response
            json_match = self._extract_json_from_response(response)
            if not json_match:
                self.logger.warning("No valid JSON found in LLM response")
                return test_cases
            
            test_data = json.loads(json_match)
            
            if not isinstance(test_data, list):
                test_data = [test_data]
            
            for i, case_data in enumerate(test_data):
                if not isinstance(case_data, dict):
                    self.logger.warning(f"Skipping non-dict test case {i}")
                    continue
                
                # Validate required fields
                if 'parameters' not in case_data:
                    self.logger.warning(f"Test case {i} missing parameters field")
                    continue
                
                # Create test case
                test_case = TestCase(
                    id=f"{api_spec.name}_{test_type.value}_{i+1}",
                    api_name=api_spec.name,
                    parameters=case_data.get('parameters', {}),
                    test_type=test_type,
                    expected_result=case_data.get('expected_result', 'success'),
                    generation_prompt=f"LLM generated {test_type.value} test case"
                )
                
                # Validate test case
                validation_result = self.validate_test_case(test_case, api_spec)
                if validation_result.is_valid:
                    test_cases.append(test_case)
                else:
                    self.logger.warning(f"Invalid test case {i}: {validation_result.errors}")
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from LLM response: {e}")
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
        
        return test_cases

    def _parse_document_response(self, response: str, count: int) -> List[TestCase]:
        test_cases: List[TestCase] = []

        json_match = self._extract_json_from_response(response)
        if not json_match:
            self.logger.warning("No valid JSON found in document response")
            return self._parse_document_code_blocks(response, count)

        try:
            test_data = json.loads(json_match)
        except json.JSONDecodeError as exc:
            self.logger.warning(f"Failed to parse document JSON: {exc}")
            return self._parse_document_code_blocks(response, count)

        if not isinstance(test_data, list):
            test_data = [test_data]

        if not test_data:
            return self._parse_document_code_blocks(response, count)

        for i, case_data in enumerate(test_data):
            if not isinstance(case_data, dict):
                continue
            java_source = case_data.get("java_source")
            if not isinstance(java_source, str) or not java_source.strip():
                continue
            java_source = self._ensure_compilable_source(self._fix_java_syntax(java_source))
            case_id = case_data.get("id") or f"document_case_{i+1}"
            test_cases.append(
                TestCase(
                    id=str(case_id),
                    api_name="document",
                    parameters={"java_source": java_source},
                    test_type=TestType.NORMAL,
                    expected_result="compile",
                    generation_prompt="document_generation",
                )
            )

            if len(test_cases) >= count:
                break

        return test_cases

    def _parse_document_code_blocks(self, response: str, count: int) -> List[TestCase]:
        import re

        test_cases: List[TestCase] = []
        def extract_class(text: str) -> Optional[str]:
            return self._extract_java_class(text)

        blocks = re.findall(r"```(?:java)?\s*([\s\S]*?)```", response, re.IGNORECASE)
        for i, block in enumerate(blocks):
            java_source = block.strip()
            if not java_source:
                continue
            java_source = self._ensure_compilable_source(self._fix_java_syntax(java_source))
            extracted = extract_class(java_source)
            if not extracted:
                continue
            test_cases.append(
                TestCase(
                    id=f"document_block_{i+1}",
                    api_name="document",
                    parameters={"java_source": extracted},
                    test_type=TestType.NORMAL,
                    expected_result="compile",
                    generation_prompt="document_generation_block",
                )
            )
            if len(test_cases) >= count:
                break

        if not test_cases:
            self.logger.warning("No Java code blocks found in document response")
            candidate = response
            if "```" in candidate:
                candidate = candidate.split("```", 1)[0]
            if "/src/" in candidate:
                candidate = candidate.split("/src/", 1)[0]
            extracted = extract_class(candidate)
            fallback = None
            if not extracted:
                fallback = self._ensure_compilable_source(self._fix_java_syntax(candidate))
            if extracted or (fallback and "public class" in fallback):
                test_cases.append(
                    TestCase(
                        id="document_raw_1",
                        api_name="document",
                        parameters={"java_source": self._ensure_compilable_source(self._fix_java_syntax(extracted)) if extracted else fallback},
                        test_type=TestType.NORMAL,
                        expected_result="compile",
                        generation_prompt="document_generation_raw",
                    )
                )
                return test_cases
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = f"output/document_generation_response_{timestamp}.txt"
            try:
                with open(out_path, "w", encoding="utf-8") as handle:
                    handle.write(response)
                self.logger.info(f"Saved document generation response to {out_path}")
            except OSError as exc:
                self.logger.warning(f"Failed to save document response: {exc}")
        return test_cases

    def _extract_java_class(self, text: str) -> Optional[str]:
        return self._extract_java_classes(text)[0] if self._extract_java_classes(text) else None

    def _extract_java_classes(self, text: str) -> List[str]:
        classes: List[str] = []
        cursor = 0
        while True:
            start = text.find("public class", cursor)
            if start == -1:
                break
            brace_start = text.find("{", start)
            if brace_start == -1:
                break
            depth = 0
            end = None
            for idx in range(brace_start, len(text)):
                char = text[idx]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            if end is None:
                snippet = text[start:].strip()
                brace_balance = snippet.count("{") - snippet.count("}")
                if brace_balance > 0:
                    snippet = snippet + ("\n" + "}" * brace_balance)
                classes.append(snippet.strip())
                break
            classes.append(text[start:end].strip())
            cursor = end
        return classes

    def _fix_java_syntax(self, java_source: str) -> str:
        source = java_source
        source = re.sub(
            r"case\s+([A-Za-z_][\w<>]*)\s+(\w+)\s*&&",
            r"case \1 \2 when",
            source,
        )
        return source

    def _ensure_compilable_source(self, java_source: str) -> str:
        source = java_source.strip()
        if "public class" in source and "main" in source:
            return source

        lines = source.splitlines()
        import_lines = []
        content_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("package "):
                import_lines.append(line)
            else:
                content_lines.append(line)
        content = "\n".join(content_lines).strip()

        if not content:
            body = (
                "public class GeneratedCase {\n"
                "  public static void main(String[] args) {\n"
                "  }\n"
                "}\n"
            )
            if import_lines:
                return "\n".join(import_lines) + "\n\n" + body
            return body

        if "public static void main" not in content:
            body = (
                "public class GeneratedCase {\n"
                "  public static void main(String[] args) throws Exception {\n"
                f"    {content}\n"
                "  }\n"
                "}\n"
            )
        elif "public class" not in content:
            body = f"public class GeneratedCase {{\n{content}\n}}\n"
        else:
            body = content

        if import_lines:
            return "\n".join(import_lines) + "\n\n" + body
        return body
    
    def _extract_json_from_response(self, response: str) -> Optional[str]:
        """Extract JSON content from LLM response with improved parsing.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Extracted JSON string or None if not found
        """
        import re
        
        # Clean up the response
        response = response.strip()
        
        # Try to find JSON array first (preferred for multiple test cases)
        array_patterns = [
            r'\[[\s\S]*\]',  # Greedy match for array
            r'```json\s*(\[[\s\S]*?\])\s*```',  # JSON in code blocks
            r'```\s*(\[[\s\S]*?\])\s*```',  # Array in code blocks
        ]
        
        for pattern in array_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                # Validate it's actually JSON
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    continue
        
        # Try to find JSON object (fallback for single test case)
        object_patterns = [
            r'\{[\s\S]*\}',  # Greedy match for object
            r'```json\s*(\{[\s\S]*?\})\s*```',  # JSON in code blocks
            r'```\s*(\{[\s\S]*?\})\s*```',  # Object in code blocks
        ]
        
        for pattern in object_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                json_str = match.group(1) if match.groups() else match.group(0)
                # Validate it's actually JSON
                try:
                    json.loads(json_str)
                    return json_str
                except json.JSONDecodeError:
                    continue
        
        # Last resort: try to find any valid JSON structure
        lines = response.split('\n')
        json_lines = []
        in_json = False
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('[') or stripped.startswith('{'):
                in_json = True
                json_lines = [line]
            elif in_json:
                json_lines.append(line)
                if stripped.endswith(']') or stripped.endswith('}'):
                    try:
                        json_str = '\n'.join(json_lines)
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        in_json = False
                        json_lines = []
        
        return None
    
    def _fallback_normal_tests(self, api_spec: APISpec, count: int) -> List[TestCase]:
        """Generate fallback normal test cases when LLM fails."""
        test_cases = []
        
        for i in range(count):
            parameters = {}
            
            for param in api_spec.parameters:
                if param.required or random.random() < 0.8:
                    parameters[param.name] = self._generate_default_value(param.type)
            
            test_case = TestCase(
                id=f"{api_spec.name}_normal_fallback_{i+1}",
                api_name=api_spec.name,
                parameters=parameters,
                test_type=TestType.NORMAL,
                expected_result="success",
                generation_prompt="Fallback normal test case"
            )
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _fallback_security_tests(self, api_spec: APISpec) -> List[TestCase]:
        """Generate fallback security test cases."""
        test_cases = []
        
        # Common security payloads
        sql_payloads = ["'; DROP TABLE users; --", "' OR '1'='1", "admin'--"]
        xss_payloads = ["<script>alert('xss')</script>", "javascript:alert(1)", "<img src=x onerror=alert(1)>"]
        
        for i, payload in enumerate(sql_payloads + xss_payloads):
            parameters = {}
            
            # Apply payload to string parameters
            for param in api_spec.parameters:
                if param.type == 'string':
                    parameters[param.name] = payload
                else:
                    parameters[param.name] = self._generate_default_value(param.type)
            
            test_case = TestCase(
                id=f"{api_spec.name}_security_fallback_{i+1}",
                api_name=api_spec.name,
                parameters=parameters,
                test_type=TestType.SECURITY,
                expected_result="error",
                generation_prompt="Fallback security test case"
            )
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _create_string_edge_cases(self, api_spec: APISpec, param) -> List[TestCase]:
        """Create edge cases for string parameters."""
        edge_values = [
            "",  # Empty string
            " ",  # Single space
            "a" * 1000,  # Very long string
            "null",  # String "null"
            "undefined",  # String "undefined"
            "\n\r\t",  # Whitespace characters
            "🚀🎉💻",  # Unicode/emoji
        ]
        
        test_cases = []
        
        for i, value in enumerate(edge_values):
            parameters = {}
            
            # Set edge value for target parameter
            parameters[param.name] = value
            
            # Set default values for other parameters
            for other_param in api_spec.parameters:
                if other_param.name != param.name:
                    parameters[other_param.name] = self._generate_default_value(other_param.type)
            
            test_case = TestCase(
                id=f"{api_spec.name}_edge_string_{param.name}_{i+1}",
                api_name=api_spec.name,
                parameters=parameters,
                test_type=TestType.EDGE,
                expected_result="success_or_error",
                generation_prompt=f"Edge case for string parameter {param.name}"
            )
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _create_numeric_edge_cases(self, api_spec: APISpec, param) -> List[TestCase]:
        """Create edge cases for numeric parameters."""
        edge_values = [
            0,  # Zero
            -1,  # Negative
            1,  # Positive
            999999999,  # Large positive
            -999999999,  # Large negative
            0.1,  # Small decimal
            -0.1,  # Small negative decimal
        ]
        
        test_cases = []
        
        for i, value in enumerate(edge_values):
            parameters = {}
            
            parameters[param.name] = value
            
            for other_param in api_spec.parameters:
                if other_param.name != param.name:
                    parameters[other_param.name] = self._generate_default_value(other_param.type)
            
            test_case = TestCase(
                id=f"{api_spec.name}_edge_numeric_{param.name}_{i+1}",
                api_name=api_spec.name,
                parameters=parameters,
                test_type=TestType.EDGE,
                expected_result="success_or_error",
                generation_prompt=f"Edge case for numeric parameter {param.name}"
            )
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _create_array_edge_cases(self, api_spec: APISpec, param) -> List[TestCase]:
        """Create edge cases for array parameters."""
        edge_values = [
            [],  # Empty array
            [None],  # Array with null
            ["a"] * 1000,  # Very large array
            [1, "string", True, None],  # Mixed types
            [[1, 2], [3, 4]],  # Nested arrays
        ]
        
        test_cases = []
        
        for i, value in enumerate(edge_values):
            parameters = {}
            
            parameters[param.name] = value
            
            for other_param in api_spec.parameters:
                if other_param.name != param.name:
                    parameters[other_param.name] = self._generate_default_value(other_param.type)
            
            test_case = TestCase(
                id=f"{api_spec.name}_edge_array_{param.name}_{i+1}",
                api_name=api_spec.name,
                parameters=parameters,
                test_type=TestType.EDGE,
                expected_result="success_or_error",
                generation_prompt=f"Edge case for array parameter {param.name}"
            )
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _create_boolean_edge_cases(self, api_spec: APISpec, param) -> List[TestCase]:
        """Create edge cases for boolean parameters."""
        edge_values = [
            True,
            False,
            1,  # Truthy number
            0,  # Falsy number
            "true",  # String representation
            "false",  # String representation
        ]
        
        test_cases = []
        
        for i, value in enumerate(edge_values):
            parameters = {}
            
            parameters[param.name] = value
            
            for other_param in api_spec.parameters:
                if other_param.name != param.name:
                    parameters[other_param.name] = self._generate_default_value(other_param.type)
            
            test_case = TestCase(
                id=f"{api_spec.name}_edge_boolean_{param.name}_{i+1}",
                api_name=api_spec.name,
                parameters=parameters,
                test_type=TestType.EDGE,
                expected_result="success_or_error",
                generation_prompt=f"Edge case for boolean parameter {param.name}"
            )
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _create_object_edge_cases(self, api_spec: APISpec, param) -> List[TestCase]:
        """Create edge cases for object parameters."""
        edge_values = [
            {},  # Empty object
            {"key": None},  # Object with null value
            {"nested": {"deep": {"value": 42}}},  # Deeply nested object
            {f"key_{i}": f"value_{i}" for i in range(100)},  # Large object
            {"special_chars": "!@#$%^&*()"},  # Special characters
        ]
        
        test_cases = []
        
        for i, value in enumerate(edge_values):
            parameters = {}
            
            parameters[param.name] = value
            
            for other_param in api_spec.parameters:
                if other_param.name != param.name:
                    parameters[other_param.name] = self._generate_default_value(other_param.type)
            
            test_case = TestCase(
                id=f"{api_spec.name}_edge_object_{param.name}_{i+1}",
                api_name=api_spec.name,
                parameters=parameters,
                test_type=TestType.EDGE,
                expected_result="success_or_error",
                generation_prompt=f"Edge case for object parameter {param.name}"
            )
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_default_value(self, param_type: str) -> Any:
        """Generate a default value for a parameter type."""
        defaults = {
            'string': 'test_value',
            'integer': 42,
            'number': 3.14,
            'boolean': True,
            'array': ['item1', 'item2'],
            'object': {'key': 'value'}
        }
        
        return defaults.get(param_type, 'default_value')
    
    def _create_malformed_value(self, param_type: str) -> Any:
        """Create a malformed value for a parameter type."""
        malformed_values = {
            'string': 123,  # Number instead of string
            'integer': 'not_a_number',  # String instead of number
            'number': 'invalid',  # String instead of number
            'boolean': 'maybe',  # String instead of boolean
            'array': 'not_an_array',  # String instead of array
            'object': 'not_an_object'  # String instead of object
        }
        
        return malformed_values.get(param_type, None)
    
    def clear_cache(self) -> None:
        """Clear the LLM response cache."""
        self.response_cache.clear()
        self.logger.info("LLM response cache cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        cache_stats = self.response_cache.get_statistics()
        cache_stats.update({
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'last_generation_time': self.last_generation_time
        })
        return cache_stats
    
    def optimize_performance(self) -> Dict[str, Any]:
        """Perform performance optimizations and return statistics.
        
        Returns:
            Dictionary with optimization results
        """
        # Clean up expired cache entries
        expired_cleaned = self.response_cache.cleanup_expired()
        
        # Get current statistics
        stats = self.get_cache_statistics()
        stats['expired_entries_cleaned'] = expired_cleaned
        
        self.logger.debug(f"Performance optimization completed: {stats}")
        return stats
    
    def learn_from_execution_results(self, test_cases: List[TestCase], 
                                   execution_results: List[ExecutionResult]) -> None:
        """Learn from test execution results to improve future generation.
        
        Args:
            test_cases: Test cases that were executed
            execution_results: Results from executing the test cases
        """
        self.adaptive_generator.learn_from_results(test_cases, execution_results)
        self.logger.debug(f"Learned from {len(test_cases)} test execution results")
    
    def generate_adaptive_test_cases(self, api_specs: List[APISpec], 
                                   count: int,
                                   strategy: Optional[AdaptationStrategy] = None) -> List[TestCase]:
        """Generate test cases using adaptive learning.
        
        Args:
            api_specs: List of API specifications
            count: Total number of test cases to generate
            strategy: Adaptation strategy to use
            
        Returns:
            List of adaptively generated test cases
        """
        if not api_specs:
            self.logger.warning("No API specifications provided for adaptive generation")
            return []
        
        test_cases = []
        cases_per_api = max(1, count // len(api_specs))
        
        for api_spec in api_specs:
            adaptive_cases = self.adaptive_generator.generate_adaptive_test_cases(
                api_spec, cases_per_api, strategy
            )
            test_cases.extend(adaptive_cases)
        
        # Trim to exact count if needed
        if len(test_cases) > count:
            test_cases = test_cases[:count]
        
        self.logger.info(f"Generated {len(test_cases)} adaptive test cases")
        return test_cases
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """Get adaptive generation statistics.
        
        Returns:
            Dictionary with adaptation statistics
        """
        return self.adaptive_generator.get_adaptation_statistics()
    
    def set_adaptation_strategy(self, strategy: AdaptationStrategy) -> None:
        """Set the adaptation strategy for test generation.
        
        Args:
            strategy: Adaptation strategy to use
        """
        self.adaptive_generator.current_strategy = strategy
        self.logger.info(f"Adaptation strategy set to: {strategy.value}")
    
    def export_learned_patterns(self) -> Dict[str, Any]:
        """Export learned patterns for analysis or transfer.
        
        Returns:
            Dictionary with all learned patterns
        """
        return self.adaptive_generator.export_learned_patterns()
    
    def import_learned_patterns(self, pattern_data: Dict[str, Any]) -> int:
        """Import previously learned patterns.
        
        Args:
            pattern_data: Pattern data from export_learned_patterns()
            
        Returns:
            Number of patterns imported
        """
        return self.adaptive_generator.import_learned_patterns(pattern_data)
