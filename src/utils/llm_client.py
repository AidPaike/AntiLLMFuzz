"""LLM client for making API calls to language models.

This module provides a unified interface for calling LLM APIs with
retry logic, error handling, and configuration management.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import requests
from src.utils.logger import get_logger


# Constants
_TEST_CONNECTION_PROMPT = "Hello!"
_TEST_CONNECTION_MAX_TOKENS = 10


class LLMClient:
    """Client for making LLM API calls with retry logic.
    
    Can be used as a context manager for better resource management:
        with LLMClient() as client:
            response = client.simple_completion("Hello")
    """

    def detect_model_family(self) -> str:
        """Best-effort model family detection for CRM profiles."""
        model_name = (self.model or "").lower()
        if "llama" in model_name:
            return "llama3"
        if "qwen" in model_name or "chatml" in model_name:
            return "qwen"
        if "deepseek" in model_name:
            return "deepseek"
        if "gpt" in model_name or "openai" in model_name:
            return "openai"
        return "generic"

    def get_stop_token_profile(self) -> str:
        """Return CRM stop-token profile based on model family."""
        return self.detect_model_family()

    
    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_backoff: Optional[float] = None
    ):
        """Initialize LLM client.
        
        Args:
            endpoint: API endpoint URL (default: from config)
            api_key: API key (default: from config or env var)
            model: Model name (default: from config)
            timeout: Request timeout in seconds (default: from config)
            max_retries: Maximum retry attempts (default: from config)
            retry_backoff: Retry backoff factor (default: from config)
        """
        self.logger = get_logger()
        
        # Load configuration - avoid circular import
        try:
            from src.utils.config_loader import get_config_loader
            config = get_config_loader()
            api_config = config.config.get('api', {})
        except ImportError:
            # Fallback to default configuration
            api_config = {}
        
        # Set parameters (use provided values or fall back to config)
        self.endpoint = endpoint or api_config.get('endpoint', 'https://api.openai.com/v1/chat/completions')
        self.model = model or api_config.get('model', 'gpt-4')
        self.timeout = timeout or api_config.get('timeout', 60)
        self.max_retries = max_retries or api_config.get('max_retries', 3)
        self.retry_backoff = retry_backoff or api_config.get('retry_backoff', 2.0)
        
        # Handle API key (check env var first, then config)
        config_api_key = api_config.get('api_key', '')
        if config_api_key.startswith('${') and config_api_key.endswith('}'):
            # Extract env var name
            env_var_name = config_api_key[2:-1]
            self.api_key = api_key or os.environ.get(env_var_name, '')
        else:
            self.api_key = api_key or config_api_key
        
        if not self.api_key and not self._is_local_endpoint():
            self.logger.warning("No API key provided. Set HUIYAN_API_KEY environment variable or provide api_key parameter.")

    def _is_local_endpoint(self) -> bool:
        parsed = urlparse(self.endpoint or "")
        if parsed.scheme not in {"http", "https"}:
            return False
        host = (parsed.hostname or "").lower()
        if host not in {"localhost", "127.0.0.1"}:
            return False
        if parsed.port != 11434:
            return False
        return True
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Currently no cleanup needed, but this allows for future resource management
        return False
    
    def _make_request_with_retry(
        self,
        headers: Dict[str, str],
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic.
        
        Args:
            headers: Request headers
            payload: Request payload
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.exceptions.RequestException: If request fails after retries
            
        Note:
            Future enhancement: Add response caching for identical requests
            to reduce API calls and improve performance.
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Making API request (attempt {attempt + 1}/{self.max_retries})")
                
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse response
                result = response.json()
                
                self.logger.debug("API request successful")
                return result
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                self.logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries}): {e}")
                
            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code
                
                # Don't retry on client errors (4xx)
                if 400 <= status_code < 500:
                    self.logger.error(f"Client error {status_code}: {e.response.text}")
                    raise

                response_text = e.response.text if e.response is not None else ""
                if response_text:
                    self.logger.warning(
                        f"Server error {status_code} response: {response_text[:500]}"
                    )
                self.logger.warning(f"HTTP error {status_code} (attempt {attempt + 1}/{self.max_retries}): {e}")
                
            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                wait_time = self.retry_backoff ** attempt
                self.logger.debug(f"Waiting {wait_time:.1f}s before retry...")
                time.sleep(wait_time)
        
        # All retries failed
        self.logger.error(f"All {self.max_retries} retry attempts failed")
        if last_exception is None:
            raise requests.exceptions.RequestException("LLM request failed")
        raise last_exception

    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Make a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            API response as dictionary
            
        Raises:
            requests.exceptions.RequestException: If request fails after retries
            ValueError: If API key is not set or parameters are invalid
        """
        if not self.api_key and not self._is_local_endpoint():
            raise ValueError("API key is required. Set HUIYAN_API_KEY environment variable or provide api_key parameter.")
        
        # Validate parameters
        if not 0.0 <= temperature <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {temperature}")
        
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")
        
        # Prepare request payload
        if self._is_local_endpoint() and self.endpoint.endswith("/api/chat"):
            options = {
                "temperature": temperature,
            }
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            if "top_p" in kwargs:
                options["top_p"] = kwargs["top_p"]
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": stream,
                "options": options,
            }
            headers = {
                "Content-Type": "application/json",
            }
            return self._make_request_with_retry(headers, payload)

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return self._make_request_with_retry(headers, payload)
    
    def simple_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Make a simple completion request with a single prompt.
        
        Args:
            prompt: User prompt
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        if self.endpoint.endswith("/api/generate"):
            options = {
                "temperature": temperature,
            }
            if max_tokens is not None:
                options["num_predict"] = max_tokens
            if stop:
                options["stop"] = stop
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }
            headers = {
                "Content-Type": "application/json",
            }
            response = self._make_request_with_retry(headers, payload)
            if isinstance(response, dict):
                return response.get("response", "")
            self.logger.error("Failed to extract Ollama generate response")
            self.logger.debug(f"Response: {json.dumps(response, indent=2)}")
            raise ValueError("Invalid Ollama generate response format")

        if self.endpoint.endswith("/chat/completions"):
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": prompt})

            response = self.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            if self._is_local_endpoint() and self.endpoint.endswith("/api/chat"):
                if isinstance(response, dict):
                    if "message" in response and isinstance(response.get("message"), dict):
                        return response["message"].get("content", "")
                    if "response" in response:
                        return response.get("response", "")
                self.logger.error("Failed to extract Ollama response content")
                self.logger.debug(f"Response: {json.dumps(response, indent=2)}")
                raise ValueError("Invalid Ollama response format")

            try:
                content = response['choices'][0]['message']['content']
                return content
            except (KeyError, IndexError) as e:
                self.logger.error(f"Failed to extract content from response: {e}")
                self.logger.debug(f"Response: {json.dumps(response, indent=2)}")
                raise ValueError(f"Invalid API response format: {e}")

        if self.endpoint.endswith("/v1/completions") or self.endpoint.endswith("/completions"):
            payload = {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
            }
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if stop:
                payload["stop"] = stop
            headers = {
                "Content-Type": "application/json",
            }
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            response = self._make_request_with_retry(headers, payload)
            return self._extract_completion_text(response)

        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        response = self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Extract content from response
        if self._is_local_endpoint() and self.endpoint.endswith("/api/chat"):
            if isinstance(response, dict):
                if "message" in response and isinstance(response.get("message"), dict):
                    return response["message"].get("content", "")
                if "response" in response:
                    return response.get("response", "")
            self.logger.error("Failed to extract Ollama response content")
            self.logger.debug(f"Response: {json.dumps(response, indent=2)}")
            raise ValueError("Invalid Ollama response format")

        try:
            content = response['choices'][0]['message']['content']
            return content
        except (KeyError, IndexError) as e:
            self.logger.error(f"Failed to extract content from response: {e}")
            self.logger.debug(f"Response: {json.dumps(response, indent=2)}")
            raise ValueError(f"Invalid API response format: {e}")

    def _extract_completion_text(self, response: Dict[str, Any]) -> str:
        try:
            return response["choices"][0]["text"]
        except (KeyError, IndexError) as exc:
            self.logger.error(f"Failed to extract completion text: {exc}")
            self.logger.debug(f"Response: {json.dumps(response, indent=2)}")
            raise ValueError(f"Invalid completion response format: {exc}")
    
    def test_connection(self) -> bool:
        """Test API connection with a simple request.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.logger.info("Testing API connection...")
            
            response = self.simple_completion(
                prompt=_TEST_CONNECTION_PROMPT,
                temperature=0.0,
                max_tokens=_TEST_CONNECTION_MAX_TOKENS
            )
            
            # Truncate response for logging (avoid logging potentially long responses)
            response_preview = response
            self.logger.info(f"API connection successful! Response: {response_preview}")
            return True
            
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the configured model.
        
        Returns:
            Dictionary with model configuration (API key is never included)
        """
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_backoff": self.retry_backoff,
            "api_key_set": bool(self.api_key),
            "api_key_length": len(self.api_key) if self.api_key else 0
        }


def get_llm_client(**kwargs) -> LLMClient:
    """Get an LLM client instance with configuration from config file.
    
    Args:
        **kwargs: Optional parameters to override config values
        
    Returns:
        LLMClient instance
    """
    return LLMClient(**kwargs)


# Example usage
if __name__ == "__main__":
    # Test the client
    client = get_llm_client()
    
    print("Model Info:")
    info = client.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\nTesting connection...")
    if client.test_connection():
        print("[+] Connection test passed!")
    else:
        print("[!] Connection test failed!")
