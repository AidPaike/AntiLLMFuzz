"""LLM客户端管理器 - 支持多后端和优雅降级"""

import logging
import requests
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """LLM客户端基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.available = False
    
    @abstractmethod
    def test_connection(self) -> bool:
        """测试连接是否可用"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass


class OllamaClient(BaseLLMClient):
    """Ollama本地客户端"""
    
    def __init__(self, model: str = "qwen2.5-coder", endpoint: str = "http://localhost:11434"):
        super().__init__("Ollama")
        self.model = model
        self.endpoint = endpoint
        self.available = self.test_connection()
    
    def test_connection(self) -> bool:
        """测试Ollama连接"""
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if any(self.model in name for name in model_names):
                    logger.info(f"✅ Ollama可用，模型: {self.model}")
                    return True
                else:
                    logger.warning(f"⚠️ Ollama可用但缺少模型: {self.model}")
                    logger.info(f"💡 可用模型: {model_names}")
                    return False
        except Exception as e:
            logger.debug(f"Ollama连接失败: {e}")
        return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """使用Ollama生成文本"""
        if not self.available:
            raise RuntimeError("Ollama不可用")
        
        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get('response', '')
            
        except Exception as e:
            logger.error(f"Ollama生成失败: {e}")
            raise


class MockLLMClient(BaseLLMClient):
    """模拟LLM客户端 - 用于测试和演示"""
    
    def __init__(self):
        super().__init__("Mock")
        self.available = True
    
    def test_connection(self) -> bool:
        return True
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成模拟响应"""
        # 根据提示词类型生成不同的模拟响应
        if "test case" in prompt.lower():
            return self._generate_mock_test_case()
        elif "security" in prompt.lower():
            return self._generate_mock_security_test()
        elif "edge case" in prompt.lower():
            return self._generate_mock_edge_case()
        else:
            return self._generate_generic_response()
    
    def _generate_mock_test_case(self) -> str:
        return """
// Mock test case
public class TestExample {
    public void testBasicFunctionality() {
        String input = "test input";
        String result = processInput(input);
        assertEquals("expected", result);
    }
}
"""
    
    def _generate_mock_security_test(self) -> str:
        return """
// Mock security test
public class SecurityTest {
    public void testInputValidation() {
        String maliciousInput = "<script>alert('xss')</script>";
        assertThrows(SecurityException.class, () -> {
            processUntrustedInput(maliciousInput);
        });
    }
}
"""
    
    def _generate_mock_edge_case(self) -> str:
        return """
// Mock edge case test
public class EdgeCaseTest {
    public void testNullInput() {
        assertThrows(IllegalArgumentException.class, () -> {
            processInput(null);
        });
    }
    
    public void testEmptyInput() {
        String result = processInput("");
        assertNotNull(result);
    }
}
"""
    
    def _generate_generic_response(self) -> str:
        return "这是一个模拟的LLM响应。在实际环境中，这里会是真实的AI生成内容。"


class OpenAIClient(BaseLLMClient):
    """OpenAI API客户端"""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: Optional[str] = None):
        super().__init__("OpenAI")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or "https://api.openai.com/v1"
        self.available = self.test_connection() if api_key else False
    
    def test_connection(self) -> bool:
        """测试OpenAI API连接"""
        if not self.api_key:
            return False
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("✅ OpenAI API可用")
                return True
            else:
                logger.warning(f"⚠️ OpenAI API响应异常: {response.status_code}")
                return False
                
        except Exception as e:
            logger.debug(f"OpenAI API连接失败: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """使用OpenAI API生成文本"""
        if not self.available:
            raise RuntimeError("OpenAI API不可用")
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7)
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"OpenAI API生成失败: {e}")
            raise


class LLMClientManager:
    """LLM客户端管理器 - 支持多后端和自动降级"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.clients: List[BaseLLMClient] = []
        self.current_client: Optional[BaseLLMClient] = None
        self._init_clients()
    
    def _init_clients(self):
        """初始化所有可用的LLM客户端"""
        logger.info("🔍 初始化LLM客户端...")
        
        # 1. 尝试Ollama (本地优先)
        try:
            ollama_config = self.config.get('ollama', {})
            model = ollama_config.get('model', 'qwen2.5-coder')
            endpoint = ollama_config.get('endpoint', 'http://localhost:11434')
            
            ollama_client = OllamaClient(model, endpoint)
            self.clients.append(ollama_client)
            
            if ollama_client.available:
                logger.info(f"✅ Ollama客户端已就绪: {model}")
            else:
                logger.info("💡 Ollama不可用，尝试其他选项...")
                
        except Exception as e:
            logger.debug(f"Ollama初始化失败: {e}")
        
        # 2. 尝试OpenAI API
        try:
            openai_config = self.config.get('openai', {})
            api_key = openai_config.get('api_key') or self.config.get('api_key')
            
            if api_key and api_key != "${OPENAI_API_KEY}":
                model = openai_config.get('model', 'gpt-3.5-turbo')
                base_url = openai_config.get('base_url')
                
                openai_client = OpenAIClient(api_key, model, base_url)
                self.clients.append(openai_client)
                
                if openai_client.available:
                    logger.info(f"✅ OpenAI客户端已就绪: {model}")
                    
        except Exception as e:
            logger.debug(f"OpenAI初始化失败: {e}")
        
        # 3. 总是添加Mock客户端作为兜底
        mock_client = MockLLMClient()
        self.clients.append(mock_client)
        logger.info("✅ Mock客户端已就绪 (兜底方案)")
        
        # 选择第一个可用的客户端
        self._select_best_client()
    
    def _select_best_client(self):
        """选择最佳可用客户端"""
        for client in self.clients:
            if client.available:
                self.current_client = client
                logger.info(f"🎯 选择LLM客户端: {client.name}")
                return
        
        # 如果没有可用客户端，使用Mock
        self.current_client = self.clients[-1]  # Mock客户端
        logger.warning("⚠️ 没有可用的LLM客户端，使用Mock模式")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本，自动处理失败和重试"""
        if not self.current_client:
            raise RuntimeError("没有可用的LLM客户端")
        
        # 尝试当前客户端
        try:
            return self.current_client.generate(prompt, **kwargs)
        except Exception as e:
            logger.warning(f"当前客户端 {self.current_client.name} 失败: {e}")
            
            # 尝试降级到下一个可用客户端
            return self._fallback_generate(prompt, **kwargs)
    
    def _fallback_generate(self, prompt: str, **kwargs) -> str:
        """降级生成 - 尝试其他可用客户端"""
        if not self.current_client:
            raise RuntimeError("没有当前客户端")
            
        current_index = self.clients.index(self.current_client)
        
        for i in range(current_index + 1, len(self.clients)):
            client = self.clients[i]
            if client.available:
                try:
                    logger.info(f"🔄 降级到客户端: {client.name}")
                    self.current_client = client
                    return client.generate(prompt, **kwargs)
                except Exception as e:
                    logger.warning(f"降级客户端 {client.name} 也失败: {e}")
                    continue
        
        # 最后尝试Mock客户端
        mock_client = self.clients[-1]
        logger.warning("🔄 降级到Mock客户端")
        self.current_client = mock_client
        return mock_client.generate(prompt, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """获取所有客户端状态"""
        return {
            'current_client': self.current_client.name if self.current_client else None,
            'available_clients': [
                {
                    'name': client.name,
                    'available': client.available
                }
                for client in self.clients
            ]
        }
    
    def is_mock_mode(self) -> bool:
        """检查是否在Mock模式"""
        return isinstance(self.current_client, MockLLMClient)


# 全局实例
_llm_manager = None

def get_llm_manager(config: Optional[Dict[str, Any]] = None) -> LLMClientManager:
    """获取全局LLM管理器实例"""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMClientManager(config)
    return _llm_manager


def reset_llm_manager():
    """重置LLM管理器（用于测试）"""
    global _llm_manager
    _llm_manager = None