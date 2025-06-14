"""LLM Client module for backward compatibility and testing."""

from typing import Any, Dict, Optional
from .llm import LLM, EnhancedLLMService, get_llm_service


class LLMClient:
    """Simple LLM client wrapper for backward compatibility."""
    
    def __init__(self, timeout: int = 30, max_requests_per_minute: int = 12):
        """Initialize the LLM client.
        
        Args:
            timeout: Request timeout in seconds
            max_requests_per_minute: Rate limit for requests
        """
        self.timeout = timeout
        self.max_requests_per_minute = max_requests_per_minute
        self._llm = LLM(timeout=timeout, max_requests_per_minute=max_requests_per_minute)
        self._enhanced_service = get_llm_service()
    
    def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using the LLM.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments
            
        Returns:
            Generated content as string
        """
        return self._llm.generate_content(prompt)
    
    async def generate_content_async(self, prompt: str, **kwargs) -> str:
        """Generate content asynchronously.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments
            
        Returns:
            Generated content as string
        """
        response = await self._enhanced_service.generate_content(prompt)
        return response.content
    
    @property
    def chat(self):
        """Provide OpenAI-style chat interface for compatibility."""
        return ChatInterface(self._enhanced_service)


class ChatInterface:
    """OpenAI-style chat interface for compatibility."""
    
    def __init__(self, llm_service: EnhancedLLMService):
        self.llm_service = llm_service
        self.completions = CompletionsInterface(llm_service)


class CompletionsInterface:
    """OpenAI-style completions interface."""
    
    def __init__(self, llm_service: EnhancedLLMService):
        self.llm_service = llm_service
    
    async def create(self, model: str, messages: list, **kwargs) -> Any:
        """Create a chat completion.
        
        Args:
            model: Model name
            messages: List of messages
            **kwargs: Additional arguments
            
        Returns:
            Mock response object with OpenAI-style structure
        """
        # Extract the user message
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Generate response
        response = await self.llm_service.generate_content(user_message)
        
        # Return OpenAI-style response structure
        return MockOpenAIResponse(response.content)


class MockOpenAIResponse:
    """Mock OpenAI response structure for compatibility."""
    
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock OpenAI choice structure."""
    
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock OpenAI message structure."""
    
    def __init__(self, content: str):
        self.content = content