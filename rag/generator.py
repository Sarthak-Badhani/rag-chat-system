"""
Answer Generation Module.

This module handles LLM-based answer generation using the
constructed prompts and retrieved context.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod

from .prompt_builder import BuiltPrompt

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class GeneratedAnswer:
    """
    A generated answer with metadata.
    
    Attributes:
        answer: The generated answer text
        sources: List of source citations
        model: Name of the model used
        prompt_tokens: Estimated prompt token count
        completion_tokens: Estimated completion token count
        metadata: Additional metadata
    """
    
    answer: str
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    metadata: dict = field(default_factory=dict)
    
    def get_formatted_sources(self) -> str:
        """Get sources formatted as string."""
        if not self.sources:
            return "No sources available"
        
        source_lines = []
        for source in self.sources:
            file_name = source.get("file_name", "Unknown")
            pages = source.get("pages", [])
            if pages:
                page_str = ", ".join(str(p) for p in pages)
                source_lines.append(f"- {file_name} (Pages: {page_str})")
            else:
                source_lines.append(f"- {file_name}")
        
        return "\n".join(source_lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "metadata": self.metadata
        }


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate(
        self, 
        prompt: BuiltPrompt, 
        temperature: float = 0,
        max_tokens: int = 512
    ) -> GeneratedAnswer:
        """Generate an answer from the prompt."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class OpenAILLM(BaseLLM):
    """
    OpenAI-based LLM for answer generation.
    
    Uses the OpenAI Chat API for generating answers.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0,
        max_tokens: int = 512
    ):
        """
        Initialize OpenAI LLM.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        
        try:
            from langchain_openai import ChatOpenAI
            
            self._llm = ChatOpenAI(
                openai_api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            logger.info(f"Initialized OpenAI LLM: {model}")
            
        except ImportError:
            raise ImportError(
                "Please install langchain-openai: pip install langchain-openai"
            )
    
    def generate(
        self,
        prompt: BuiltPrompt,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GeneratedAnswer:
        """
        Generate an answer using OpenAI.
        
        Args:
            prompt: The built prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            GeneratedAnswer with the response
        """
        from langchain_core.messages import HumanMessage, SystemMessage
        
        # Create messages
        messages = [
            SystemMessage(content=prompt.system_message),
            HumanMessage(content=prompt.user_message)
        ]
        
        # Update LLM settings if needed
        if temperature is not None:
            self._llm.temperature = temperature
        if max_tokens is not None:
            self._llm.max_tokens = max_tokens
        
        try:
            # Generate response
            response = self._llm.invoke(messages)
            
            # Extract answer
            answer_text = response.content
            
            # Get token usage if available
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'response_metadata'):
                usage = response.response_metadata.get('token_usage', {})
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
            
            return GeneratedAnswer(
                answer=answer_text,
                sources=prompt.sources,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    @property
    def model_name(self) -> str:
        return self.model


class HuggingFaceLLM(BaseLLM):
    """
    HuggingFace-based LLM for local inference.
    
    Runs models locally without API calls.
    """
    
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        temperature: float = 0,
        max_tokens: int = 512
    ):
        """
        Initialize HuggingFace LLM.
        
        Args:
            model_name: HuggingFace model name
            temperature: Generation temperature
            max_tokens: Maximum tokens in response
        """
        self.model = model_name
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        
        try:
            from langchain_huggingface import HuggingFacePipeline
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Create pipeline
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=max_tokens
            )
            
            self._llm = HuggingFacePipeline(pipeline=pipe)
            
            logger.info(f"Initialized HuggingFace LLM: {model_name}")
            
        except ImportError as e:
            raise ImportError(
                "Please install required packages: "
                "pip install langchain-huggingface transformers torch"
            )
    
    def generate(
        self,
        prompt: BuiltPrompt,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GeneratedAnswer:
        """
        Generate an answer using HuggingFace model.
        
        Args:
            prompt: The built prompt
            temperature: Override temperature (limited support)
            max_tokens: Override max tokens (limited support)
            
        Returns:
            GeneratedAnswer with the response
        """
        try:
            # Combine prompt into single string
            full_prompt = prompt.to_single_prompt()
            
            # Generate response
            response = self._llm.invoke(full_prompt)
            
            # Estimate tokens
            prompt_tokens = len(full_prompt.split()) * 4 // 3
            completion_tokens = len(response.split()) * 4 // 3
            
            return GeneratedAnswer(
                answer=response,
                sources=prompt.sources,
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise
    
    @property
    def model_name(self) -> str:
        return self.model


class AnswerGenerator:
    """
    High-level answer generation manager.
    
    This class provides a unified interface for generating answers
    using different LLM backends.
    
    Features:
        - Support for multiple LLM providers
        - Consistent answer format
        - Error handling and logging
        - Token usage tracking
    
    Attributes:
        provider: The LLM provider ("openai" or "huggingface")
        model_name: Name of the model being used
    
    Example:
        >>> generator = AnswerGenerator(provider="openai", api_key="sk-...")
        >>> answer = generator.generate(prompt)
        >>> print(answer.answer)
        >>> print(answer.get_formatted_sources())
    """
    
    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_tokens: int = 512
    ):
        """
        Initialize the answer generator.
        
        Args:
            provider: "openai" or "huggingface"
            api_key: API key (required for OpenAI)
            model_name: Model name (uses default if not specified)
            temperature: Generation temperature
            max_tokens: Maximum response tokens
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set default models
        if model_name is None:
            if self.provider == "openai":
                model_name = "gpt-3.5-turbo"
            else:
                model_name = "google/flan-t5-base"
        
        self._model_name = model_name
        
        # Initialize LLM
        if self.provider == "openai":
            if not api_key:
                raise ValueError("API key required for OpenAI")
            self._llm = OpenAILLM(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        else:
            self._llm = HuggingFaceLLM(
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        logger.info(f"AnswerGenerator initialized: {provider}/{model_name}")
    
    def generate(
        self,
        prompt: BuiltPrompt,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> GeneratedAnswer:
        """
        Generate an answer from a prompt.
        
        Args:
            prompt: Built prompt with context and question
            temperature: Override default temperature
            max_tokens: Override default max tokens
            
        Returns:
            GeneratedAnswer with response and metadata
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        logger.info(f"Generating answer (temp={temp}, max_tokens={tokens})")
        
        answer = self._llm.generate(
            prompt=prompt,
            temperature=temp,
            max_tokens=tokens
        )
        
        logger.info(
            f"Generated answer: {len(answer.answer)} chars, "
            f"{answer.completion_tokens} tokens"
        )
        
        return answer
    
    def generate_from_context(
        self,
        context: str,
        question: str,
        system_prompt: Optional[str] = None
    ) -> GeneratedAnswer:
        """
        Generate answer from raw context and question.
        
        Convenience method that builds the prompt internally.
        
        Args:
            context: Context text
            question: User question
            system_prompt: Optional custom system prompt
            
        Returns:
            GeneratedAnswer
        """
        from .prompt_builder import PromptBuilder, PromptConfig
        
        config = PromptConfig()
        if system_prompt:
            config.system_prompt = system_prompt
        
        builder = PromptBuilder(config)
        prompt = builder.build_simple(context, question)
        
        return self.generate(prompt)
    
    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model_name
    
    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "provider": self.provider,
            "model_name": self._model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
