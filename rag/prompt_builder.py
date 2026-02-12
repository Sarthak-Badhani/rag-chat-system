"""
Prompt Building Module.

This module handles the construction of prompts for the LLM,
incorporating retrieved context and following best practices
for RAG systems.
"""

import logging
from dataclasses import dataclass
from typing import Optional
from string import Template

from .retriever import RetrievalResult, RetrievedChunk

# Configure logging
logger = logging.getLogger(__name__)


# Default system prompt template
DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context. 

IMPORTANT RULES:
1. Answer the question using ONLY the information from the provided context.
2. If the answer cannot be found in the context, respond with: "I cannot find the answer to this question in the provided documents."
3. Do NOT use any external knowledge or make assumptions beyond what's in the context.
4. When possible, cite the source by mentioning the page number.
5. Be concise but thorough in your answers.
6. If the context contains conflicting information, acknowledge this and present both perspectives.
"""

# Default context template
DEFAULT_CONTEXT_TEMPLATE = """### Context from Documents:

$context

### End of Context

"""

# Default question template  
DEFAULT_QUESTION_TEMPLATE = """### Question:
$question

### Answer:
Please answer based on the context above. If the answer is not in the context, say so."""


@dataclass
class PromptConfig:
    """Configuration for prompt building."""
    
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    context_template: str = DEFAULT_CONTEXT_TEMPLATE
    question_template: str = DEFAULT_QUESTION_TEMPLATE
    include_citations: bool = True
    max_context_length: Optional[int] = None  # Characters
    chunk_separator: str = "\n\n---\n\n"


@dataclass
class BuiltPrompt:
    """A constructed prompt ready for the LLM."""
    
    system_message: str
    user_message: str
    context_used: str
    question: str
    sources: list[dict]
    truncated: bool = False
    
    def to_messages(self) -> list[dict]:
        """Convert to OpenAI-style messages format."""
        return [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.user_message}
        ]
    
    def to_single_prompt(self) -> str:
        """Convert to a single prompt string."""
        return f"{self.system_message}\n\n{self.user_message}"


class PromptBuilder:
    """
    Builder for RAG prompts with context integration.
    
    This class constructs prompts that:
    - Include retrieved context
    - Instruct the model to use only provided information
    - Include source citations
    - Handle context length limits
    
    The prompts are designed to minimize hallucination by
    explicitly instructing the model to answer only from context.
    
    Attributes:
        config: Prompt configuration settings
    
    Example:
        >>> builder = PromptBuilder()
        >>> prompt = builder.build(retrieval_result, "What is RAG?")
        >>> print(prompt.user_message)
    """
    
    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize the prompt builder.
        
        Args:
            config: Optional prompt configuration
        """
        self.config = config or PromptConfig()
        logger.info("PromptBuilder initialized")
    
    def build(
        self,
        retrieval_result: RetrievalResult,
        question: str
    ) -> BuiltPrompt:
        """
        Build a complete prompt from retrieval results.
        
        Args:
            retrieval_result: Retrieved chunks and metadata
            question: The user's question
            
        Returns:
            BuiltPrompt object ready for LLM
        """
        # Format context from chunks
        context_text, truncated = self._format_context(retrieval_result.chunks)
        
        # Build context section
        context_section = Template(self.config.context_template).substitute(
            context=context_text
        )
        
        # Build question section
        question_section = Template(self.config.question_template).substitute(
            question=question
        )
        
        # Combine into user message
        user_message = context_section + question_section
        
        # Get sources for citation
        sources = retrieval_result.get_unique_sources()
        
        prompt = BuiltPrompt(
            system_message=self.config.system_prompt,
            user_message=user_message,
            context_used=context_text,
            question=question,
            sources=sources,
            truncated=truncated
        )
        
        logger.info(
            f"Built prompt with {len(retrieval_result.chunks)} chunks, "
            f"truncated={truncated}"
        )
        
        return prompt
    
    def _format_context(
        self, 
        chunks: list[RetrievedChunk]
    ) -> tuple[str, bool]:
        """
        Format chunks into context string.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Tuple of (formatted context, was_truncated)
        """
        if not chunks:
            return "No relevant context found.", False
        
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            # Format each chunk with source info
            if self.config.include_citations:
                chunk_text = (
                    f"[Source: {chunk.file_name}, Page {chunk.page_number}]\n"
                    f"{chunk.content}"
                )
            else:
                chunk_text = chunk.content
            
            formatted_chunks.append(chunk_text)
        
        # Join chunks
        context = self.config.chunk_separator.join(formatted_chunks)
        
        # Check if truncation needed
        truncated = False
        if self.config.max_context_length:
            if len(context) > self.config.max_context_length:
                context = context[:self.config.max_context_length]
                context += "\n\n[Context truncated due to length...]"
                truncated = True
        
        return context, truncated
    
    def build_simple(
        self,
        context: str,
        question: str
    ) -> BuiltPrompt:
        """
        Build a prompt from raw context string.
        
        Args:
            context: Pre-formatted context string
            question: The user's question
            
        Returns:
            BuiltPrompt object
        """
        # Build context section
        context_section = Template(self.config.context_template).substitute(
            context=context
        )
        
        # Build question section
        question_section = Template(self.config.question_template).substitute(
            question=question
        )
        
        user_message = context_section + question_section
        
        return BuiltPrompt(
            system_message=self.config.system_prompt,
            user_message=user_message,
            context_used=context,
            question=question,
            sources=[],
            truncated=False
        )
    
    def build_follow_up(
        self,
        retrieval_result: RetrievalResult,
        question: str,
        conversation_history: list[dict]
    ) -> BuiltPrompt:
        """
        Build a prompt for follow-up questions with conversation history.
        
        Args:
            retrieval_result: Retrieved chunks
            question: The follow-up question
            conversation_history: Previous Q&A pairs
            
        Returns:
            BuiltPrompt with conversation context
        """
        # Format the main prompt
        prompt = self.build(retrieval_result, question)
        
        # Add conversation history to user message
        if conversation_history:
            history_text = "\n\n### Previous Conversation:\n"
            for entry in conversation_history[-3:]:  # Last 3 exchanges
                history_text += f"Q: {entry.get('question', '')}\n"
                history_text += f"A: {entry.get('answer', '')}\n\n"
            
            prompt.user_message = history_text + prompt.user_message
        
        return prompt
    
    @staticmethod
    def create_custom_system_prompt(
        persona: str = "helpful assistant",
        domain: str = "general",
        response_style: str = "concise",
        language: str = "English"
    ) -> str:
        """
        Create a customized system prompt.
        
        Args:
            persona: The role the AI should take
            domain: The domain of expertise
            response_style: How responses should be formatted
            language: Response language
            
        Returns:
            Custom system prompt string
        """
        return f"""You are a {persona} specializing in {domain}.

CRITICAL INSTRUCTIONS:
1. Answer questions using ONLY the provided context/documents.
2. If the answer is not in the context, say: "I cannot find this information in the provided documents."
3. Do NOT make up information or use external knowledge.
4. Cite sources when possible (mention page numbers).
5. Respond in {language}.
6. Keep your responses {response_style}.
7. If asked about topics outside the provided documents, politely redirect to the available content.

Always prioritize accuracy over completeness. It's better to say you don't know than to guess."""
    
    def estimate_token_count(self, prompt: BuiltPrompt) -> int:
        """
        Estimate the token count of a prompt.
        
        This is a rough estimate using character count / 4.
        For accurate counting, use tiktoken.
        
        Args:
            prompt: The built prompt
            
        Returns:
            Estimated token count
        """
        full_text = prompt.system_message + prompt.user_message
        # Rough estimate: ~4 characters per token
        return len(full_text) // 4
