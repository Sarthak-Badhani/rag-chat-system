"""
Chat with Your Documents - Streamlit Application

A production-quality Retrieval-Augmented Generation (RAG) system
that allows users to upload PDF documents and ask questions about them.

Author: Your Name
Version: 1.0.0
"""

import streamlit as st
import logging
import os
from pathlib import Path
from typing import Optional

from config import get_config, reload_config
from rag import (
    PDFLoader,
    TextChunker,
    EmbeddingManager,
    VectorStoreManager,
    DocumentRetriever,
    PromptBuilder,
    AnswerGenerator
)
from rag.retriever import RetrievalResult
from rag.generator import GeneratedAnswer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Chat with Your Documents",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state():
    """Initialize session state variables."""
    # Always reload config to pick up .env changes
    st.session_state.config = reload_config()
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = None
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    # Reset generator to None so it's recreated with fresh config
    st.session_state.generator = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []
    
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []


# ============================================================================
# Core RAG Functions
# ============================================================================

@st.cache_resource
def get_embedding_manager(provider: str, model_name: str, api_key: Optional[str] = None):
    """Get or create embedding manager (cached)."""
    config = get_config()
    return EmbeddingManager(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        cache_dir=config.paths.cache_dir,
        cache_enabled=config.embedding.cache_enabled
    )


def process_documents(uploaded_files, chunk_size: int, chunk_overlap: int) -> bool:
    """
    Process uploaded PDF files through the indexing pipeline.
    
    Args:
        uploaded_files: List of uploaded file objects
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        True if processing succeeded
    """
    config = st.session_state.config
    
    try:
        with st.spinner("Loading PDFs..."):
            loader = PDFLoader()
            all_documents = []
            
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                temp_path = config.paths.data_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Load documents
                docs = loader.load(temp_path)
                all_documents.extend(docs)
                
                # Track processed files
                st.session_state.processed_files.append(uploaded_file.name)
            
            st.success(f"Loaded {len(all_documents)} pages from {len(uploaded_files)} file(s)")
        
        with st.spinner("Chunking text..."):
            chunker = TextChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = chunker.chunk_documents(all_documents)
            stats = chunker.get_chunk_statistics(chunks)
            st.success(f"Created {stats['total_chunks']} chunks (avg size: {stats['avg_chunk_size']} chars)")
        
        with st.spinner("Generating embeddings..."):
            # Get API key if using OpenAI
            api_key = None
            if config.embedding.provider == "openai":
                api_key = config.openai_api_key
            
            # Create or get embedding manager
            embedding_manager = get_embedding_manager(
                provider=config.embedding.provider,
                model_name=(
                    config.embedding.openai_model 
                    if config.embedding.provider == "openai" 
                    else config.embedding.huggingface_model
                ),
                api_key=api_key
            )
            st.session_state.embedding_manager = embedding_manager
            st.success(f"Using {embedding_manager.provider} embeddings ({embedding_manager.dimension}d)")
        
        with st.spinner("Building vector index..."):
            vector_store = VectorStoreManager(
                embedding_manager=embedding_manager,
                index_dir=config.paths.index_dir
            )
            
            # Convert chunks to LangChain documents
            langchain_docs = [chunk.to_langchain_document() for chunk in chunks]
            vector_store.create_index(langchain_docs)
            vector_store.save()
            
            st.session_state.vector_store = vector_store
            st.success(f"Vector index created with {vector_store.document_count} chunks")
        
        # Initialize retriever
        st.session_state.retriever = DocumentRetriever(
            vector_store=vector_store,
            top_k=config.retriever.top_k
        )
        
        # Initialize generator
        if config.llm.provider == "openai" and config.openai_api_key:
            st.session_state.generator = AnswerGenerator(
                provider="openai",
                api_key=config.openai_api_key,
                model_name=config.llm.openai_model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens
            )
        elif config.llm.provider == "huggingface":
            st.session_state.generator = AnswerGenerator(
                provider="huggingface",
                model_name=config.llm.huggingface_model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens
            )
        
        st.session_state.documents_processed = True
        embedding_manager.save_cache()
        
        return True
        
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        logger.exception("Error in document processing")
        return False


def load_existing_index() -> bool:
    """Load an existing vector index if available."""
    config = st.session_state.config
    
    try:
        # Get API key if using OpenAI
        api_key = None
        if config.embedding.provider == "openai":
            api_key = config.openai_api_key
        
        embedding_manager = get_embedding_manager(
            provider=config.embedding.provider,
            model_name=(
                config.embedding.openai_model 
                if config.embedding.provider == "openai" 
                else config.embedding.huggingface_model
            ),
            api_key=api_key
        )
        st.session_state.embedding_manager = embedding_manager
        
        vector_store = VectorStoreManager(
            embedding_manager=embedding_manager,
            index_dir=config.paths.index_dir
        )
        
        if vector_store.load():
            st.session_state.vector_store = vector_store
            st.session_state.retriever = DocumentRetriever(
                vector_store=vector_store,
                top_k=config.retriever.top_k
            )
            
            if config.llm.provider == "openai" and config.openai_api_key:
                st.session_state.generator = AnswerGenerator(
                    provider="openai",
                    api_key=config.openai_api_key,
                    model_name=config.llm.openai_model,
                    temperature=config.llm.temperature
                )
            elif config.llm.provider == "huggingface":
                st.session_state.generator = AnswerGenerator(
                    provider="huggingface",
                    model_name=config.llm.huggingface_model,
                    temperature=config.llm.temperature
                )
            
            st.session_state.documents_processed = True
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error loading index: {e}")
        return False


def ask_question(question: str, top_k: int) -> tuple[Optional[GeneratedAnswer], Optional[RetrievalResult]]:
    """
    Process a user question and generate an answer.
    
    Args:
        question: The user's question
        top_k: Number of chunks to retrieve
        
    Returns:
        Tuple of (GeneratedAnswer, RetrievalResult) or (None, None) on error
    """
    if not st.session_state.retriever:
        st.error("Please process documents first!")
        return None, None
    
    try:
        # Retrieve relevant chunks
        retrieval_result = st.session_state.retriever.retrieve(question, top_k=top_k)
        
        if not retrieval_result.has_results:
            return GeneratedAnswer(
                answer="I couldn't find any relevant information in the documents for your question.",
                sources=[]
            ), retrieval_result
        
        # Build prompt
        prompt_builder = PromptBuilder()
        prompt = prompt_builder.build(retrieval_result, question)
        
        # Get config and ensure generator matches current settings
        config = st.session_state.config
        
        # Create generator based on current config if not set or provider changed
        if st.session_state.generator is None:
            if config.llm.provider == "huggingface":
                st.session_state.generator = AnswerGenerator(
                    provider="huggingface",
                    model_name=config.llm.huggingface_model,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens
                )
            elif config.llm.provider == "openai" and config.openai_api_key:
                st.session_state.generator = AnswerGenerator(
                    provider="openai",
                    api_key=config.openai_api_key,
                    model_name=config.llm.openai_model,
                    temperature=config.llm.temperature,
                    max_tokens=config.llm.max_tokens
                )
        
        # Generate answer
        if st.session_state.generator:
            answer = st.session_state.generator.generate(prompt)
        else:
            # Fallback if no generator (show retrieved context only)
            answer = GeneratedAnswer(
                answer="[LLM not configured - showing retrieved context]\n\n" + 
                       retrieval_result.get_context_text(),
                sources=retrieval_result.get_unique_sources()
            )
        
        return answer, retrieval_result
        
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        logger.exception("Error in question answering")
        return None, None


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key configuration
        st.subheader("API Keys")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.config.openai_api_key or "",
            type="password",
            help="Required for OpenAI embeddings and LLM"
        )
        
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            st.session_state.config = reload_config()
        
        st.divider()
        
        # Chunking configuration
        st.subheader("📄 Chunking")
        chunk_size = st.slider(
            "Chunk Size",
            min_value=200,
            max_value=2000,
            value=st.session_state.config.chunking.chunk_size,
            step=100,
            help="Target size for each text chunk (characters)"
        )
        
        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=st.session_state.config.chunking.chunk_overlap,
            step=50,
            help="Overlap between consecutive chunks"
        )
        
        st.divider()
        
        # Retrieval configuration
        st.subheader("🔍 Retrieval")
        top_k = st.slider(
            "Top-K Results",
            min_value=1,
            max_value=10,
            value=st.session_state.config.retriever.top_k,
            help="Number of chunks to retrieve"
        )
        
        st.divider()
        
        # Model configuration
        st.subheader("🤖 Models")
        embedding_provider = st.selectbox(
            "Embedding Provider",
            options=["huggingface", "openai"],
            index=0 if st.session_state.config.embedding.provider == "huggingface" else 1
        )
        
        llm_provider = st.selectbox(
            "LLM Provider",
            options=["openai", "huggingface"],
            index=0 if st.session_state.config.llm.provider == "openai" else 1
        )
        
        st.divider()
        
        # Index management
        st.subheader("💾 Index Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Load Index", use_container_width=True):
                if load_existing_index():
                    st.success("Index loaded!")
                    st.rerun()
                else:
                    st.warning("No existing index found")
        
        with col2:
            if st.button("Clear Index", use_container_width=True):
                if st.session_state.vector_store:
                    st.session_state.vector_store.delete()
                st.session_state.documents_processed = False
                st.session_state.chat_history = []
                st.session_state.processed_files = []
                st.success("Index cleared!")
                st.rerun()
        
        return {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "embedding_provider": embedding_provider,
            "llm_provider": llm_provider
        }


def render_document_upload(chunk_size: int, chunk_overlap: int):
    """Render the document upload section."""
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to chat with"
    )
    
    if uploaded_files:
        st.write(f"**Selected files:** {', '.join(f.name for f in uploaded_files)}")
        
        if st.button("🚀 Process Documents", type="primary", use_container_width=True):
            success = process_documents(uploaded_files, chunk_size, chunk_overlap)
            if success:
                st.balloons()
                st.rerun()  # Refresh UI to update chat tab


def render_chat_interface(top_k: int):
    """Render the chat interface."""
    st.header("💬 Chat with Your Documents")
    
    if not st.session_state.documents_processed:
        st.info("👆 Please upload and process documents first, or load an existing index from the sidebar.")
        return
    
    # Show processed files
    if st.session_state.processed_files:
        with st.expander("📚 Processed Documents"):
            for f in st.session_state.processed_files:
                st.write(f"- {f}")
    
    # Question input
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="What is the main topic of this document?",
        key="question_input"
    )
    
    col1, col2 = st.columns([4, 1])
    with col1:
        ask_button = st.button("🔎 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    if ask_button and question:
        with st.spinner("Thinking..."):
            answer, retrieval = ask_question(question, top_k)
            
            if answer:
                # Add to chat history
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer.answer,
                    "sources": answer.sources,
                    "retrieval": retrieval
                })
    
    # Display chat history (most recent first)
    if st.session_state.chat_history:
        st.divider()
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                # Question
                st.markdown(f"**🙋 Question:** {chat['question']}")
                
                # Answer
                st.markdown(f"**🤖 Answer:**")
                st.markdown(chat['answer'])
                
                # Sources
                if chat['sources']:
                    with st.expander("📖 Sources"):
                        for source in chat['sources']:
                            pages = ", ".join(str(p) for p in source.get('pages', []))
                            st.write(f"- **{source['file_name']}** (Pages: {pages})")
                
                # Retrieved chunks
                if chat.get('retrieval'):
                    with st.expander("📝 Retrieved Chunks"):
                        for j, chunk in enumerate(chat['retrieval'].chunks):
                            st.markdown(f"**Chunk {j+1}** (Score: {chunk.similarity_score:.3f})")
                            st.markdown(f"*Source: {chunk.file_name}, Page {chunk.page_number}*")
                            st.text(chunk.content[:500] + ("..." if len(chunk.content) > 500 else ""))
                            st.divider()
                
                st.divider()


def render_evaluation_section(top_k: int):
    """Render the evaluation section."""
    st.header("🧪 Evaluation")
    
    if not st.session_state.documents_processed:
        st.info("Process documents first to use the evaluation feature.")
        return
    
    st.markdown("""
    Use this section to test retrieval quality with predefined questions.
    This helps understand how chunk size and top-k settings affect results.
    """)
    
    # Predefined test questions
    test_questions = st.text_area(
        "Test Questions (one per line)",
        value="What is the main topic?\nWhat are the key findings?\nWhat methodology was used?",
        height=100
    )
    
    if st.button("Run Evaluation", use_container_width=True):
        questions = [q.strip() for q in test_questions.split("\n") if q.strip()]
        
        for q in questions:
            st.subheader(f"Q: {q}")
            with st.spinner("Retrieving..."):
                answer, retrieval = ask_question(q, top_k)
                
                if retrieval:
                    st.markdown("**Retrieved Chunks:**")
                    for i, chunk in enumerate(retrieval.chunks):
                        with st.expander(f"Chunk {i+1} - Score: {chunk.similarity_score:.3f}"):
                            st.write(f"**Source:** {chunk.file_name}, Page {chunk.page_number}")
                            st.text(chunk.content)
                
                if answer:
                    st.markdown("**Generated Answer:**")
                    st.write(answer.answer)
            
            st.divider()


# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # App title
    st.title("📚 Chat with Your Documents")
    st.markdown("""
    *A production-quality RAG system for document question-answering*
    
    Upload PDF documents and ask questions. Answers are generated using only
    the content from your documents to minimize hallucination.
    """)
    
    # Render sidebar and get configuration
    sidebar_config = render_sidebar()
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📁 Upload", "💬 Chat", "🧪 Evaluate"])
    
    with tab1:
        render_document_upload(
            sidebar_config["chunk_size"],
            sidebar_config["chunk_overlap"]
        )
    
    with tab2:
        render_chat_interface(sidebar_config["top_k"])
    
    with tab3:
        render_evaluation_section(sidebar_config["top_k"])
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        Built with LangChain, FAISS, and Streamlit | 
        <a href='https://github.com/yourusername/rag-chat-system'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
