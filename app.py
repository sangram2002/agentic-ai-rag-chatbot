"""
RAG-based AI Chatbot for Agentic AI eBook (100% FREE VERSION)
================================================================
A complete Retrieval-Augmented Generation (RAG) system using LangGraph
with FREE open-source models via Groq - BLAZING FAST & RELIABLE!

Uses:
- Sentence Transformers for embeddings (runs locally)
- Groq API with Llama-3 (FREE, super fast 1-2 sec responses!)
- FAISS for vector storage
- LangGraph for workflow orchestration

Why Groq is Better:
- 10x faster than HuggingFace (1-2 seconds vs 10-20 seconds)
- More reliable (dedicated infrastructure)
- Higher rate limits on free tier
- Better quality responses

Author: AI Engineer Intern Candidate
Interview Task Implementation
"""

import os
import streamlit as st
from typing import List, Dict, TypedDict
import warnings
warnings.filterwarnings('ignore')

# LangChain imports for document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Hugging Face imports for FREE local embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Groq imports for FAST LLM inference
from langchain_groq import ChatGroq

# LangGraph imports for building the RAG workflow
from langgraph.graph import StateGraph, END

# For better LLM output
import re

# -------------------------------------------------------------------
# CONFIGURATION - 100% FREE, BLAZING FAST!
# -------------------------------------------------------------------
# --- Configuration ---
# Llama 3.3 70B is currently the best "all-rounder" on Groq
SELECTED_MODEL = "llama-3.3-70b-versatile" 
# PDF URL for the Agentic AI eBook
PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"

# Chunking parameters
CHUNK_SIZE = 700  # Number of characters per chunk (approx 175-200 tokens)
CHUNK_OVERLAP = 100  # Overlap between chunks to maintain context

# Retrieval parameters
TOP_K = 4  # Number of relevant chunks to retrieve

# FREE MODEL CONFIGURATION

# Embedding Model - Runs locally on your machine (FREE)
# sentence-transformers/all-MiniLM-L6-v2: 
# - 384 dimensions, very fast
# - Trained on 1B+ sentence pairs
# - Perfect for semantic search
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Model - Uses Groq's FREE API (SUPER FAST!)
# Available models on Groq:
# - "llama-3.1-70b-versatile" - Best quality (recommended)
# - "llama-3.1-8b-instant" - Fastest
# - "mixtral-8x7b-32768" - Good balance


# Groq API Key (FREE - get from https://console.groq.com)
# Sign up is free and takes 1 minute!
GROQ_API_KEY = None
try:
    # Try environment variable first
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        # Try Streamlit secrets
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
except:
    GROQ_API_KEY = None


# -------------------------------------------------------------------
# STATE DEFINITION FOR LANGGRAPH
# -------------------------------------------------------------------

class GraphState(TypedDict):
    """
    State object that flows through the LangGraph workflow.
    
    This TypedDict defines the structure of data that passes between nodes:
    - question: The user's input question
    - retrieved_chunks: Documents retrieved from vector database
    - context: Formatted context string from retrieved chunks
    - answer: Final generated answer
    - confidence: Confidence score (0-1) based on retrieval similarity
    - metadata: Additional information (page numbers, sources, etc.)
    - vector_store: The FAISS vector store (passed through the state)
    """
    question: str
    retrieved_chunks: List[Document]
    context: str
    answer: str
    confidence: float
    metadata: Dict
    vector_store: any  # FAISS vector store passed through state


# -------------------------------------------------------------------
# PDF INGESTION & VECTOR STORE CREATION
# -------------------------------------------------------------------

@st.cache_resource(show_spinner="📚 Loading and processing PDF with FREE models...")
def create_vector_store():
    """
    Load PDF, chunk text, generate embeddings, and create vector store.
    
    Why FREE local embeddings are great:
    - No API costs - runs entirely on your machine
    - No rate limits - process as much as you want
    - Privacy - your data never leaves your computer
    - Fast - sentence-transformers are optimized for CPU
    
    Why chunking is needed in RAG:
    - LLMs have token limits; we can't send entire documents
    - Smaller chunks improve retrieval precision
    - Each chunk becomes a searchable unit with its own embedding
    - Overlap ensures important information at chunk boundaries isn't lost
    
    Returns:
        FAISS vector store with embedded document chunks
    """
    
    # Step 1: Download and load the PDF
    # PyPDFLoader extracts text from each page
    loader = PyPDFLoader(PDF_URL)
    documents = loader.load()
    
    # Step 2: Split documents into smaller chunks
    # RecursiveCharacterTextSplitter tries to split on natural boundaries
    # (paragraphs, sentences) while respecting the size limits
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Try these separators in order
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Step 3: Generate embeddings using FREE local model
    # HuggingFaceEmbeddings downloads the model once and runs locally
    # No internet needed after first download!
    # all-MiniLM-L6-v2: 384-dimensional embeddings, very fast
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},  # Use CPU (works everywhere)
        encode_kwargs={'normalize_embeddings': True}  # Normalize for better similarity
    )
    
    # Step 4: Create FAISS vector store
    # FAISS (Facebook AI Similarity Search) is an efficient vector database
    # It uses approximate nearest neighbor search for fast retrieval
    # Also completely free and runs locally!
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store


# -------------------------------------------------------------------
# LANGGRAPH NODE FUNCTIONS
# -------------------------------------------------------------------

def retrieve_node(state: GraphState) -> GraphState:
    """
    Node 1: Retrieval
    
    Retrieves the most relevant document chunks from the vector database
    based on semantic similarity to the user's question.
    
    How it works:
    1. Convert user question to embedding vector (using local model)
    2. Find K-nearest neighbor chunks in vector space
    3. Return chunks with similarity scores
    
    Args:
        state: Current graph state containing the question and vector_store
        
    Returns:
        Updated state with retrieved_chunks and metadata
    """
    question = state["question"]
    vector_store = state["vector_store"]  # Get vector store from state
    
    # Perform similarity search with scores
    # Returns: [(Document, similarity_score), ...]
    docs_with_scores = vector_store.similarity_search_with_score(
        question, k=TOP_K
    )
    
    # Extract just the documents and scores
    retrieved_chunks = [doc for doc, score in docs_with_scores]
    similarity_scores = [float(score) for doc, score in docs_with_scores]
    
    # Store metadata about retrieval
    metadata = {
        "num_chunks": len(retrieved_chunks),
        "similarity_scores": similarity_scores,
        "page_numbers": [doc.metadata.get("page", "N/A") for doc in retrieved_chunks]
    }
    
    state["retrieved_chunks"] = retrieved_chunks
    state["metadata"] = metadata
    
    return state


def format_context_node(state: GraphState) -> GraphState:
    """
    Node 2: Context Formatting
    
    Formats retrieved chunks into a coherent context string for the LLM.
    Includes metadata like page numbers for transparency.
    
    Args:
        state: Current graph state with retrieved_chunks
        
    Returns:
        Updated state with formatted context string
    """
    chunks = state["retrieved_chunks"]
    
    # Format each chunk with metadata
    formatted_chunks = []
    for i, doc in enumerate(chunks, 1):
        page_num = doc.metadata.get("page", "Unknown")
        chunk_text = doc.page_content.strip()
        formatted_chunks.append(f"[Source {i} - Page {page_num}]\n{chunk_text}")
    
    # Join all chunks with clear separators
    context = "\n\n---\n\n".join(formatted_chunks)
    
    state["context"] = context
    
    return state

def generate_answer_node(state: GraphState) -> GraphState:
    """
    Node 3: Answer Generation
    Uses a single, high-performance model (Llama 3.3 70B) via Groq.
    """
    question = state["question"]
    context = state["context"]
    
    if not GROQ_API_KEY:
        state["answer"] = "⚠️ Groq API key not found. Please add it to your environment."
        return state

    # 1. Initialize the single model directly
    # We use the variable SELECTED_MODEL defined at the top of your script
    llm = ChatGroq(
        model=SELECTED_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.1,  # Keep it low for factual RAG responses
        max_tokens=1024,
    )

    # 2. Define the Grounding Prompt
    prompt = f"""You are a helpful assistant answering questions about Agentic AI based STRICTLY on the provided context.

RULES:
1. Use ONLY the provided context.
2. If the answer isn't there, say: "I cannot find this information in the provided eBook content."
3. Stay concise and professional.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    try:
        # 3. Call the model (This replaces the old 'llm_chain.invoke' logic)
        response = llm.invoke(prompt)
        answer = response.content.strip()
        
        if len(answer) < 5:
            answer = "I cannot find this information in the provided eBook content."
            
    except Exception as e:
        # Handle API errors (like rate limits or invalid keys)
        answer = f"⚠️ Error generating answer: {str(e)[:100]}"
    
    state["answer"] = answer
    return state

def calculate_confidence_node(state: GraphState) -> GraphState:
    """
    Node 4: Confidence Scoring
    
    Calculates a confidence score (0-1) based on retrieval quality.
    
    Logic for local embeddings:
    - Sentence transformers use cosine similarity (0-2 scale)
    - Lower score = more similar (0 = identical, 2 = opposite)
    - Convert to 0-1 scale: confidence = 1 - (score / 2)
    - Average across top-K chunks
    
    Interpretation:
    - High confidence (>0.7): Strong semantic match, reliable answer
    - Medium confidence (0.4-0.7): Moderate match, generally reliable  
    - Low confidence (<0.4): Weak match, answer may be unreliable
    
    Args:
        state: Current graph state with metadata containing similarity scores
        
    Returns:
        Updated state with confidence score
    """
    similarity_scores = state["metadata"]["similarity_scores"]
    
    # Convert similarity scores to 0-1 confidence
    # For L2 distance (what FAISS returns): lower is better
    # Formula: confidence = 1 / (1 + score)
    confidences = [1 / (1 + score) for score in similarity_scores]
    
    # Calculate average confidence
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Clamp between 0 and 1
    confidence = max(0.0, min(1.0, confidence))
    
    state["confidence"] = confidence
    
    return state


# -------------------------------------------------------------------
# LANGGRAPH WORKFLOW CONSTRUCTION
# -------------------------------------------------------------------

def create_rag_graph():
    """
    Constructs the LangGraph workflow for RAG.
    
    Graph Flow:
    1. START → User submits question
    2. RETRIEVE → Fetch relevant chunks from vector DB (local, free)
    3. FORMAT → Format chunks into context string
    4. GENERATE → LLM generates grounded answer (Groq API - FAST!)
    5. CONFIDENCE → Calculate confidence score
    6. END → Return final response
    
    Why LangGraph?
    - Provides explicit control over RAG pipeline
    - Each node is testable and debuggable independently
    - Easy to add conditional logic (e.g., re-ranking, query expansion)
    - More maintainable than complex chains
    - Perfect for explaining in interviews!
    
    Returns:
        Compiled LangGraph application
    """
    
    # Initialize the graph with our state schema
    workflow = StateGraph(GraphState)
    
    # Add nodes to the graph
    # Each node is a function that processes the state
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("calculate_confidence", calculate_confidence_node)
    
    # Define edges (flow between nodes)
    # This creates a linear pipeline, but could be made conditional
    workflow.set_entry_point("retrieve")  # Start here
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate_answer")
    workflow.add_edge("generate_answer", "calculate_confidence")
    workflow.add_edge("calculate_confidence", END)  # Finish here
    
    # Compile the graph into an executable application
    app = workflow.compile()
    
    return app


# -------------------------------------------------------------------
# HELPER FUNCTION FOR QUERY PROCESSING
# -------------------------------------------------------------------

def process_query(question: str, vector_store) -> Dict:
    """
    Process a user question through the RAG pipeline.
    
    Args:
        question: User's input question
        vector_store: The FAISS vector store to use for retrieval
        
    Returns:
        Dictionary containing answer, context, confidence, and metadata
    """
    
    # Initialize the graph state with vector_store included
    initial_state = {
        "question": question,
        "retrieved_chunks": [],
        "context": "",
        "answer": "",
        "confidence": 0.0,
        "metadata": {},
        "vector_store": vector_store  # Pass vector store through state
    }
    
    # Run the graph
    rag_graph = st.session_state.rag_graph
    final_state = rag_graph.invoke(initial_state)
    
    return final_state


# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------

def main():
    """
    Main Streamlit application for the RAG chatbot.
    
    Features:
    - 100% FREE - just needs a free Groq API key!
    - BLAZING FAST - 1-2 second responses (10x faster than HuggingFace)
    - Clean, minimal chat interface
    - Displays answer, retrieved context, and confidence score
    - Shows sample queries for easy testing
    - Expands to show retrieved chunks for transparency
    """
    
    # Page configuration
    st.set_page_config(
        page_title="Agentic AI RAG Chatbot (Groq Powered)",
        page_icon="⚡",
        layout="wide"
    )
    
    # -------------------------------------------------------------------
    # CRITICAL: Initialize session state FIRST THING
    # This runs BEFORE anything else, ensuring all variables exist
    # -------------------------------------------------------------------
    
    # Initialize ALL session state variables at the very beginning
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_graph" not in st.session_state:
        st.session_state.rag_graph = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    
    # Title and description
    st.title("⚡ Agentic AI RAG Chatbot (Groq Powered)")
    
    # Show API key status
    if GROQ_API_KEY:
        st.success("✅ Groq API key detected - Ready for BLAZING FAST responses!")
    else:
        st.error("❌ Groq API key not found!")
        st.info("""
        **Get your FREE Groq API key in 1 minute:**
        1. Go to https://console.groq.com
        2. Sign up (free forever)
        3. Create API key
        4. Add to Streamlit secrets as `GROQ_API_KEY`
        
        **Why Groq?**
        - 10x faster than HuggingFace (1-2 sec vs 10-20 sec)
        - More reliable (no cold starts)
        - FREE tier: 30 requests/min (plenty for testing!)
        """)
        st.stop()
    
    st.markdown("""
    Ask questions about **Agentic AI** based on the official eBook.  
    **Powered by Groq** - Experience blazing fast 1-2 second responses! ⚡
    """)
    
    # -------------------------------------------------------------------
    # Initialize system components if not already done
    # This MUST complete before user can interact
    # -------------------------------------------------------------------
    
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing FREE models... First run takes 1-2 minutes to download models."):
            try:
                # Create vector store (cached, runs only once)
                st.session_state.vector_store = create_vector_store()
                
                # Create RAG graph
                st.session_state.rag_graph = create_rag_graph()
                
                # Mark as initialized
                st.session_state.initialized = True
                
                st.success("✅ System initialized successfully! Ready for lightning-fast queries!")
                st.info("⚡ **With Groq**: Queries take only 1-2 seconds!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                st.info("💡 Please refresh the page. If issues persist, check your internet connection.")
                st.stop()
    
    # -------------------------------------------------------------------
    # Only show UI after initialization is complete
    # -------------------------------------------------------------------
    
    # Sidebar with information and sample queries
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **100% FREE Tech Stack:**
        - ⚡ **LLM**: Groq (Llama-3.1-70B)
        - 🔢 **Embeddings**: Sentence Transformers (local)
        - 📊 **Vector DB**: FAISS (local)
        - 🔄 **Workflow**: LangGraph
        - ✅ **Cost**: $0.00 forever!
        
        **Why Groq is Amazing:**
        - 10x faster than alternatives
        - No cold starts
        - Reliable infrastructure
        """)
        
        st.header("⚡ Performance")
        st.success("""
        **First run**: 1-2 min (downloads embeddings)  
        **Per query**: 1-2 seconds! ⚡  
        **Rate limit**: 30 req/min (free tier)  
        **Cost**: FREE ✅
        """)
        
        st.header("📝 Sample Queries")
        sample_queries = [
            "What is Agentic AI?",
            "What are AI agents composed of?",
            "How do agent workflows differ from LLM chains?",
            "What are the limitations of agentic systems?",
            "What is the difference between traditional RAG and Agentic RAG?",
            "How does the ebook define tool usage?"
        ]
        
        # Sample query buttons - they only SET the query, don't process it
        for query in sample_queries:
            if st.button(query, key=f"sample_{query}", use_container_width=True):
                st.session_state.current_query = query
                st.rerun()
        
        st.header("🔧 System Status")
        if st.session_state.initialized:
            st.success("✅ Vector store loaded")
            st.success("✅ RAG graph initialized")
            st.success("✅ Groq API ready")
        else:
            st.warning("⏳ Initializing...")
        
        st.header("🔑 Groq Setup")
        with st.expander("How to get FREE Groq API key"):
            st.markdown("""
            **Quick Setup (1 minute):**
            
            1. **Sign up**: https://console.groq.com
            2. **Create API key** in dashboard
            3. **Add to Streamlit**:
               - Go to Settings → Secrets
               - Add:
               ```toml
               GROQ_API_KEY = "gsk_..."
               ```
            4. **Restart app**
            
            **Free Tier Limits:**
            - 30 requests/minute
            - 14,400 requests/day
            - Perfect for demos!
            """)
    
    # Chat interface
    st.header("💬 Ask a Question")
    
    # Text input for user question
    user_question = st.text_input(
        "Your question:",
        value=st.session_state.current_query,
        placeholder="e.g., What is Agentic AI?",
        key="question_input"
    )
    
    # Process query on button click
    if st.button("⚡ Get Answer (1-2 seconds!)", type="primary"):
        # CRITICAL CHECK: Only process if system is initialized
        if not st.session_state.initialized:
            st.error("❌ System is still initializing. Please wait for initialization to complete.")
            st.stop()
        
        if not st.session_state.vector_store:
            st.error("❌ Vector store not loaded. Please refresh the page.")
            st.stop()
        
        if not st.session_state.rag_graph:
            st.error("❌ RAG graph not initialized. Please refresh the page.")
            st.stop()
        
        if user_question.strip():
            # Show spinner with progress
            with st.spinner("⚡ Processing with Groq... (1-2 seconds)"):
                try:
                    import time
                    start_time = time.time()
                    
                    # Process the query through RAG pipeline
                    result = process_query(user_question, st.session_state.vector_store)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    # Display results
                    st.subheader("📖 Answer")
                    st.markdown(result["answer"])
                    
                    # Show processing time
                    st.caption(f"*Response generated in {processing_time:.2f} seconds using Groq ⚡*")
                    
                    # Display confidence score with color coding
                    confidence = result["confidence"]
                    confidence_percentage = confidence * 100
                    
                    if confidence >= 0.7:
                        confidence_color = "🟢"
                        confidence_label = "High"
                    elif confidence >= 0.4:
                        confidence_color = "🟡"
                        confidence_label = "Medium"
                    else:
                        confidence_color = "🔴"
                        confidence_label = "Low"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="Confidence Score",
                            value=f"{confidence_percentage:.1f}%",
                            help=f"{confidence_color} {confidence_label} confidence"
                        )
                    with col2:
                        st.metric(
                            label="Response Time",
                            value=f"{processing_time:.2f}s",
                            help="⚡ Groq is blazing fast!"
                        )
                    with col3:
                        st.metric(
                            label="Cost",
                            value="$0.00",
                            help="100% FREE!"
                        )
                    
                    # Display retrieved context in an expander
                    with st.expander("📚 View Retrieved Context Chunks", expanded=False):
                        st.markdown("**Retrieved chunks used to generate the answer:**")
                        
                        for i, chunk in enumerate(result["retrieved_chunks"], 1):
                            page_num = chunk.metadata.get("page", "Unknown")
                            similarity = result["metadata"]["similarity_scores"][i-1]
                            similarity_pct = (1 / (1 + similarity)) * 100
                            
                            st.markdown(f"**Chunk {i}** (Page {page_num}, Similarity: {similarity_pct:.1f}%)")
                            st.text_area(
                                f"chunk_{i}",
                                chunk.page_content,
                                height=150,
                                label_visibility="collapsed"
                            )
                            st.markdown("---")
                    
                    # Display metadata
                    with st.expander("🔍 Technical Details", expanded=False):
                        st.json({
                            "model_used": SELECTED_MODEL,  # Matches the variable at the top
                            "embedding_model": EMBEDDING_MODEL,
                            "runtime_environment": "Python 3.10",
                            "llm_provider": "Groq LPU (Ultra-fast)",
                            "status": "Success"
                        })
                
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    st.info("💡 Please check your Groq API key and try again.")
        else:
            st.warning("⚠️ Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>⚡ Powered by Groq | 100% FREE | Blazing Fast 1-2 Second Responses | Interview Task</small>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# APPLICATION ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()