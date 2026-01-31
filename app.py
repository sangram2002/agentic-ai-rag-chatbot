"""
RAG-based AI Chatbot for Agentic AI eBook (100% FREE VERSION)
================================================================
A complete Retrieval-Augmented Generation (RAG) system using LangGraph
with FREE open-source models - NO API KEYS REQUIRED!

Uses:
- Sentence Transformers for embeddings (runs locally)
- Mistral-7B via Hugging Face Inference API (free tier)
- FAISS for vector storage
- LangGraph for workflow orchestration

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

# Hugging Face imports for FREE embeddings and LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub

# LangGraph imports for building the RAG workflow
from langgraph.graph import StateGraph, END

# For better LLM output
import re

# -------------------------------------------------------------------
# CONFIGURATION - 100% FREE, NO API KEYS NEEDED!
# -------------------------------------------------------------------

# PDF URL for the Agentic AI eBook
PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"

# Chunking parameters
CHUNK_SIZE = 700  # Number of characters per chunk (approx 175-200 tokens)
CHUNK_OVERLAP = 100  # Overlap between chunks to maintain context

# Retrieval parameters
TOP_K = 4  # Number of relevant chunks to retrieve

# FREE MODEL CONFIGURATION
# These models are completely free and run without any API keys!

# Embedding Model - Runs locally on your machine
# sentence-transformers/all-MiniLM-L6-v2: 
# - 384 dimensions, very fast
# - Trained on 1B+ sentence pairs
# - Perfect for semantic search
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Model - Uses Hugging Face's FREE Inference API
# Options (in order of quality vs speed):
# 1. "mistralai/Mistral-7B-Instruct-v0.2" - Best quality, slower (RECOMMENDED)
# 2. "google/flan-t5-large" - Good balance, faster
# 3. "facebook/opt-1.3b" - Fastest, lower quality
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Hugging Face API Token (optional, but recommended for better rate limits)
# Get free token at: https://huggingface.co/settings/tokens
# Even without this, the models will work on the free tier!
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")


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

@st.cache_resource(show_spinner="Loading and processing PDF with FREE models...")
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
    
    Generates a grounded answer using a FREE open-source LLM via Hugging Face.
    
    CRITICAL GROUNDING RULES:
    - LLM must ONLY use information from the provided context
    - If answer is not in context, LLM must explicitly state this
    - No external knowledge or assumptions allowed
    - This ensures factual accuracy and prevents hallucinations
    
    Why Mistral-7B?
    - One of the best open-source models (comparable to GPT-3.5)
    - Specifically trained for instruction following
    - FREE via Hugging Face Inference API
    - No API key required (but works better with free HF token)
    
    Args:
        state: Current graph state with question and context
        
    Returns:
        Updated state with generated answer
    """
    question = state["question"]
    context = state["context"]
    
    # Initialize the FREE LLM via Hugging Face
    # Even without HF_TOKEN, this works on the free tier!
    # With a free HF token, you get better rate limits
    llm = HuggingFaceHub(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=HF_TOKEN if HF_TOKEN else None,
        model_kwargs={
            "temperature": 0.1,  # Low temperature for focused, deterministic output
            "max_new_tokens": 512,  # Maximum length of generated answer
            "return_full_text": False,  # Only return the generated part
        }
    )
    
    # GROUNDING PROMPT TEMPLATE FOR MISTRAL
    # Mistral uses [INST] instruction format for best results
    prompt = f"""[INST] You are a helpful assistant answering questions about Agentic AI based STRICTLY on the provided context from an eBook.

STRICT RULES:
1. Use ONLY the information in the context below - DO NOT use any external knowledge
2. If the answer is not in the context, respond EXACTLY with: "I cannot find this information in the provided eBook content."
3. Be concise and direct - no unnecessary explanations
4. Quote specific parts when possible
5. Stay focused on the question

CONTEXT FROM EBOOK:
{context}

QUESTION: {question}

ANSWER (based only on context above): [/INST]"""

    # Generate the answer using the free LLM
    try:
        response = llm.invoke(prompt)
        # Clean up the response (remove any residual formatting)
        answer = response.strip()
        
        # Post-process to ensure quality
        # Remove any instruction artifacts
        answer = re.sub(r'\[INST\].*?\[/INST\]', '', answer, flags=re.DOTALL).strip()
        answer = re.sub(r'</?s>', '', answer).strip()  # Remove special tokens
        
        # If answer is too short or looks like an error, provide fallback
        if len(answer) < 10 or "error" in answer.lower():
            answer = "I cannot find this information in the provided eBook content."
            
    except Exception as e:
        # If LLM fails (rate limit, network issue), provide informative fallback
        answer = f"Unable to generate answer at this time. Please try again in a few seconds. (The free API may be busy)"
    
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
    4. GENERATE → LLM generates grounded answer (free Hugging Face API)
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
    - 100% FREE - no API keys required!
    - Clean, minimal chat interface
    - Displays answer, retrieved context, and confidence score
    - Shows sample queries for easy testing
    - Expands to show retrieved chunks for transparency
    """
    
    # Page configuration
    st.set_page_config(
        page_title="Agentic AI RAG Chatbot (FREE)",
        page_icon="🤖",
        layout="wide"
    )
    
    # -------------------------------------------------------------------
    # CRITICAL FIX: Initialize session state FIRST THING
    # This runs BEFORE anything else, ensuring vector_store always exists
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
    st.title("🤖 Agentic AI RAG Chatbot")
    st.markdown("""
    Ask questions about **Agentic AI** based on the official eBook.  
    **100% FREE** - Uses open-source models, no API keys required! 🎉
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
                
                st.success("✅ System initialized successfully! Models downloaded and ready.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                st.info("💡 If you see errors, try refreshing the page. If issues persist, check your internet connection.")
                st.stop()
    
    # -------------------------------------------------------------------
    # Only show UI after initialization is complete
    # -------------------------------------------------------------------
    
    # Sidebar with information and sample queries
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **100% FREE Tech Stack:**
        - 🧠 **LLM**: Mistral-7B (via HuggingFace)
        - 🔢 **Embeddings**: Sentence Transformers (local)
        - 📊 **Vector DB**: FAISS (local)
        - 🔄 **Workflow**: LangGraph
        - ✅ **Cost**: $0.00 forever!
        
        **No API keys needed!** Everything runs free.
        """)
        
        st.header("⚡ Performance")
        st.info("""
        **First run**: 1-2 min (downloads models)  
        **Subsequent**: 5-10 sec per query  
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
                # Force a rerun to update the text input
                st.rerun()
        
        st.header("🔧 System Status")
        if st.session_state.initialized:
            st.success("✅ Vector store loaded")
            st.success("✅ RAG graph initialized")
        else:
            st.warning("⏳ Initializing...")
            
        st.header("💡 Optional Setup")
        st.markdown("""
        For better rate limits (optional):
        1. Get free token at [HuggingFace](https://huggingface.co/settings/tokens)
        2. Add to Streamlit secrets as `HUGGINGFACEHUB_API_TOKEN`
        
        **Not required** - works without it!
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
    if st.button("🔍 Get Answer", type="primary"):
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
            with st.spinner("🤔 Processing your question with FREE models... (5-10 seconds)"):
                try:
                    # Process the query through RAG pipeline
                    # Pass vector_store explicitly to avoid any session_state issues
                    result = process_query(user_question, st.session_state.vector_store)
                    
                    # Display results
                    st.subheader("📖 Answer")
                    st.markdown(result["answer"])
                    
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
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Confidence Score",
                            value=f"{confidence_percentage:.1f}%",
                            help=f"{confidence_color} {confidence_label} confidence"
                        )
                    with col2:
                        st.metric(
                            label="Cost",
                            value="$0.00",
                            help="100% FREE - no API charges!"
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
                            "embedding_model": EMBEDDING_MODEL,
                            "llm_model": LLM_MODEL,
                            "num_chunks_retrieved": result["metadata"]["num_chunks"],
                            "page_numbers": result["metadata"]["page_numbers"],
                            "similarity_scores": [f"{s:.4f}" for s in result["metadata"]["similarity_scores"]],
                            "total_cost": "$0.00 (FREE!)"
                        })
                
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
                    st.info("💡 If you see rate limit errors from HuggingFace, please wait a few seconds and try again. The free tier has rate limits.")
        else:
            st.warning("⚠️ Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>100% FREE RAG Chatbot | Powered by Open-Source Models | No API Keys Required | Interview Task</small>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# APPLICATION ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()