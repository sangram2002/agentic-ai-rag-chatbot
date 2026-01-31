"""
RAG-based AI Chatbot for Agentic AI eBook
==========================================
A complete Retrieval-Augmented Generation (RAG) system using LangGraph
that answers questions strictly based on the Agentic AI eBook.

Author: AI Engineer Intern Candidate
Interview Task Implementation
"""

import os
import streamlit as st
from typing import List, Dict, TypedDict, Annotated
import operator
from datetime import datetime

# LangChain imports for document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# LangGraph imports for building the RAG workflow
from langgraph.graph import StateGraph, END

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------

# OpenAI API Key (you'll need to set this in Streamlit secrets or .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))

# PDF URL for the Agentic AI eBook
PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"

# Chunking parameters
CHUNK_SIZE = 700  # Number of characters per chunk (approx 175-200 tokens)
CHUNK_OVERLAP = 100  # Overlap between chunks to maintain context

# Retrieval parameters
TOP_K = 4  # Number of relevant chunks to retrieve

# Model configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
LLM_MODEL = "gpt-4o-mini"  # Fast and cost-effective LLM for generation


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
    """
    question: str
    retrieved_chunks: List[Document]
    context: str
    answer: str
    confidence: float
    metadata: Dict


# -------------------------------------------------------------------
# PDF INGESTION & VECTOR STORE CREATION
# -------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading and processing PDF...")
def create_vector_store():
    """
    Load PDF, chunk text, generate embeddings, and create vector store.
    
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
    
    # Step 3: Generate embeddings and create vector store
    # Embeddings convert text into numerical vectors (1536 dimensions for this model)
    # Similar concepts will have similar vector representations
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY
    )
    
    # FAISS (Facebook AI Similarity Search) is an efficient vector database
    # It uses approximate nearest neighbor search for fast retrieval
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
    1. Convert user question to embedding vector
    2. Find K-nearest neighbor chunks in vector space
    3. Return chunks with similarity scores
    
    Args:
        state: Current graph state containing the question
        
    Returns:
        Updated state with retrieved_chunks and metadata
    """
    question = state["question"]
    
    # Perform similarity search with scores
    # Returns: [(Document, similarity_score), ...]
    retriever = st.session_state.vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    
    # Get documents with similarity scores
    docs_with_scores = st.session_state.vector_store.similarity_search_with_score(
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
        formatted_chunks.append(f"[Chunk {i} - Page {page_num}]\n{chunk_text}")
    
    # Join all chunks with clear separators
    context = "\n\n---\n\n".join(formatted_chunks)
    
    state["context"] = context
    
    return state


def generate_answer_node(state: GraphState) -> GraphState:
    """
    Node 3: Answer Generation
    
    Generates a grounded answer using the LLM, strictly based on retrieved context.
    
    CRITICAL GROUNDING RULES:
    - LLM must ONLY use information from the provided context
    - If answer is not in context, LLM must explicitly state this
    - No external knowledge or assumptions allowed
    - This ensures factual accuracy and prevents hallucinations
    
    Args:
        state: Current graph state with question and context
        
    Returns:
        Updated state with generated answer
    """
    question = state["question"]
    context = state["context"]
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,  # Deterministic output for consistency
        openai_api_key=OPENAI_API_KEY
    )
    
    # GROUNDING PROMPT TEMPLATE
    # This is the most critical part - it enforces strict grounding
    prompt = f"""You are a helpful assistant answering questions about Agentic AI based STRICTLY on the provided context from an eBook.

STRICT RULES:
1. Use ONLY the information provided in the context below
2. Do NOT use any external knowledge or make assumptions
3. If the answer is not found in the context, respond with: "I cannot find this information in the provided eBook content."
4. Quote or reference specific parts of the context when possible
5. Be concise but comprehensive

CONTEXT FROM EBOOK:
{context}

QUESTION:
{question}

ANSWER (based only on the context above):"""

    # Generate the answer
    response = llm.invoke(prompt)
    answer = response.content
    
    state["answer"] = answer
    
    return state


def calculate_confidence_node(state: GraphState) -> GraphState:
    """
    Node 4: Confidence Scoring
    
    Calculates a confidence score (0-1) based on retrieval quality.
    
    Logic:
    - FAISS returns distance scores (lower = more similar)
    - We need to convert distance to similarity
    - For L2 distance: similarity = 1 / (1 + distance)
    - Average across top-K chunks
    - Normalize to 0-1 range
    
    High confidence (>0.7): Strong semantic match
    Medium confidence (0.4-0.7): Moderate match
    Low confidence (<0.4): Weak match, answer may be unreliable
    
    Args:
        state: Current graph state with metadata containing similarity scores
        
    Returns:
        Updated state with confidence score
    """
    similarity_scores = state["metadata"]["similarity_scores"]
    
    # Convert FAISS L2 distances to similarity scores (0-1)
    # Lower distance = higher similarity
    # Formula: similarity = 1 / (1 + distance)
    similarities = [1 / (1 + score) for score in similarity_scores]
    
    # Calculate average similarity as confidence
    confidence = sum(similarities) / len(similarities) if similarities else 0.0
    
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
    2. RETRIEVE → Fetch relevant chunks from vector DB
    3. FORMAT → Format chunks into context string
    4. GENERATE → LLM generates grounded answer
    5. CONFIDENCE → Calculate confidence score
    6. END → Return final response
    
    Why LangGraph?
    - Provides explicit control over RAG pipeline
    - Each node is testable and debuggable independently
    - Easy to add conditional logic (e.g., re-ranking, query expansion)
    - More maintainable than complex chains
    
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

def process_query(question: str) -> Dict:
    """
    Process a user question through the RAG pipeline.
    
    Args:
        question: User's input question
        
    Returns:
        Dictionary containing answer, context, confidence, and metadata
    """
    
    # Initialize the graph state
    initial_state = {
        "question": question,
        "retrieved_chunks": [],
        "context": "",
        "answer": "",
        "confidence": 0.0,
        "metadata": {}
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
    - Clean, minimal chat interface
    - Displays answer, retrieved context, and confidence score
    - Shows sample queries for easy testing
    - Expands to show retrieved chunks for transparency
    """
    
    # Page configuration
    st.set_page_config(
        page_title="Agentic AI RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Title and description
    st.title("🤖 Agentic AI RAG Chatbot")
    st.markdown("""
    Ask questions about **Agentic AI** based on the official eBook.  
    All answers are strictly grounded in the document - no external knowledge used.
    """)
    
    # Sidebar with information and sample queries
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses:
        - **LangGraph** for RAG workflow
        - **FAISS** for vector search
        - **OpenAI** for embeddings & LLM
        - **Strict grounding** to prevent hallucinations
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
        
        for query in sample_queries:
            if st.button(query, key=query, use_container_width=True):
                st.session_state.current_query = query
        
        st.header("🔧 System Status")
        if "vector_store" in st.session_state:
            st.success("✅ Vector store loaded")
        if "rag_graph" in st.session_state:
            st.success("✅ RAG graph initialized")
    
    # Initialize system components
    if "vector_store" not in st.session_state:
        with st.spinner("Initializing system... This may take a minute on first run."):
            try:
                # Check if API key is available
                if not OPENAI_API_KEY:
                    st.error("❌ OpenAI API key not found. Please set OPENAI_API_KEY in Streamlit secrets or environment variables.")
                    st.stop()
                
                # Create vector store (cached, runs only once)
                st.session_state.vector_store = create_vector_store()
                
                # Create RAG graph
                st.session_state.rag_graph = create_rag_graph()
                
                st.success("✅ System initialized successfully!")
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                st.stop()
    
    # Chat interface
    st.header("💬 Ask a Question")
    
    # Text input for user question
    user_question = st.text_input(
        "Your question:",
        value=st.session_state.get("current_query", ""),
        placeholder="e.g., What is Agentic AI?",
        key="question_input"
    )
    
    # Process query on button click
    if st.button("🔍 Get Answer", type="primary"):
        if user_question.strip():
            with st.spinner("Processing your question..."):
                try:
                    # Process the query through RAG pipeline
                    result = process_query(user_question)
                    
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
                    
                    st.metric(
                        label="Confidence Score",
                        value=f"{confidence_percentage:.1f}%",
                        help=f"{confidence_color} {confidence_label} confidence"
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
                    with st.expander("🔍 Query Metadata", expanded=False):
                        st.json({
                            "num_chunks_retrieved": result["metadata"]["num_chunks"],
                            "page_numbers": result["metadata"]["page_numbers"],
                            "similarity_scores": [f"{s:.4f}" for s in result["metadata"]["similarity_scores"]],
                            "timestamp": datetime.now().isoformat()
                        })
                
                except Exception as e:
                    st.error(f"❌ Error processing query: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>RAG Chatbot | Powered by LangGraph + OpenAI | Interview Task</small>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------------------------
# APPLICATION ENTRY POINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
