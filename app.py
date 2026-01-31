"""
RAG-based AI Chatbot for Agentic AI eBook (100% FREE VERSION)
================================================================
A complete Retrieval-Augmented Generation (RAG) system using LangGraph
with FREE open-source models - NO API KEYS REQUIRED!

GUARANTEED TO WORK:
- Primary: HuggingFace Mistral-7B API (free)
- Fallback 1: Alternative HuggingFace models (faster)
- Fallback 2: Context-based answer from retrieved chunks
- ALWAYS returns an answer!

Uses:
- Sentence Transformers for embeddings (runs locally)
- Multiple LLM fallbacks for reliability
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
import requests
import json

# LangChain imports for document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Hugging Face imports for FREE embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# LangGraph imports for building the RAG workflow
from langgraph.graph import StateGraph, END

# For better LLM output
import re
import time

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
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Multiple LLM options for fallback reliability
# We'll try these in order until one works
LLM_MODELS = [
    {
        "name": "Mistral-7B-Instruct",
        "repo": "mistralai/Mistral-7B-Instruct-v0.2",
        "quality": "high",
        "speed": "medium"
    },
    {
        "name": "Flan-T5-Large",
        "repo": "google/flan-t5-large",
        "quality": "medium",
        "speed": "fast"
    },
    {
        "name": "Flan-T5-Base",
        "repo": "google/flan-t5-base",
        "quality": "medium",
        "speed": "very-fast"
    }
]

# Try to get HuggingFace token from environment or Streamlit secrets
# This is OPTIONAL - works without it!
HF_TOKEN = None
try:
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not HF_TOKEN:
        HF_TOKEN = st.secrets.get("HUGGINGFACEHUB_API_TOKEN", None)
except:
    HF_TOKEN = None


# -------------------------------------------------------------------
# STATE DEFINITION FOR LANGGRAPH
# -------------------------------------------------------------------

class GraphState(TypedDict):
    """
    State object that flows through the LangGraph workflow.
    """
    question: str
    retrieved_chunks: List[Document]
    context: str
    answer: str
    confidence: float
    metadata: Dict
    vector_store: any


# -------------------------------------------------------------------
# MULTI-LEVEL LLM FALLBACK SYSTEM
# -------------------------------------------------------------------

def call_huggingface_model(model_repo: str, prompt: str, timeout: int = 20) -> tuple[str, bool]:
    """
    Call a specific HuggingFace model.
    
    Returns:
        (response_text, success_flag)
    """
    headers = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    api_url = f"https://api-inference.huggingface.co/models/{model_repo}"
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.1,
            "max_new_tokens": 400,
            "return_full_text": False,
        }
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "")
                return text, True
            elif isinstance(result, dict):
                text = result.get("generated_text", "")
                return text, True
        
        return f"Model returned status {response.status_code}", False
        
    except Exception as e:
        return f"Error: {str(e)}", False


def generate_from_context_fallback(context: str, question: str) -> str:
    """
    FINAL FALLBACK: Extract answer directly from context using simple logic.
    This ALWAYS works even if all APIs fail.
    
    This is a basic extractive QA approach - finds the most relevant sentence.
    """
    # Split context into sentences
    sentences = re.split(r'[.!?]+', context)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    # Simple keyword matching - find sentences with question keywords
    question_lower = question.lower()
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    question_words = question_words - {'what', 'is', 'are', 'how', 'why', 'when', 'where', 'the', 'a', 'an'}
    
    # Score each sentence
    scored_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        sentence_words = set(re.findall(r'\b\w+\b', sentence_lower))
        overlap = len(question_words & sentence_words)
        if overlap > 0:
            scored_sentences.append((overlap, sentence))
    
    # Get top sentences
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    
    if scored_sentences:
        # Return top 2-3 most relevant sentences
        top_sentences = [s[1] for s in scored_sentences[:3]]
        answer = ". ".join(top_sentences)
        if not answer.endswith('.'):
            answer += '.'
        return f"Based on the eBook content: {answer}"
    else:
        # If no match, return first chunk of context
        first_chunk = '. '.join(sentences[:2])
        return f"Here's what I found in the eBook: {first_chunk}."


def call_llm_with_fallbacks(prompt: str, context: str, question: str) -> tuple[str, str]:
    """
    Try multiple LLM models in sequence, with a guaranteed fallback.
    
    Returns:
        (answer_text, method_used)
    """
    
    # Try each model in sequence
    for i, model in enumerate(LLM_MODELS):
        try:
            st.info(f"🔄 Trying {model['name']}... (attempt {i+1}/{len(LLM_MODELS)})")
            
            # For Mistral, use [INST] format
            if "mistral" in model['repo'].lower():
                formatted_prompt = prompt
            else:
                # For T5 models, use simpler format
                formatted_prompt = f"""Answer this question based ONLY on the context provided.

Context: {context}

Question: {question}

Answer:"""
            
            response, success = call_huggingface_model(model['repo'], formatted_prompt, timeout=15)
            
            if success and len(response.strip()) > 10:
                # Clean up response
                answer = response.strip()
                answer = re.sub(r'\[INST\].*?\[/INST\]', '', answer, flags=re.DOTALL).strip()
                answer = re.sub(r'</?s>', '', answer).strip()
                
                st.success(f"✅ Success using {model['name']}!")
                return answer, f"{model['name']} (HuggingFace API)"
            else:
                st.warning(f"⚠️ {model['name']} failed, trying next model...")
                time.sleep(2)
                continue
                
        except Exception as e:
            st.warning(f"⚠️ {model['name']} error: {str(e)[:50]}...")
            continue
    
    # If ALL models fail, use context-based fallback (ALWAYS works)
    st.info("🔄 All API models busy. Using context-based extraction (guaranteed to work)...")
    answer = generate_from_context_fallback(context, question)
    return answer, "Context Extraction (No API needed)"


# -------------------------------------------------------------------
# PDF INGESTION & VECTOR STORE CREATION
# -------------------------------------------------------------------

@st.cache_resource(show_spinner="📚 Loading and processing PDF with FREE models...")
def create_vector_store():
    """
    Load PDF, chunk text, generate embeddings, and create vector store.
    """
    
    loader = PyPDFLoader(PDF_URL)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store


# -------------------------------------------------------------------
# LANGGRAPH NODE FUNCTIONS
# -------------------------------------------------------------------

def retrieve_node(state: GraphState) -> GraphState:
    """Node 1: Retrieval"""
    question = state["question"]
    vector_store = state["vector_store"]
    
    docs_with_scores = vector_store.similarity_search_with_score(question, k=TOP_K)
    
    retrieved_chunks = [doc for doc, score in docs_with_scores]
    similarity_scores = [float(score) for doc, score in docs_with_scores]
    
    metadata = {
        "num_chunks": len(retrieved_chunks),
        "similarity_scores": similarity_scores,
        "page_numbers": [doc.metadata.get("page", "N/A") for doc in retrieved_chunks]
    }
    
    state["retrieved_chunks"] = retrieved_chunks
    state["metadata"] = metadata
    
    return state


def format_context_node(state: GraphState) -> GraphState:
    """Node 2: Context Formatting"""
    chunks = state["retrieved_chunks"]
    
    formatted_chunks = []
    for i, doc in enumerate(chunks, 1):
        page_num = doc.metadata.get("page", "Unknown")
        chunk_text = doc.page_content.strip()
        formatted_chunks.append(f"[Source {i} - Page {page_num}]\n{chunk_text}")
    
    context = "\n\n---\n\n".join(formatted_chunks)
    
    state["context"] = context
    
    return state


def generate_answer_node(state: GraphState) -> GraphState:
    """
    Node 3: Answer Generation with GUARANTEED fallbacks.
    
    This ALWAYS returns an answer using one of these methods:
    1. Mistral-7B API (best quality)
    2. Flan-T5-Large API (good quality, faster)
    3. Flan-T5-Base API (decent quality, very fast)
    4. Context extraction (guaranteed to work, no API)
    """
    question = state["question"]
    context = state["context"]
    
    # Build prompt for instruction-following models
    prompt = f"""[INST] You are a helpful assistant answering questions about Agentic AI based STRICTLY on the provided context from an eBook.

STRICT RULES:
1. Use ONLY the information in the context below
2. If the answer is not in the context, say: "I cannot find this information in the provided eBook content."
3. Be concise and direct
4. Quote specific parts when possible

CONTEXT FROM EBOOK:
{context}

QUESTION: {question}

ANSWER (based only on context above): [/INST]"""

    try:
        # Try multiple models with fallback
        answer, method = call_llm_with_fallbacks(prompt, context, question)
        
        # Store which method was used in metadata
        state["metadata"]["generation_method"] = method
        state["answer"] = answer
        
    except Exception as e:
        # Ultimate fallback - should never reach here, but just in case
        st.error(f"Unexpected error: {str(e)}")
        answer = generate_from_context_fallback(context, question)
        state["metadata"]["generation_method"] = "Emergency Context Extraction"
        state["answer"] = answer
    
    return state


def calculate_confidence_node(state: GraphState) -> GraphState:
    """Node 4: Confidence Scoring"""
    similarity_scores = state["metadata"]["similarity_scores"]
    
    confidences = [1 / (1 + score) for score in similarity_scores]
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    confidence = max(0.0, min(1.0, confidence))
    
    state["confidence"] = confidence
    
    return state


# -------------------------------------------------------------------
# LANGGRAPH WORKFLOW CONSTRUCTION
# -------------------------------------------------------------------

def create_rag_graph():
    """Constructs the LangGraph workflow for RAG."""
    
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("format_context", format_context_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("calculate_confidence", calculate_confidence_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate_answer")
    workflow.add_edge("generate_answer", "calculate_confidence")
    workflow.add_edge("calculate_confidence", END)
    
    app = workflow.compile()
    
    return app


# -------------------------------------------------------------------
# HELPER FUNCTION FOR QUERY PROCESSING
# -------------------------------------------------------------------

def process_query(question: str, vector_store) -> Dict:
    """Process a user question through the RAG pipeline."""
    
    initial_state = {
        "question": question,
        "retrieved_chunks": [],
        "context": "",
        "answer": "",
        "confidence": 0.0,
        "metadata": {},
        "vector_store": vector_store
    }
    
    rag_graph = st.session_state.rag_graph
    final_state = rag_graph.invoke(initial_state)
    
    return final_state


# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------

def main():
    """Main Streamlit application for the RAG chatbot."""
    
    st.set_page_config(
        page_title="Agentic AI RAG Chatbot (FREE)",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_graph" not in st.session_state:
        st.session_state.rag_graph = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    
    # Title
    st.title("🤖 Agentic AI RAG Chatbot")
    
    # Show token status
    if HF_TOKEN:
        st.success("✅ Running with HuggingFace token - Better performance & reliability!")
    else:
        st.info("ℹ️ Running without token - Still works! Multi-level fallback ensures you ALWAYS get an answer.")
    
    st.markdown("""
    Ask questions about **Agentic AI** based on the official eBook.  
    **100% FREE** with **GUARANTEED answers** using multi-level fallback system!
    """)
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing FREE models... First run takes 1-2 minutes."):
            try:
                st.session_state.vector_store = create_vector_store()
                st.session_state.rag_graph = create_rag_graph()
                st.session_state.initialized = True
                
                st.success("✅ System initialized successfully!")
                st.info("💡 **Multi-level fallback system active**: If one model is busy, automatically tries alternatives!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                st.info("💡 Please refresh the page.")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **100% FREE Tech Stack:**
        - 🧠 **LLM**: Multi-model fallback system
        - 🔢 **Embeddings**: Sentence Transformers
        - 📊 **Vector DB**: FAISS (local)
        - 🔄 **Workflow**: LangGraph
        - ✅ **Reliability**: GUARANTEED answers!
        """)
        
        st.header("🛡️ Fallback System")
        st.success("""
        **Tries in order:**
        1. Mistral-7B (best)
        2. Flan-T5-Large (good)
        3. Flan-T5-Base (fast)
        4. Context extraction (always works!)
        
        **You ALWAYS get an answer!**
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
            if st.button(query, key=f"sample_{query}", use_container_width=True):
                st.session_state.current_query = query
                st.rerun()
        
        st.header("🔧 System Status")
        if st.session_state.initialized:
            st.success("✅ Ready to answer!")
    
    # Chat interface
    st.header("💬 Ask a Question")
    
    user_question = st.text_input(
        "Your question:",
        value=st.session_state.current_query,
        placeholder="e.g., What is Agentic AI?",
        key="question_input"
    )
    
    if st.button("🔍 Get Answer", type="primary"):
        if not st.session_state.initialized:
            st.error("❌ System is still initializing.")
            st.stop()
        
        if user_question.strip():
            with st.spinner("🤔 Processing... Multi-level fallback ensures you get an answer!"):
                try:
                    result = process_query(user_question, st.session_state.vector_store)
                    
                    # Display results
                    st.subheader("📖 Answer")
                    st.markdown(result["answer"])
                    
                    # Show which method was used
                    generation_method = result["metadata"].get("generation_method", "Unknown")
                    st.caption(f"*Generated using: {generation_method}*")
                    
                    # Confidence score
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
                    
                    col1= st.columns(1)
                    with col1:
                        st.metric(
                            label="Confidence Score",
                            value=f"{confidence_percentage:.1f}%",
                            help=f"{confidence_color} {confidence_label} confidence"
                        )
                    
                    
                    # Retrieved context
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
                    
                    # Technical details
                    with st.expander("🔍 Technical Details", expanded=False):
                        st.json({
                            "embedding_model": EMBEDDING_MODEL,
                            "generation_method": generation_method,
                            "num_chunks_retrieved": result["metadata"]["num_chunks"],
                            "page_numbers": result["metadata"]["page_numbers"],
                            "similarity_scores": [f"{s:.4f}" for s in result["metadata"]["similarity_scores"]]
                            
                        })
                
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.info("💡 This should not happen with our fallback system. Please try again.")
        else:
            st.warning("⚠️ Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>100% FREE RAG Chatbot | Multi-Level Fallback System | GUARANTEED Answers | Interview Task</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()