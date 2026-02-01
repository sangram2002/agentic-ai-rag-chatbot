"""
RAG Chatbot for Agentic AI eBook
This is a simple question-answering system that reads from a PDF and answers questions.
It uses free tools and runs fast thanks to Groq API.

Built for AI Engineer Interview Task
"""

import os
import streamlit as st
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Import tools for PDF and text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Import embedding model (runs on your computer)
from langchain_huggingface import HuggingFaceEmbeddings

# Import Groq for fast LLM responses
from langchain_groq import ChatGroq

# Import LangGraph for building the workflow
from langgraph.graph import StateGraph, END

# Settings and configuration
PDF_LINK = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
MODEL_NAME = "llama-3.3-70b-versatile"  # Best Groq model right now
CHUNK_SIZE = 700  # How long each text chunk should be
CHUNK_OVERLAP = 100  # How much chunks overlap
TOP_K = 4  # Number of relevant chunks to retrieve
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Get Groq API key from environment or Streamlit secrets
GROQ_API_KEY = None
try:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
except:
    GROQ_API_KEY = None


# This function loads the PDF and prepares it for searching
# It only runs once thanks to st.cache_resource
@st.cache_resource(show_spinner="Loading PDF and creating search index...")
def setup_vector_database():
    """
    This function does three important things:
    1. Downloads and reads the PDF
    2. Splits the text into smaller pieces (chunks)
    3. Converts chunks into numbers (embeddings) so we can search them
    
    Why do we need chunks?
    - The PDF is too big to send to the AI all at once
    - Smaller chunks are easier to search
    - We can find exactly the relevant parts
    
    Returns a searchable database of the PDF content
    """
    
    # Step 1: Load the PDF file
    print("Downloading PDF...")
    loader = PyPDFLoader(PDF_LINK)
    pages = loader.load()
    
    # Step 2: Split the PDF into smaller chunks
    # This makes it easier to find relevant information
    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Split on paragraphs first, then sentences
    )
    text_chunks = splitter.split_documents(pages)
    
    # Step 3: Create embeddings (convert text to numbers)
    # This model runs on your computer, no internet needed after download
    print("Creating embeddings (this takes a minute first time)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Step 4: Build a searchable database
    # FAISS is like a smart search engine for text
    print("Building search database...")
    database = FAISS.from_documents(text_chunks, embedding_model)
    
    return database


# Step 1 of the workflow: Find relevant chunks
def find_relevant_chunks(current_state):
    """
    This function searches the database for chunks that match the question.
    It uses similarity search to find the best matches.
    
    How it works:
    - Takes the user's question
    - Converts it to numbers (embedding)
    - Finds the closest matching chunks
    - Returns the top matches with scores
    """
    
    question = current_state["question"]
    database = current_state["vector_store"]
    
    # Search for similar chunks in the database
    results = database.similarity_search_with_score(question, k=TOP_K)
    
    # Separate the chunks and their scores
    chunks = [doc for doc, score in results]
    scores = [float(score) for doc, score in results]
    
    # Save the results
    current_state["retrieved_chunks"] = chunks
    current_state["metadata"] = {
        "num_chunks": len(chunks),
        "similarity_scores": scores,
        "page_numbers": [doc.metadata.get("page", "N/A") for doc in chunks]
    }
    
    return current_state


# Step 2 of the workflow: Format the chunks nicely
def prepare_context(current_state):
    """
    This takes the retrieved chunks and formats them nicely for the AI.
    We add page numbers so the user knows where information came from.
    """
    
    chunks = current_state["retrieved_chunks"]
    
    # Format each chunk with its page number
    formatted_parts = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk.metadata.get("page", "Unknown")
        text = chunk.page_content.strip()
        formatted_parts.append(f"[Source {i} - Page {page}]\n{text}")
    
    # Join everything together with separators
    full_context = "\n\n---\n\n".join(formatted_parts)
    current_state["context"] = full_context
    
    return current_state


# Step 3 of the workflow: Generate the answer using AI
def create_answer(current_state):
    """
    This is where the magic happens. We send the context and question to Groq,
    and it generates an answer based ONLY on the provided information.
    
    Important: The AI is instructed to only use the context we provide.
    If the answer isn't in the context, it should say so.
    """
    
    question = current_state["question"]
    context = current_state["context"]
    
    # Check if we have the API key
    if not GROQ_API_KEY:
        current_state["answer"] = "Error: Groq API key not found. Please add it to your secrets."
        return current_state
    
    # Set up the AI model
    ai_model = ChatGroq(
        model=MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        temperature=0.1,  # Low temperature means more focused answers
        max_tokens=1024,
    )
    
    # Create the instructions for the AI
    # This is the most important part - we tell it to ONLY use the context
    instructions = f"""You are helping someone learn about Agentic AI from an eBook.

Important rules:
1. Only use information from the context below
2. If the answer is not in the context, say: "I cannot find this information in the eBook"
3. Be direct and helpful
4. Don't add information from outside the context

Context from the eBook:
{context}

Question:
{question}

Answer based on the context above:"""

    # Ask the AI for an answer
    try:
        response = ai_model.invoke(instructions)
        answer = response.content.strip()
        
        # Make sure we got a valid answer
        if len(answer) < 5:
            answer = "I cannot find this information in the eBook"
            
    except Exception as error:
        # If something goes wrong, show a helpful error
        answer = f"Error getting answer: {str(error)[:100]}"
    
    current_state["answer"] = answer
    return current_state


# Step 4 of the workflow: Calculate confidence score
def calculate_confidence(current_state):
    """
    This calculates how confident we are about the answer.
    It's based on how well the retrieved chunks matched the question.
    
    Higher score = better match = more confident answer
    Lower score = weaker match = less confident answer
    """
    
    scores = current_state["metadata"]["similarity_scores"]
    
    # Convert similarity scores to confidence (0 to 1)
    # Lower distance means higher similarity
    confidence_values = [1 / (1 + score) for score in scores]
    
    # Take the average
    if confidence_values:
        average_confidence = sum(confidence_values) / len(confidence_values)
    else:
        average_confidence = 0.0
    
    # Make sure it's between 0 and 1
    final_confidence = max(0.0, min(1.0, average_confidence))
    
    current_state["confidence"] = final_confidence
    return current_state


# This builds the complete workflow (the graph)
def build_workflow():
    """
    This connects all the steps together into a workflow.
    
    The flow goes like this:
    1. Find relevant chunks (retrieve)
    2. Format them nicely (prepare_context)
    3. Generate answer with AI (create_answer)
    4. Calculate confidence (calculate_confidence)
    5. Done!
    
    LangGraph makes this easy to understand and modify.
    """
    
    # Create a new workflow
    workflow = StateGraph(dict)
    
    # Add each step as a node
    workflow.add_node("retrieve", find_relevant_chunks)
    workflow.add_node("format_context", prepare_context)
    workflow.add_node("generate_answer", create_answer)
    workflow.add_node("calculate_confidence", calculate_confidence)
    
    # Connect the steps in order
    workflow.set_entry_point("retrieve")  # Start here
    workflow.add_edge("retrieve", "format_context")
    workflow.add_edge("format_context", "generate_answer")
    workflow.add_edge("generate_answer", "calculate_confidence")
    workflow.add_edge("calculate_confidence", END)  # End here
    
    # Build and return the workflow
    return workflow.compile()


# This runs a question through the complete workflow
def answer_question(question, database):
    """
    This is the main function that processes a question.
    It runs through all the steps and returns the final result.
    """
    
    # Set up the initial state
    state = {
        "question": question,
        "retrieved_chunks": [],
        "context": "",
        "answer": "",
        "confidence": 0.0,
        "metadata": {},
        "vector_store": database
    }
    
    # Run the workflow
    workflow = st.session_state.rag_graph
    final_state = workflow.invoke(state)
    
    return final_state


# Main application starts here
def main():
    """
    This is the main app that users interact with.
    It shows the UI and handles button clicks.
    """
    
    # Set up the page
    st.set_page_config(
        page_title="Agentic AI Chatbot",
        page_icon="⚡",
        layout="wide"
    )
    
    # Initialize session state variables
    # These keep track of things between page refreshes
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "rag_graph" not in st.session_state:
        st.session_state.rag_graph = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "current_query" not in st.session_state:
        st.session_state.current_query = ""
    
    # Show the title
    st.title("⚡ Agentic AI Chatbot")
    
    # Check if we have the API key
    if GROQ_API_KEY:
        st.success("Ready! Groq API key found.")
    else:
        st.error("Missing Groq API key!")
        st.info("""
        To use this app, you need a free Groq API key:
        1. Go to https://console.groq.com
        2. Sign up (it's free)
        3. Create an API key
        4. Add it to Streamlit secrets as GROQ_API_KEY
        """)
        st.stop()
    
    st.markdown("Ask questions about Agentic AI based on the eBook.")
    
    # Initialize the system if not done yet
    if not st.session_state.initialized:
        with st.spinner("Setting up... This takes 1-2 minutes the first time."):
            try:
                # Load the PDF and create the search database
                st.session_state.vector_store = setup_vector_database()
                
                # Build the workflow
                st.session_state.rag_graph = build_workflow()
                
                # Mark as ready
                st.session_state.initialized = True
                
                st.success("All set! You can ask questions now.")
                st.balloons()
                
            except Exception as error:
                st.error(f"Setup failed: {str(error)}")
                st.info("Try refreshing the page.")
                st.stop()
    
    # Sidebar with information
    with st.sidebar:
        st.header("About This App")
        st.markdown("""
        This app uses:
        - Groq API (super fast AI)
        - Local embeddings (runs on your computer)
        - FAISS database (for searching)
        - LangGraph (for workflow)
        """)
        
        st.header("Performance")
        st.info("""
        First run: 1-2 minutes
        Each question: 1-2 seconds
        All free to use!
        """)
        
        st.header("Try These Questions")
        example_questions = [
            "What is Agentic AI?",
            "What are AI agents composed of?",
            "How do agent workflows differ from LLM chains?",
            "What are the limitations of agentic systems?",
            "What is the difference between traditional RAG and Agentic RAG?",
            "How does the ebook define tool usage?"
        ]
        
        for question in example_questions:
            if st.button(question, key=f"btn_{question}", use_container_width=True):
                st.session_state.current_query = question
                st.rerun()
    
    # Main chat area
    st.header("Ask Your Question")
    
    # Input box for the question
    user_input = st.text_input(
        "Type your question here:",
        value=st.session_state.current_query,
        placeholder="For example: What is Agentic AI?",
        key="user_question"
    )
    
    # When user clicks the button
    if st.button("Get Answer", type="primary"):
        
        # Make sure everything is ready
        if not st.session_state.initialized:
            st.error("Still setting up. Please wait.")
            st.stop()
        
        if not st.session_state.vector_store:
            st.error("Database not loaded. Try refreshing.")
            st.stop()
        
        if not st.session_state.rag_graph:
            st.error("Workflow not ready. Try refreshing.")
            st.stop()
        
        # Check if user entered something
        if user_input.strip():
            with st.spinner("Thinking..."):
                try:
                    import time
                    start = time.time()
                    
                    # Get the answer
                    result = answer_question(user_input, st.session_state.vector_store)
                    
                    end = time.time()
                    elapsed = end - start
                    
                    # Show the answer
                    st.subheader("Answer")
                    st.markdown(result["answer"])
                    
                    st.caption(f"Generated in {elapsed:.2f} seconds")
                    
                    # Show confidence score
                    confidence = result["confidence"]
                    confidence_percent = confidence * 100
                    
                    if confidence >= 0.7:
                        color = "🟢"
                        label = "High"
                    elif confidence >= 0.4:
                        color = "🟡"
                        label = "Medium"
                    else:
                        color = "🔴"
                        label = "Low"
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="Confidence",
                            value=f"{confidence_percent:.1f}%",
                            help=f"{color} {label} confidence"
                        )
                    with col2:
                        st.metric(
                            label="Speed",
                            value=f"{elapsed:.2f}s",
                            help="Response time"
                        )
                    
                    # Show the source chunks (optional)
                    with st.expander("Show Source Chunks"):
                        st.markdown("These are the parts of the eBook used to answer:")
                        
                        for i, chunk in enumerate(result["retrieved_chunks"], 1):
                            page = chunk.metadata.get("page", "?")
                            score = result["metadata"]["similarity_scores"][i-1]
                            match_percent = (1 / (1 + score)) * 100
                            
                            st.markdown(f"**Chunk {i}** from Page {page} (Match: {match_percent:.1f}%)")
                            st.text_area(
                                f"content_{i}",
                                chunk.page_content,
                                height=150,
                                label_visibility="collapsed"
                            )
                            st.markdown("---")
                    
                    # Show technical details (optional)
                    with st.expander("Technical Details"):
                        st.json({
                            "model": MODEL_NAME,
                            "embedding_model": EMBEDDING_MODEL,
                            "chunks_used": result["metadata"]["num_chunks"],
                            "pages": result["metadata"]["page_numbers"],
                            "processing_time": f"{elapsed:.2f}s"
                        })
                
                except Exception as error:
                    st.error(f"Error: {str(error)}")
                    st.info("Check your API key and try again.")
        else:
            st.warning("Please type a question first.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>Powered by Groq </small>
    </div>
    """, unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    main()