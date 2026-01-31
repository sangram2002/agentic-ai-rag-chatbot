# 🤖 RAG-based AI Chatbot for Agentic AI eBook

A production-ready Retrieval-Augmented Generation (RAG) chatbot built with **LangGraph** that answers questions strictly based on the [Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf).

**Interview Task Implementation** - AI Engineer Intern Position

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Sample Queries](#sample-queries)
- [Technical Deep Dive](#technical-deep-dive)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

---

## ✨ Features

- **Strict Grounding**: Answers based ONLY on the eBook content, no hallucinations
- **LangGraph Workflow**: Explicit, debuggable RAG pipeline with multiple nodes
- **Confidence Scoring**: Provides transparency on answer reliability
- **Context Display**: Shows retrieved chunks for verification
- **Clean UI**: Simple Streamlit interface for easy interaction
- **Production-Ready**: Modular, well-commented, deployable code

---

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                              │
│                 (User Interaction Layer)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  LANGGRAPH WORKFLOW                          │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐  │
│  │ RETRIEVE │──▶│  FORMAT  │──▶│ GENERATE │──▶│CONFIDENCE│ │
│  │   NODE   │   │   NODE   │   │   NODE   │   │  NODE   │  │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘  │
│       │              │              │              │         │
│       ▼              ▼              ▼              ▼         │
│   Get chunks    Format text    LLM Answer    Score result   │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  VECTOR DATABASE (FAISS)                     │
│                                                              │
│  📄 PDF → 📝 Chunks → 🔢 Embeddings → 🗄️ Storage           │
└─────────────────────────────────────────────────────────────┘
```

### RAG Pipeline Flow

1. **PDF Ingestion** (Runs once, cached)
   - Download Agentic AI eBook
   - Extract text from all pages
   - Split into ~700 character chunks with 100 character overlap
   - Generate embeddings using OpenAI's `text-embedding-3-small`
   - Store in FAISS vector database

2. **Query Processing** (Per user question)
   - **Node 1 - Retrieve**: Find top-4 most similar chunks using semantic search
   - **Node 2 - Format**: Structure chunks into coherent context with metadata
   - **Node 3 - Generate**: LLM creates answer using ONLY the provided context
   - **Node 4 - Confidence**: Calculate reliability score from similarity metrics
   - **Return**: Answer + Context + Confidence score

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Git (for cloning)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/rag-agentic-ai.git
   cd rag-agentic-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up OpenAI API key**
   
   **Option A: Environment variable** (Recommended for local development)
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```
   
   **Option B: Streamlit secrets** (Recommended for deployment)
   
   Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-api-key-here"
   ```

---

## ▶️ How to Run

### Local Development

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repository
4. Add `OPENAI_API_KEY` in Streamlit Cloud secrets settings

---

## 📝 Sample Queries

Try these questions to test the chatbot:

### Basic Concepts
1. **"What is Agentic AI?"**
   - Tests understanding of core definition
   - Expected: High confidence answer with clear explanation

2. **"What are AI agents composed of?"**
   - Tests retrieval of structural information
   - Expected: Components breakdown from the eBook

### Technical Comparisons
3. **"How do agent workflows differ from LLM chains?"**
   - Tests comparative understanding
   - Expected: Detailed comparison from relevant sections

4. **"What is the difference between traditional RAG and Agentic RAG?"**
   - Tests specific technical distinctions
   - Expected: Focused answer on RAG evolution

### Advanced Topics
5. **"What are the limitations of agentic systems?"**
   - Tests extraction of critical analysis
   - Expected: List of limitations mentioned in eBook

6. **"How does the ebook define tool usage?"**
   - Tests specific definition retrieval
   - Expected: Precise definition with context

### Edge Cases
7. **"What is quantum computing?"** *(Not in eBook)*
   - Tests grounding enforcement
   - Expected: "I cannot find this information in the provided eBook content."

---

## 🔬 Technical Deep Dive

### Why Chunking?

**Problem**: LLMs have token limits (~8k-128k tokens), but documents can be much larger.

**Solution**: Split documents into smaller, overlapping chunks.

**Benefits**:
- Each chunk fits within context window
- Improves retrieval precision (find exact relevant section)
- Overlap preserves context at boundaries
- Each chunk gets its own embedding for semantic search

**Our Configuration**:
- Chunk size: 700 characters (~175-200 tokens)
- Overlap: 100 characters (~25 tokens)
- Separator priority: Double newline → newline → period → space

### How Embeddings Work

**Concept**: Convert text into high-dimensional vectors (1536 dimensions for our model)

**Process**:
1. Each chunk → Embedding model → Vector [0.123, -0.456, ...]
2. Semantically similar text → Similar vectors
3. Vector similarity = Semantic similarity

**Search**:
- User question → Embedding vector
- Find nearest vectors in database (cosine similarity / L2 distance)
- Return corresponding chunks

### LangGraph Workflow Design

**Why LangGraph over Simple Chains?**

| Feature | Simple Chain | LangGraph |
|---------|-------------|-----------|
| Debuggability | Hard to debug black box | Each node is testable |
| Flexibility | Linear flow only | Conditional routing possible |
| State Management | Implicit | Explicit TypedDict |
| Observability | Limited | Full state inspection |
| Modularity | Tightly coupled | Loosely coupled nodes |

**Our Graph Structure**:
```python
class GraphState(TypedDict):
    question: str              # User input
    retrieved_chunks: List     # From vector DB
    context: str              # Formatted for LLM
    answer: str               # Generated response
    confidence: float         # 0-1 score
    metadata: Dict            # Debugging info
```

**Node Functions**:
1. `retrieve_node`: Vector similarity search
2. `format_context_node`: Structure chunks for LLM
3. `generate_answer_node`: Grounded answer generation
4. `calculate_confidence_node`: Score computation

### Grounding Enforcement

**The Problem**: LLMs can hallucinate or use pre-trained knowledge instead of provided context.

**Our Solution**: Strict prompt engineering

```python
prompt = f"""You are a helpful assistant answering questions about Agentic AI 
based STRICTLY on the provided context from an eBook.

STRICT RULES:
1. Use ONLY the information provided in the context below
2. Do NOT use any external knowledge or make assumptions
3. If the answer is not found in the context, respond with: 
   "I cannot find this information in the provided eBook content."
4. Quote or reference specific parts of the context when possible
5. Be concise but comprehensive

CONTEXT FROM EBOOK:
{context}

QUESTION:
{question}

ANSWER (based only on the context above):"""
```

**Why This Works**:
- Explicit instruction to use only context
- Clear fallback response for missing information
- Temperature = 0 for deterministic output
- Context provided immediately before question

### Confidence Scoring Logic

**Input**: FAISS returns L2 distance scores for each retrieved chunk

**Problem**: Lower distance = higher similarity (inverse relationship)

**Conversion**:
```python
similarity = 1 / (1 + distance)
```

**Aggregation**: Average across top-K chunks

**Interpretation**:
- **High (>0.7)**: Strong semantic match, reliable answer
- **Medium (0.4-0.7)**: Moderate match, generally reliable
- **Low (<0.4)**: Weak match, answer may be speculative

**Limitations**: 
- Doesn't account for contradiction between chunks
- Doesn't measure if LLM correctly used the context
- Future: Could add answer-context entailment scoring

---

## ⚠️ Known Limitations

### 1. **Chunking Boundary Issues**
- Important information spanning chunk boundaries may be fragmented
- Overlap helps but doesn't completely solve the problem
- **Impact**: Some complex multi-paragraph concepts may not be fully captured

### 2. **No Re-ranking**
- Initial retrieval uses only embedding similarity
- Doesn't consider query-specific relevance
- **Future**: Add cross-encoder re-ranking for better precision

### 3. **Single-hop Reasoning Only**
- Can't combine information from multiple disparate sections
- No multi-step reasoning across chunks
- **Future**: Implement iterative retrieval or query decomposition

### 4. **No Context Compression**
- Sends all top-K chunks to LLM
- May include redundant or less relevant information
- **Future**: Add LLM-based context compression

### 5. **Static Chunk Size**
- Uses same chunk size for all content types
- Some sections may benefit from larger/smaller chunks
- **Future**: Implement dynamic chunking based on content structure

### 6. **Confidence Score Simplicity**
- Only based on retrieval similarity, not answer quality
- Doesn't detect hallucinations or incorrect inference
- **Future**: Add answer verification step

### 7. **No Conversational Memory**
- Each query is independent
- Can't handle follow-up questions or context from previous turns
- **Future**: Implement conversation history in state

---

## 🚀 Future Improvements

### Short-term (Can implement in 1-2 days)

1. **Query Expansion**
   - Generate multiple phrasings of user question
   - Retrieve more diverse relevant chunks
   - Improves recall for ambiguous queries

2. **Hybrid Search**
   - Combine semantic (embedding) search with keyword (BM25) search
   - Better handles specific terms and proper nouns
   - Implementation: Add BM25 retriever alongside FAISS

3. **Answer Citation**
   - Extract specific sentences from chunks used in answer
   - Show inline citations with chunk/page numbers
   - Improves transparency and verifiability

4. **Streaming Responses**
   - Stream LLM output token-by-token
   - Better UX for longer answers
   - Streamlit supports `st.write_stream()`

### Medium-term (1 week)

5. **Multi-query Retrieval**
   - Generate sub-questions for complex queries
   - Retrieve for each sub-question
   - Combine results for comprehensive answer

6. **Conversation History**
   - Maintain chat history in session state
   - Include context from previous turns
   - Enable follow-up questions

7. **Advanced Chunking**
   - Semantic chunking (split on topic boundaries)
   - Hierarchical chunking (parent-child relationships)
   - Better preserves document structure

8. **Evaluation Suite**
   - Ground truth Q&A pairs
   - Automated metrics (BLEU, ROUGE, semantic similarity)
   - Regression testing for changes

### Long-term (2+ weeks)

9. **Agentic RAG**
   - Add routing: decide if query needs retrieval
   - Add tool use: search, calculate, visualize
   - Self-correction: verify answer quality

10. **Multi-document Support**
    - Ingest multiple PDFs or sources
    - Tag chunks by source
    - Enable source filtering in queries

11. **Fine-tuned Embeddings**
    - Train domain-specific embedding model
    - Improve retrieval for technical terms
    - May require labeled query-document pairs

12. **Production Monitoring**
    - Log all queries and responses
    - Track confidence score distribution
    - Alert on low-confidence answers
    - A/B testing for improvements

---

## 📊 Performance Metrics

*Note: These are example metrics - actual performance depends on query complexity*

| Metric | Value | Notes |
|--------|-------|-------|
| Avg Response Time | 2-3s | Includes retrieval + generation |
| First Load Time | 30-45s | One-time PDF processing + embedding |
| Chunk Count | ~150-200 | Depends on PDF length |
| Embedding Dimensions | 1536 | OpenAI text-embedding-3-small |
| Top-K Retrieved | 4 | Configurable |
| Cost per Query | ~$0.001 | OpenAI API costs (embedding + LLM) |

---

## 🤝 Contributing

This is an interview task implementation, but suggestions are welcome!

**Areas for contribution**:
- Better chunking strategies
- Improved confidence scoring
- Additional evaluation metrics
- UI/UX enhancements

---

## 📄 License

This project is for interview/educational purposes.  
The Agentic AI eBook content is property of its respective authors.

---

## 🙏 Acknowledgments

- **LangChain** team for excellent RAG tooling
- **Anthropic** for Claude (used for code review)
- **OpenAI** for embeddings and LLM APIs
- **Streamlit** for rapid UI development
- **Konverge.ai** for the Agentic AI eBook

---

## 📞 Contact

**Candidate Name**: [Your Name]  
**Email**: [your.email@example.com]  
**GitHub**: [github.com/yourusername]  
**LinkedIn**: [linkedin.com/in/yourprofile]

---

## 🎯 Interview Discussion Points

### Architecture Decisions

1. **Why FAISS over Pinecone?**
   - FAISS: Free, local, no API limits, sufficient for single-document use case
   - Pinecone: Better for production, multi-user, cloud-native
   - Trade-off: Chose simplicity and cost-effectiveness

2. **Why LangGraph?**
   - More control than LangChain LCEL chains
   - Better debugging and observability
   - Easier to explain in interview setting
   - Sets foundation for advanced agentic features

3. **Why OpenAI embeddings vs open-source?**
   - Quality: State-of-the-art performance
   - Reliability: Consistent, well-documented
   - Speed: Fast inference
   - Trade-off: Cost and API dependency (could use HuggingFace for cost savings)

### Code Quality Highlights

- **Comprehensive comments**: Every function, parameter, and design choice explained
- **Type hints**: All functions use proper TypedDict and type annotations
- **Error handling**: Graceful failures with user-friendly messages
- **Caching**: Vector store cached to avoid re-processing
- **Modularity**: Each node is independent and testable
- **Production-ready**: Can deploy to Streamlit Cloud immediately

### Potential Interview Questions & Answers

**Q: How would you handle updates to the PDF?**  
A: Add a version check + re-ingestion trigger. Could use file hash to detect changes, then clear cache and rebuild vector store.

**Q: How would you scale this to 100 PDFs?**  
A: (1) Use Pinecone for distributed vector search, (2) Add metadata filtering by document, (3) Implement batch embedding to reduce API calls, (4) Add document routing to select relevant PDFs per query.

**Q: How do you prevent hallucinations?**  
A: (1) Strict prompt engineering with explicit rules, (2) Temperature = 0 for determinism, (3) Context-only instruction, (4) Verification layer could add entailment checking.

**Q: Why not use GPT-4?**  
A: GPT-4o-mini is 10-20x cheaper, faster, and sufficient for straightforward RAG. GPT-4 is better for complex reasoning, but adds cost and latency.

---

**Built with ❤️ for the AI Engineer Interview**

*Last Updated: January 2026*
