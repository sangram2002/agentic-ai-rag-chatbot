# Agentic AI RAG Chatbot

A simple chatbot that answers questions from the Agentic AI eBook. Built with free tools and designed to be fast and easy to understand.

## What It Does

You ask a question, it finds the relevant parts in the PDF, and gives you an answer. That's it. No complicated setup, no paid APIs (except Groq which is free).

## Why This Stack?

I tested a few options and here's what worked best:

- **Groq**: Responses in 1-2 seconds. HuggingFace was taking 10-20 seconds and failing half the time.
- **Local Embeddings**: Runs on your machine. No API costs, no rate limits.
- **FAISS**: Simple vector database. Works offline once set up.
- **LangGraph**: Makes the workflow clear. Easy to debug and explain in interviews.

## Quick Start

```bash
# Install stuff
pip install -r requirements.txt

# Get a free Groq API key
# Go to: https://console.groq.com
# Sign up and create an API key

# Add it to .streamlit/secrets.toml
mkdir .streamlit
echo 'GROQ_API_KEY = "your-key-here"' > .streamlit/secrets.toml

# Run it
streamlit run app.py
```

First run takes 1-2 minutes to download the embedding model. After that, each query is 1-2 seconds.

## How It Works

The flow is straightforward:

1. **Load PDF**: Download the eBook and split it into chunks
2. **Create Embeddings**: Convert chunks to numbers (happens once)
3. **User Asks Question**: Find the 4 most relevant chunks
4. **Send to Groq**: AI reads those chunks and answers
5. **Show Result**: Display answer with confidence score

That's the whole thing. No magic.

## Code Structure

```
app.py (500 lines)
├── setup_vector_database()    # Loads PDF, creates search index
├── find_relevant_chunks()      # Searches for matching text
├── prepare_context()           # Formats chunks nicely
├── create_answer()             # Asks Groq for answer
├── calculate_confidence()      # Scores the result
└── main()                      # Streamlit UI
```

Each function does one thing. Easy to test and modify.

## Key Design Decisions

**Why chunks?**
The PDF is too big to send all at once. Chunking lets us find exactly the relevant parts. I use 700 characters with 100 overlap so important info doesn't get split across boundaries.


**Why local embeddings?**
Running sentence-transformers locally means no API costs and no internet needed after setup. Worth the 1-minute wait on first run.

**Why LangGraph?**
Could've used a simple chain, but LangGraph makes the steps explicit. Each node can be tested separately. Plus it's easier to explain what's happening.

## What Could Be Better

Things I'd add with more time:

- **Caching**: Store common questions to skip Groq API entirely
- **Streaming**: Show the answer as it's being generated
- **Better chunking**: Right now it splits at 700 chars. Could be smarter about where to split.
- **Re-ranking**: After initial retrieval, re-score chunks based on the specific question

But for an interview task, keeping it simple seemed smarter.

## Files

- `app.py` - Main application (well commented)
- `requirements.txt` - Dependencies
- `GROQ_SETUP.md` - Detailed API key instructions
- `INTERVIEW_PREP.md` - Questions you might get asked

## Deployment

Works on Streamlit Cloud. Just:
1. Push to GitHub
2. Connect on share.streamlit.io
3. Add GROQ_API_KEY in secrets
4. Done

## Gotchas

**"Still setting up"**: First run downloads an 80MB model. Wait for it.

**Rate limits**: Free Groq tier is 30 requests/min. Fine for demos. For production you'd need Groq Pro or multiple keys.

**No conversation memory**: Each question is independent. If you wanted chat history, you'd need to modify the state to include previous messages.

## Why I Built It This Way

I wanted something that:
- Actually works reliably (not just works once)
- Is easy to explain in an interview
- Uses modern tools (LangGraph is new, shows I keep up)
- Doesn't require paid APIs
- Can be deployed in 5 minutes

Could've made it fancier, but simple and working beats complex and buggy.

## Questions?

The code is heavily commented. If something's unclear, read the comments in that function. I tried to explain not just what the code does, but why I made that choice.