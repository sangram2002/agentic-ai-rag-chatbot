# 🚀 Streamlit Cloud Deployment Guide

Follow these steps to deploy your RAG chatbot to Streamlit Cloud.

## Prerequisites

- GitHub account
- Streamlit Cloud account (free at [share.streamlit.io](https://share.streamlit.io))
- OpenAI API key

---

## Step 1: Push to GitHub

1. **Create a new repository on GitHub**
   - Go to github.com and create a new public repository
   - Name it: `rag-agentic-ai` (or your preferred name)

2. **Push your code**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: RAG chatbot with LangGraph"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/rag-agentic-ai.git
   git push -u origin main
   ```

---

## Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create new app**
   - Click "New app"
   - Select your repository: `YOUR_USERNAME/rag-agentic-ai`
   - Select branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

---

## Step 3: Configure Secrets

1. **Add OpenAI API Key**
   - In your deployed app, click "⚙️ Settings" (bottom right)
   - Go to "Secrets" section
   - Add the following:
   
   ```toml
   OPENAI_API_KEY = "sk-your-actual-api-key-here"
   ```

2. **Save and reboot**
   - Click "Save"
   - The app will automatically restart with the secrets

---

## Step 4: Test Your Deployment

1. **Wait for deployment** (may take 2-3 minutes on first run)
2. **Test with sample queries** from the sidebar
3. **Share your app URL** (e.g., `https://your-app-name.streamlit.app`)

---

## Troubleshooting

### Issue: "OpenAI API key not found"
**Solution**: Make sure you added `OPENAI_API_KEY` in Streamlit secrets (not environment variables)

### Issue: "Module not found"
**Solution**: Check that all packages in `requirements.txt` are spelled correctly and versions are compatible

### Issue: App takes too long to load
**Solution**: First load processes the PDF - this is normal. Subsequent loads use cached vector store.

### Issue: Out of memory
**Solution**: Streamlit Cloud free tier has memory limits. Try:
- Reduce `TOP_K` to 3
- Use smaller embedding model
- Upgrade to Streamlit Cloud Pro

---

## Advanced Configuration

### Custom Domain
- Go to Settings → General
- Add your custom domain
- Follow DNS configuration instructions

### Resource Limits
- Free tier: 1 GB RAM, shared CPU
- For production: Consider Streamlit Cloud Pro or self-hosting

### Monitoring
- View logs: Click "Manage app" → "Logs"
- Check analytics: See user engagement stats

---

## Cost Considerations

### OpenAI API Costs (Approximate)
- **Embedding**: ~$0.00001 per 1K tokens
  - One-time PDF processing: ~$0.01
- **LLM (GPT-4o-mini)**: ~$0.00015 per 1K input tokens
  - Per query: ~$0.0005 (with 4 chunks × 700 chars)
- **Total per user session**: ~$0.01
- **100 users per day**: ~$1/day = ~$30/month

### Free Tier Limits
- Streamlit Cloud: Free for public apps
- OpenAI: Pay-per-use, no free tier for API
  - Set usage limits in OpenAI dashboard to prevent overuse

---

## Security Best Practices

1. **Never commit API keys to Git**
   - Use `.gitignore` for `.env` files
   - Always use Streamlit secrets for deployment

2. **Set OpenAI rate limits**
   - Go to OpenAI dashboard → Usage limits
   - Set monthly cap (e.g., $10)

3. **Monitor usage**
   - Check Streamlit analytics
   - Monitor OpenAI usage dashboard

---

## Alternative Deployment Options

### Option 1: Docker + Cloud Run
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501"]
```

### Option 2: Heroku
- Add `Procfile`: `web: streamlit run app.py`
- Set config vars for `OPENAI_API_KEY`

### Option 3: AWS EC2
- Launch t2.micro instance
- Install dependencies
- Run with `tmux` or `systemd`

---

## Updating Your App

To update after deployment:

```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push

# Streamlit Cloud auto-deploys on push
```

---

## Sharing Your App

**Public URL**: `https://your-app-name.streamlit.app`

**Interview Tips**:
- Share the URL in your job application
- Add it to your GitHub README
- Include it on your resume/portfolio
- Demo it live during technical interviews

---

## Support

- **Streamlit Docs**: [docs.streamlit.io](https://docs.streamlit.io)
- **Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
- **GitHub Issues**: Report bugs in your repo

---

**Happy Deploying! 🚀**
