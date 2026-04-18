FROM python:3.11-slim

# System dependencies for XGBoost and FAISS
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Pre-build FAISS index with built-in knowledge
RUN python -c "\
import sys; sys.path.insert(0, '.'); \
from rag.indexer import KnowledgeBaseIndexer; \
indexer = KnowledgeBaseIndexer(); \
indexer.build_and_save('rag/data/pdfs', 'rag/data/faiss_index')" \
    || echo 'Index build skipped — will build at runtime'

# Expose Streamlit port (HF Spaces default)
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
