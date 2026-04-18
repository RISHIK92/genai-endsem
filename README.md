# 🎓 EduAgent-OS — The Autonomous Assessment Optimizer

> An agentic AI system that acts as a **Senior Examiner**, analyzing MCQ questions with ML models, retrieving pedagogical best practices, and autonomously refining questions across difficulty tiers.

## 🏗️ Architecture

```
User Question → [Analyst] → [Pedagogy Retriever] → [Reasoner] → [Refiner]
                  XGBoost       FAISS RAG              LLM         LLM + XGBoost
                  Prediction    Context Retrieval       Analysis    Validation Loop
```

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **ML Models** | XGBoost (.json) | Difficulty/discrimination prediction |
| **Feature Extraction** | Custom pipeline | 25-30 question features |
| **Knowledge Base** | FAISS + all-MiniLM-L6-v2 | Pedagogical research retrieval |
| **Agent Framework** | LangGraph | 4-node state machine with conditional routing |
| **LLM** | Groq (Llama 3.1 70B) | Reasoning and question generation |
| **UI** | Streamlit | Chat interface + analytics dashboard |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [Groq API Key](https://console.groq.com) (free tier)

### Local Setup

```bash
# Clone and install
cd gen-ai-endsem
pip install -r requirements.txt

# Set API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# Run
streamlit run app.py
```

### Docker

```bash
docker build -t eduagent-os .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key eduagent-os
```

## 🧪 Features

### 1. Chat Interface
Natural language interaction with the assessment optimizer:
- *"Analyze this question and suggest improvements"*
- *"My students failed Q5. Explain why and fix it."*
- *"Make this question test Critical Thinking"*

### 2. ML Analysis Dashboard
- Difficulty/discrimination gauges
- SHAP-style feature importance charts
- Quality audit warnings
- Bloom's Taxonomy level detection

### 3. Multi-Tier Refinement
Generates Easy/Medium/Hard versions with XGBoost validation to ensure difficulty alignment.

### 4. Multi-Agent Debate
Three AI agents collaborate:
- **Professor**: Designs questions using pedagogy
- **ML Predictor**: Validates with XGBoost
- **Student Persona**: Simulates misconceptions

### 5. RAG Knowledge Base
FAISS-indexed pedagogical research covering:
- Bloom's Taxonomy
- Item Response Theory (IRT)
- MCQ writing best practices
- Assessment design principles

## 📁 Project Structure

```
├── app.py                    # Streamlit entry point
├── config/settings.py        # Centralized configuration
├── ml/                       # ML pipeline
│   ├── feature_extractor.py  # 25/30 feature extraction
│   ├── predictor.py          # XGBoost inference + SHAP
│   └── models/               # Pre-trained models
├── rag/                      # RAG knowledge base
│   ├── indexer.py            # PDF → FAISS index
│   └── retriever.py          # Query → relevant context
├── agent/                    # LangGraph agent
│   ├── state.py              # State schema
│   ├── graph.py              # State machine assembly
│   ├── nodes/                # 4 pipeline nodes
│   └── tools/                # XGBoost as agent tools
├── debate/                   # Multi-agent debate
│   └── debate_runner.py      # 3-agent orchestration
├── ui/                       # Streamlit components
│   ├── analysis_dashboard.py
│   ├── question_editor.py
│   └── report_generator.py
├── Dockerfile                # HF Spaces deployment
└── requirements.txt
```

## 🌐 Deployment (Hugging Face Spaces)

1. Create a new HF Space → Select **Docker** SDK
2. Set `GROQ_API_KEY` in Space secrets
3. Push this repository to the Space
4. Monitor build logs

## 📊 Mid-Sem → End-Sem Integration

| Mid-Sem Asset | End-Sem Agentic Upgrade |
|---------------|------------------------|
| XGBoost Classifier | Used as a **Validator** — agent generates questions and validates difficulty |
| Feature Extractor | Used to **Audit** LLM outputs — flags issues for agent correction |
| Streamlit App | Enhanced with **Chat Interface** and **Multi-Agent Debate** |

## 📄 License

MIT
