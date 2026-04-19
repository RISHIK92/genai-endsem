# EduAgent-OS — System Architecture

## Overview

**EduAgent-OS** is an autonomous MCQ assessment optimizer that combines **XGBoost ML models**, **FAISS-backed RAG**, **LLM multi-agent reasoning** (via Groq/Llama 3.1 70B), and a **multi-agent debate system** to analyze, critique, and refine education assessment questions. The application is built with **Streamlit** and orchestrated by **LangGraph**.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    🖥️  STREAMLIT FRONTEND (app.py)                  │
│   ┌──────────────┐   ┌───────────────────┐   ┌──────────────────┐  │
│   │ 💬 Chat      │   │ 📊 Direct Analysis│   │ 🎭 Agent Debate  │  │
│   └──────┬───────┘   └────────┬──────────┘   └────────┬─────────┘  │
└──────────┼────────────────────┼────────────────────────┼────────────┘
           │                    │                        │
           ▼                    ▼                        ▼
┌──────────────────────┐  ┌──────────┐   ┌──────────────────────────┐
│ 🔄 LangGraph Pipeline│  │ XGBoost  │   │ 🎭 Multi-Agent Debate    │
│                      │  │ (Direct) │   │  👨‍🏫 Professor            │
│  Analyst ──────────┐ │  └──────────┘   │  🤖 ML Predictor         │
│  Pedagogy Retriever│ │                 │  🧑‍🎓 Student              │
│  Reasoner ─────────┤ │                 └──────────────────────────┘
│  Refiner ──────────┘ │                        │
│       ↑    (loop)    │                        ▼
└──────────────────────┘                 ┌──────────────┐
           │                             │ ⚡ Groq API   │
           ▼                             │ Llama 3.1 70B│
┌──────────────────────────────────┐     └──────────────┘
│          Supporting Systems       │
│  ┌────────┐ ┌──────┐ ┌────────┐  │
│  │XGBoost │ │ FAISS│ │Semantic│  │
│  │Models  │ │ RAG  │ │Scorer  │  │
│  └────────┘ └──────┘ └────────┘  │
└──────────────────────────────────┘
```

---

## Core Components

### 1. Streamlit Frontend — `app.py`

The entry point and UI layer. Organizes the app into **three tabs**:

| Tab                    | Purpose                                                               | Backend Path                             |
| ---------------------- | --------------------------------------------------------------------- | ---------------------------------------- |
| **💬 Chat**            | Natural language interaction; paste a question + ask for improvements | Full LangGraph pipeline                  |
| **📊 Direct Analysis** | Instant XGBoost analysis (no LLM required) OR full pipeline           | `FeatureExtractor` → `QuestionPredictor` |
| **🎭 Agent Debate**    | Multi-agent debate to generate optimal questions from scratch         | `DebateRunner`                           |

**Sidebar** provides:

- Groq API key input
- Target difficulty & Bloom's level selectors
- XGBoost model choice (Model A: 25 text features vs Model B: 30 all features)
- Quick actions (rebuild knowledge base, clear conversation)
- System stack info display

---

### 2. LangGraph Pipeline — `agent/graph.py`

The core reasoning engine. A **4-node state machine** with conditional looping:

```
START → Analyst → Pedagogy Retriever → Reasoner → Refiner → [conditional]
                                          ↑                       ↓
                                          └───── (if mismatch) ───┘
                                               (else) → END
```

**Conditional Routing:**

- If the Refiner detects a difficulty mismatch AND iterations < 3 AND the user specified a target → loop back to Reasoner with a critique
- Otherwise → END

#### Shared State — `agent/state.py`

All nodes read/write from a single `AgentState` (TypedDict):

| State Group          | Fields                                                                               | Written By                 |
| -------------------- | ------------------------------------------------------------------------------------ | -------------------------- |
| **Input**            | `raw_question`, `user_request`, `target_bloom_level`, `target_difficulty`, `subject` | User / app.py              |
| **Analyst Output**   | `ml_stats`, `feature_audit`, `current_bloom_level`, `semantic_scores`                | Node 1: Analyst            |
| **Retriever Output** | `pedagogy_context`, `retrieval_sources`, `rag_query`                                 | Node 2: Pedagogy Retriever |
| **Reasoner Output**  | `reasoning`, `improvement_strategy`, `thinking`                                      | Node 3: Reasoner           |
| **Refiner Output**   | `refined_questions`, `difficulty_justification`, `validation_results`                | Node 4: Refiner            |
| **Control Flow**     | `critique`, `iteration_count`, `should_continue`, `error`                            | Node 4: Refiner            |
| **Messages**         | `messages` (append-only via `Annotated[list, add]`)                                  | All nodes                  |

---

### 3. Pipeline Nodes (Detailed)

#### Node 1: Analyst — `agent/nodes/analyst.py`

**Purpose:** Quantitatively analyze the raw question using ML models.

```
Raw MCQ
  ├──→ Feature Extractor (25/30 features)
  │      ├──→ XGBoost Model A (text-only, 25 features)
  │      └──→ XGBoost Model B (all features, 30 features)   ← primary
  │
  ├──→ LLM Difficulty Rater (conceptual difficulty 1–10)
  ├──→ Student Simulator (confidence as inverse difficulty)
  │
  └──→ Score Blender
         XGBoost (50%) + LLM (30%) + Student (20%)
         → blended difficulty index + class
```

**What it does:**

1. Extracts 25 or 30 features (text length, word count, LaTeX density, Bloom's level, misconception patterns, option analysis, etc.)
2. Runs both XGBoost models — Classifier (3-class: Easy/Medium/Hard) + Regressor (continuous difficulty index)
3. Calls LLM for **semantic difficulty rating** (1–10 scale; captures cognitive depth that XGBoost's surface features miss)
4. Simulates a student attempting the question → reports confidence as inverse difficulty proxy
5. **Blends** all three signals into a unified difficulty estimate
6. Audits the question for common issues (long stems, high LaTeX density, low readability, distractor imbalance, negation)

**Outputs:** `ml_stats`, `feature_audit`, `current_bloom_level`, `semantic_scores`

---

#### Node 2: Pedagogy Retriever — `agent/nodes/pedagogy_retriever.py`

**Purpose:** Retrieve relevant educational research from the FAISS knowledge base.

```
ml_stats + feature_audit + user_request
  └──→ Query Builder
         → "How to improve a question with Hard difficulty and Poor
            discrimination, high LaTeX density, focusing on Bloom's
            taxonomy upgrade"
         └──→ FAISS Search (top-5 chunks, cosine similarity)
                └──→ Format Context (with source attribution)
                       → pedagogy_context (string for LLM prompt)
```

**What it does:**

1. Builds a targeted RAG query from ML stats and audit findings
2. Queries FAISS index for top-5 semantically similar chunks
3. Formats results with source attribution for downstream LLM consumption
4. Falls back to keyword-matched built-in knowledge if no index exists

**Outputs:** `pedagogy_context`, `retrieval_sources`, `rag_query`

---

#### Node 3: Reasoner — `agent/nodes/reasoner.py`

**Purpose:** LLM-powered expert analysis combining ML data + RAG context.

**System Role:** "Senior Examiner and Assessment Specialist" with expertise in Bloom's Taxonomy, IRT, MCQ design, and pedagogy.

**Prompt incorporates:**

- Raw question text
- Full ML analysis (difficulty, discrimination, class probabilities, feature importance)
- Audit warnings
- Retrieved pedagogical research context
- User request and target parameters
- Previous critique (if iterating)

**LLM produces:**

1. **Root Cause Analysis** — WHY the question has its predicted difficulty/discrimination
2. **Bloom's Taxonomy Assessment** — current level, alignment gaps with target
3. **Improvement Strategy** — concrete step-by-step plan
4. **Key Modifications** — 3–5 most impactful changes

**Chain-of-thought** (`<think>` blocks) are separated via `utils/think_parser.py` and shown in collapsible UI expanders.

**Outputs:** `reasoning`, `improvement_strategy`, `thinking`

---

#### Node 4: Refiner — `agent/nodes/refiner.py`

**Purpose:** Generate three refined versions (Easy/Medium/Hard) and validate them.

```
reasoning + improvement_strategy + raw_question
  └──→ LLM generates 3 versions
         ├── EASY — Bloom's: Remember/Understand
         ├── MEDIUM — Bloom's: Apply/Analyze
         └── HARD — Bloom's: Evaluate/Create

Each version ──→ XGBoost re-validates difficulty
                    ├── ✅ Match → Done
                    └── ❌ Mismatch → set should_continue = True
                                     + critique → loop to Reasoner
```

**What it does:**

1. Prompts LLM to generate three question versions at different cognitive levels
2. Parses the structured response into `{"easy": ..., "medium": ..., "hard": ...}`
3. **Re-validates each version** through XGBoost to ensure difficulty alignment
4. If mismatches exist and user specified a target → loops back to Reasoner with critique
5. Maximum **3 refinement iterations**

**Outputs:** `refined_questions`, `difficulty_justification`, `validation_results`, `should_continue`, `iteration_count`, `critique`

---

### 4. ML Layer

#### Feature Extractor — `ml/feature_extractor.py`

Extracts a comprehensive feature vector from raw MCQ text:

| Feature Category   | Count | Examples                                                                                                       |
| ------------------ | ----- | -------------------------------------------------------------------------------------------------------------- |
| Text Statistics    | 4     | `text_length`, `word_count`, `sentence_count`, `avg_word_length`                                               |
| LaTeX/Math         | 5     | `latex_density`, `has_latex`, `latex_command_count`, `math_operator_count`, `number_count`                     |
| Complexity         | 2     | `vocab_richness`, `text_complexity_score`                                                                      |
| Option Analysis    | 5     | `answer_a_length`–`answer_d_length`, `avg_answer_length`, `answer_length_variance`                             |
| Domain Terms       | 4     | `has_advanced_terms`, `has_algebra_terms`, `has_geometry_terms`, `has_stats_terms`                             |
| Misconceptions     | 2     | `num_misconceptions`, `has_misconception`                                                                      |
| Structure          | 2     | `subject_difficulty_tier`, `construct_frequency`                                                               |
| **Model B extras** | **5** | `avg_response_time_sec`, `std_response_time_sec`, `discrimination_index`, `point_biserial_corr`, `irt_a_param` |

**Audit Engine** checks for:

- Excessive stem length (> 200 chars)
- High LaTeX density (> 0.30)
- Low readability (Flesch-Kincaid < 30)
- Low Bloom's level (Remember/Understand)
- Distractor similarity issues (too similar or too dissimilar)
- Negation in stem
- Option length imbalance
- Image references without visuals

#### XGBoost Predictor — `ml/predictor.py`

Two model variants:

| Model       | Features              | Architecture                                     |
| ----------- | --------------------- | ------------------------------------------------ |
| **Model A** | 25 text-only features | Classifier (3-class) + Regressor (continuous DI) |
| **Model B** | 30 all features       | Classifier (3-class) + Regressor (continuous DI) |

**Outputs:**

- `difficulty`: Easy / Medium / Hard (classification)
- `difficulty_index`: 0.0–1.0 (regression)
- `difficulty_probabilities`: per-class probabilities
- `confidence`: max probability
- `feature_importance`: SHAP-based feature weights (normalized)
- `discrimination` / `discrimination_index`: estimated from parabolic model (best at DI ≈ 0.5)

**Fallback:** If model files are missing, uses **synthetic heuristic-based predictions** combining LaTeX density, word count, readability, negation, Bloom's level, and formula count.

#### Semantic Scorer — `ml/semantic_scorer.py`

Augments XGBoost with LLM-based understanding:

| Component                | What It Does                                         | Temperature |
| ------------------------ | ---------------------------------------------------- | ----------- |
| **LLM Difficulty Rater** | Rates conceptual difficulty 1–10 (normalized to 0–1) | 0.1         |
| **Student Simulator**    | Simulates average student; reports confidence 0–1    | 0.3         |
| **Score Blender**        | Weighted combination of all three signals            | N/A         |

**Blending Weights:**

```
All available:    50% XGBoost + 30% LLM + 20% Student
LLM only:         60% XGBoost + 40% LLM
Student only:     70% XGBoost + 30% Student
Neither:         100% XGBoost
```

**Key Insight:** The score blender solves the "short but hard" problem — a concise question about bijections can be conceptually harder than a verbose LaTeX question, but XGBoost's surface features would predict the opposite.

---

### 5. RAG Layer

#### Indexer — `rag/indexer.py`

**Ingestion Pipeline:**

```
PDF/TXT/MD files (rag/data/pdfs/)
  → Text extraction (PyPDF2 for PDFs)
  → Chunking (1000 chars, 200 overlap, sentence-boundary aware)
  → Embedding (Sentence Transformers: all-MiniLM-L6-v2, 384-dim)
  → FAISS IndexFlatIP (Inner Product on normalized vectors = cosine similarity)
  → Persist to disk (index.faiss + metadata.json)
```

**Built-in Knowledge Base:** If no documents exist, creates an index from **18 embedded pedagogical knowledge chunks**:

| Topic                      | Chunks | Coverage                                                                       |
| -------------------------- | ------ | ------------------------------------------------------------------------------ |
| Bloom's Taxonomy           | 7      | Overview + each of 6 levels with action verbs and examples                     |
| MCQ Writing Best Practices | 4      | Stem writing, distractor design, discrimination improvement, common mistakes   |
| Item Response Theory       | 4      | IRT parameters, difficulty index, discrimination index, improvement strategies |
| Assessment Design          | 3      | Design principles, LaTeX complexity, Bloom's level conversion                  |
| Student Misconceptions     | 2      | Common STEM misconceptions, designing distractors from misconceptions          |

#### Retriever — `rag/retriever.py`

- **Semantic Search:** Encodes query → FAISS top-K nearest neighbors (cosine similarity)
- **Query Builder:** Converts ML stats into retrieval-optimized queries with feature-specific vocabulary
- **Fallback:** Keyword-based responses for common topics (Bloom's, distractors, IRT, LaTeX, difficulty, critical thinking)
- **Formatting:** Generates source-attributed context strings for LLM prompts

---

### 6. Multi-Agent Debate System — `debate/debate_runner.py`

A **standalone system** (separate from the LangGraph pipeline) for generating questions **from scratch** through multi-agent collaboration.

#### The Three Agents

| Agent               | Role                    | LLM Temp            | Functionality                                                                                  |
| ------------------- | ----------------------- | ------------------- | ---------------------------------------------------------------------------------------------- |
| **👨‍🏫 Professor**    | Question designer       | 0.7–0.8             | Designs and refines MCQs targeting specific Bloom's level + difficulty                         |
| **🤖 ML Predictor** | Difficulty validator    | N/A (deterministic) | Runs XGBoost prediction, checks match against target, reports feature drivers and audit issues |
| **🧑‍🎓 Student**      | Misconception simulator | 0.9                 | Attempts the question with realistic reasoning, identifies confusing elements                  |

#### Debate Flow

```
Round 1:
  Professor → designs initial question
  ML Predictor → validates (match/mismatch + feedback)
  Student (average) → attempts question

Round 2 (if mismatch):
  Professor → refines based on Round 1 feedback
  ML Predictor → re-validates
  Student (struggling) → attempts question

Round 3 (if still mismatch):
  Professor → refines based on Round 2 feedback
  ML Predictor → re-validates
  Student (high-achieving) → attempts question

... up to max_rounds (configurable, default 3, max 5)
```

**Consensus:** Reached when the ML Predictor confirms the predicted difficulty matches the target difficulty.

**Student Profile Escalation:** Each round uses a different student persona:

- Round 1: Average student
- Round 2: Struggling student (attracted to familiar-looking incorrect options)
- Round 3+: High-achieving student (overthinks simple questions)

---

### 7. End-to-End Data Flow

```
USER INPUT
  │  Raw MCQ text + target difficulty + target Bloom's + subject
  │
  ▼
PHASE 1: ANALYSIS (Node 1 — Analyst)
  │  • Extract 25/30 features from question text
  │  • XGBoost Model A (text features) + Model B (all features)
  │  • LLM rates conceptual difficulty (1–10)
  │  • Student simulator reports confidence
  │  • Blend: XGBoost 50% + LLM 30% + Student 20%
  │  • Audit for structural issues
  │  OUTPUTS: ml_stats, feature_audit, current_bloom_level, semantic_scores
  │
  ▼
PHASE 2: RETRIEVAL (Node 2 — Pedagogy Retriever)
  │  • Build targeted query from ML stats + audit findings
  │  • FAISS semantic search → top-5 relevant chunks
  │  • Format with source attribution
  │  OUTPUTS: pedagogy_context, retrieval_sources, rag_query
  │
  ▼
PHASE 3: REASONING (Node 3 — Reasoner)
  │  • Mega-prompt: question + ML stats + RAG context + user request
  │  • Llama 3.1 70B (via Groq) generates expert analysis
  │  • Root cause analysis, Bloom's assessment, improvement strategy
  │  • <think> blocks separated for UI display
  │  OUTPUTS: reasoning, improvement_strategy, thinking
  │
  ▼
PHASE 4: REFINEMENT (Node 4 — Refiner)
  │  • LLM generates 3 versions (Easy / Medium / Hard)
  │  • Each version re-validated through XGBoost
  │  • If mismatch + iterations < 3 → LOOP BACK to Phase 3 with critique
  │  • Otherwise → DONE
  │  OUTPUTS: refined_questions, validation_results, difficulty_justification
  │
  ▼
OUTPUT
  • 📊 Analysis Dashboard (metrics, feature importance, radar charts)
  • ✏️ Question Editor (side-by-side original vs refined comparison)
  • 📥 Downloadable Report
```

---

### 8. Supporting Utilities

#### Think Parser — `utils/think_parser.py`

The Groq/Llama model sometimes emits chain-of-thought reasoning inside `<think>...</think>` tags. This module:

- Strips `<think>` tags from visible responses
- Returns thinking content separately for collapsible UI display
- Ensures clean output is passed to downstream agent nodes (prevents tag leakage)

#### UI Components — `ui/`

| File                    | Purpose                                                                    |
| ----------------------- | -------------------------------------------------------------------------- |
| `analysis_dashboard.py` | Renders metrics, feature importance charts, radar plots, difficulty gauges |
| `question_editor.py`    | Side-by-side original vs refined comparison, editable question fields      |
| `report_generator.py`   | Creates downloadable PDF/text reports of the analysis                      |

---

### 9. Tech Stack Summary

| Layer                   | Technology                               | Purpose                                          |
| ----------------------- | ---------------------------------------- | ------------------------------------------------ |
| **Frontend**            | Streamlit                                | Interactive UI with chat, dashboard, debate tabs |
| **Agent Orchestration** | LangGraph                                | State machine with conditional routing           |
| **ML Prediction**       | XGBoost (Classifier + Regressor)         | Difficulty classification and regression         |
| **Feature Engineering** | Custom Python (555 lines)                | 25–30 features from MCQ text                     |
| **RAG / Vector DB**     | FAISS (IndexFlatIP)                      | Pedagogical knowledge retrieval                  |
| **Embeddings**          | Sentence Transformers (all-MiniLM-L6-v2) | 384-dim document/query encoding                  |
| **LLM**                 | Groq (Llama 3.1 70B Versatile)           | Reasoning, question generation, semantic scoring |
| **LLM Framework**       | LangChain (langchain-groq)               | LLM invocation and message formatting            |
| **Config**              | python-dotenv + centralized settings.py  | Environment variables, model paths, thresholds   |
| **Deployment**          | Docker                                   | Containerized deployment                         |

---

### 10. Configuration — `config/settings.py`

Key configurable parameters:

| Parameter                   | Default            | Description                  |
| --------------------------- | ------------------ | ---------------------------- |
| `LLM_MODEL`                 | `qwen/qwen3-32b`   | Groq LLM model               |
| `EMBEDDING_MODEL`           | `all-MiniLM-L6-v2` | Sentence Transformer for RAG |
| `EMBEDDING_DIMENSION`       | 384                | Vector dimension             |
| `CHUNK_SIZE`                | 1000               | Characters per RAG chunk     |
| `CHUNK_OVERLAP`             | 200                | Overlap between chunks       |
| `RAG_TOP_K`                 | 5                  | Number of chunks to retrieve |
| `MAX_REFINEMENT_ITERATIONS` | 3                  | Max Refiner → Reasoner loops |
| `LLM_TEMPERATURE`           | 0.7                | Default LLM temperature      |
| `LLM_MAX_TOKENS`            | 2048               | Max tokens per LLM response  |
| `MAX_STEM_LENGTH`           | 200                | Audit threshold (chars)      |
| `MAX_LATEX_DENSITY`         | 0.30               | Audit threshold (ratio)      |
| `MIN_READABILITY_SCORE`     | 30                 | Flesch-Kincaid threshold     |

---

### 11. File Tree

```
gen-ai-endsem/
├── app.py                          # Streamlit entry point (824 lines)
├── ARCHITECTURE.md                 # This document
├── Dockerfile                      # Container deployment
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables (API keys)
│
├── config/
│   └── settings.py                 # Centralized configuration
│
├── agent/
│   ├── graph.py                    # LangGraph state machine assembly
│   ├── state.py                    # AgentState TypedDict
│   ├── nodes/
│   │   ├── analyst.py              # Node 1: XGBoost + semantic analysis
│   │   ├── pedagogy_retriever.py   # Node 2: FAISS RAG retrieval
│   │   ├── reasoner.py             # Node 3: LLM expert reasoning
│   │   └── refiner.py              # Node 4: Question generation + validation
│   └── tools/
│       ├── xgb_tool.py             # XGBoost tool wrapper
│       └── feature_audit_tool.py   # Feature audit tool wrapper
│
├── ml/
│   ├── feature_extractor.py        # 25/30 feature extraction (555 lines)
│   ├── predictor.py                # XGBoost prediction wrapper
│   ├── semantic_scorer.py          # LLM difficulty + student simulation + blender
│   └── models/
│       ├── xgb_all_model.pkl       # Pre-trained model (pickle)
│       └── xgb_text_model.pkl      # Pre-trained model (pickle)
│
├── rag/
│   ├── indexer.py                  # FAISS index builder (PDF → chunks → vectors)
│   ├── retriever.py                # Semantic search + query builder
│   └── data/
│       ├── pdfs/                   # Source PDFs for knowledge base
│       └── faiss_index/            # Persisted FAISS index + metadata
│
├── debate/
│   └── debate_runner.py            # Multi-agent debate system (3 agents)
│
├── ui/
│   ├── analysis_dashboard.py       # Metrics, charts, and visualizations
│   ├── question_editor.py          # Side-by-side question comparison
│   └── report_generator.py         # Downloadable report creation
│
├── utils/
│   └── think_parser.py             # <think> tag separation utility
│
└── model/
    ├── xgb_all_model.pkl           # Additional model files
    └── xgb_text_model.pkl
```
