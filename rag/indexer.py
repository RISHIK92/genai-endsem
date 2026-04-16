"""
RAG Knowledge Base Indexer.

Processes PDF documents on assessment design, Bloom's Taxonomy,
Item Response Theory, etc. into a FAISS vector index.
"""
import json
import numpy as np
import faiss
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
from config.settings import EMBEDDING_MODEL, EMBEDDING_DIMENSION, CHUNK_SIZE, CHUNK_OVERLAP


class KnowledgeBaseIndexer:
    """Builds and persists the FAISS index from PDF documents or text files."""

    def __init__(self, embedding_model: str = EMBEDDING_MODEL):
        """Initialize with embedding model."""
        self.embedder = SentenceTransformer(embedding_model)
        self.chunk_size = CHUNK_SIZE
        self.chunk_overlap = CHUNK_OVERLAP

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            return text
        except Exception as e:
            print(f"  ⚠️ Error reading {pdf_path}: {e}")
            return ""

    def _extract_text_from_txt(self, txt_path: str) -> str:
        """Extract text from a plain text file."""
        with open(txt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _chunk_text(self, text: str, source: str) -> list:
        """Split text into overlapping chunks with metadata."""
        chunks = []
        
        if len(text) <= self.chunk_size:
            chunks.append({
                "text": text.strip(),
                "source": source,
                "chunk_id": 0
            })
            return chunks
        
        start = 0
        chunk_id = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                if break_point > self.chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            if chunk.strip():
                chunks.append({
                    "text": chunk.strip(),
                    "source": source,
                    "chunk_id": chunk_id
                })
                chunk_id += 1
            
            start = end - self.chunk_overlap
        
        return chunks

    def process_directory(self, data_dir: str) -> list:
        """Process all PDFs and text files in a directory.
        
        Returns:
            List of chunk dicts with 'text', 'source', 'chunk_id' keys.
        """
        data_path = Path(data_dir)
        all_chunks = []
        
        # Process PDFs
        for pdf_file in sorted(data_path.glob("*.pdf")):
            print(f"  📄 Processing: {pdf_file.name}")
            text = self._extract_text_from_pdf(str(pdf_file))
            if text.strip():
                chunks = self._chunk_text(text, pdf_file.name)
                all_chunks.extend(chunks)
                print(f"     → {len(chunks)} chunks extracted")
        
        # Process text files
        for txt_file in sorted(data_path.glob("*.txt")):
            print(f"  📝 Processing: {txt_file.name}")
            text = self._extract_text_from_txt(str(txt_file))
            if text.strip():
                chunks = self._chunk_text(text, txt_file.name)
                all_chunks.extend(chunks)
                print(f"     → {len(chunks)} chunks extracted")
        
        # Process markdown files
        for md_file in sorted(data_path.glob("*.md")):
            print(f"  📝 Processing: {md_file.name}")
            text = self._extract_text_from_txt(str(md_file))
            if text.strip():
                chunks = self._chunk_text(text, md_file.name)
                all_chunks.extend(chunks)
                print(f"     → {len(chunks)} chunks extracted")
        
        print(f"\n✅ Total: {len(all_chunks)} chunks from {data_path}")
        return all_chunks

    def build_index(self, chunks: list) -> tuple:
        """Build FAISS index from text chunks.
        
        Returns:
            (faiss.Index, list[dict] metadata)
        """
        if not chunks:
            raise ValueError("No chunks provided to index")
        
        texts = [c["text"] for c in chunks]
        print(f"  🧮 Encoding {len(texts)} chunks...")
        
        embeddings = self.embedder.encode(
            texts, 
            show_progress_bar=True,
            normalize_embeddings=True  # For cosine similarity via inner product
        )
        embeddings = np.array(embeddings, dtype='float32')
        
        # Build FAISS index (Inner Product for cosine similarity on normalized vectors)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        print(f"  ✅ FAISS index built: {index.ntotal} vectors, dim={dimension}")
        
        # Build metadata list (parallel to index vectors)
        metadata = [
            {"text": c["text"], "source": c["source"], "chunk_id": c["chunk_id"]}
            for c in chunks
        ]
        
        return index, metadata

    def save(self, index: faiss.Index, metadata: list, output_dir: str):
        """Persist FAISS index and metadata to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, str(output_path / "index.faiss"))
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  💾 Saved index ({index.ntotal} vectors) and metadata to {output_dir}")

    def build_and_save(self, data_dir: str, output_dir: str):
        """One-shot: process PDFs → build index → save to disk."""
        print("🔨 Building knowledge base index...")
        chunks = self.process_directory(data_dir)
        
        if not chunks:
            print("⚠️ No documents found to index. Creating empty placeholder.")
            # Create an empty index with a placeholder
            self._create_placeholder_index(output_dir)
            return
        
        index, metadata = self.build_index(chunks)
        self.save(index, metadata, output_dir)
        print("✅ Knowledge base ready!")

    def _create_placeholder_index(self, output_dir: str):
        """Create a minimal index with built-in pedagogical knowledge."""
        # Built-in knowledge chunks (no PDFs needed)
        builtin_chunks = self._get_builtin_knowledge()
        
        if builtin_chunks:
            index, metadata = self.build_index(builtin_chunks)
            self.save(index, metadata, output_dir)
        else:
            # Truly empty fallback
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
            faiss.write_index(index, str(output_path / "index.faiss"))
            with open(output_path / "metadata.json", 'w') as f:
                json.dump([], f)

    def _get_builtin_knowledge(self) -> list:
        """Return built-in pedagogical knowledge chunks."""
        knowledge = [
            # Bloom's Taxonomy
            {
                "text": "Bloom's Taxonomy is a hierarchical classification of cognitive skills used in education. The six levels from lowest to highest are: Remember (recall facts), Understand (explain ideas), Apply (use information in new situations), Analyze (break information into parts), Evaluate (justify decisions), and Create (produce new work). Each level builds upon the previous ones.",
                "source": "builtin_blooms_taxonomy",
                "chunk_id": 0
            },
            {
                "text": "Remember level questions ask students to recall or recognize previously learned information. Action verbs include: define, list, state, identify, name, recall, recognize, label, match. Example: 'What is the chemical formula for water?' These questions test the lowest cognitive level and are appropriate for introductory assessments.",
                "source": "builtin_blooms_taxonomy",
                "chunk_id": 1
            },
            {
                "text": "Understand level questions require students to demonstrate comprehension by explaining ideas or concepts. Action verbs include: explain, describe, summarize, interpret, classify, compare, discuss, distinguish, predict. Example: 'Explain the difference between mitosis and meiosis.' These questions go beyond simple recall.",
                "source": "builtin_blooms_taxonomy",
                "chunk_id": 2
            },
            {
                "text": "Apply level questions ask students to use learned information in new situations. Action verbs include: apply, solve, use, demonstrate, calculate, compute, implement, execute, determine. Example: 'Calculate the velocity of an object given force and mass.' These questions require students to transfer knowledge to practical scenarios.",
                "source": "builtin_blooms_taxonomy",
                "chunk_id": 3
            },
            {
                "text": "Analyze level questions require students to break information into component parts and examine relationships. Action verbs include: analyze, examine, differentiate, organize, deconstruct, attribute, investigate, contrast. Example: 'Compare and contrast the economic policies of two different countries.' These questions develop critical thinking skills.",
                "source": "builtin_blooms_taxonomy",
                "chunk_id": 4
            },
            {
                "text": "Evaluate level questions ask students to make judgments based on criteria. Action verbs include: evaluate, justify, assess, critique, judge, argue, defend, support, recommend. Example: 'Evaluate the effectiveness of this experimental design.' These questions require students to use standards and criteria for assessment.",
                "source": "builtin_blooms_taxonomy",
                "chunk_id": 5
            },
            {
                "text": "Create level questions require students to produce original work or combine elements in new ways. Action verbs include: create, design, construct, develop, formulate, propose, devise, compose, invent. Example: 'Design an experiment to test the hypothesis that...' This is the highest cognitive level in Bloom's Taxonomy.",
                "source": "builtin_blooms_taxonomy",
                "chunk_id": 6
            },

            # MCQ Writing Best Practices
            {
                "text": "Effective MCQ stem writing guidelines: 1) State the question clearly and concisely. 2) Include only relevant information. 3) Use positive phrasing; avoid negatives like 'NOT' and 'EXCEPT' when possible. 4) Ensure the stem can stand alone as a question. 5) Place most of the content in the stem rather than in the options. 6) Avoid clues that might help test-wise students.",
                "source": "builtin_mcq_guide",
                "chunk_id": 0
            },
            {
                "text": "Distractor design principles: Good distractors should be plausible to students who haven't mastered the material. They should: 1) Be clearly wrong but attractive to students with common misconceptions. 2) Be similar in length and grammatical structure to the correct answer. 3) Represent common errors or misconceptions. 4) Not include 'All of the above' or 'None of the above' options. 5) Be mutually exclusive and homogeneous in content.",
                "source": "builtin_mcq_guide",
                "chunk_id": 1
            },
            {
                "text": "To improve question discrimination: 1) Ensure distractors target specific misconceptions. 2) Avoid making the correct answer notably longer or more detailed than distractors. 3) Use plausible numerical distractors that reflect common calculation errors. 4) Avoid absolute terms like 'always' or 'never' in distractors. 5) Test at appropriate Bloom's level for the learning objective. 6) Avoid trivially obvious incorrect options.",
                "source": "builtin_mcq_guide",
                "chunk_id": 2
            },
            {
                "text": "Common MCQ writing mistakes to avoid: 1) Making the correct answer consistently longer than distractors. 2) Using 'All of the above' or 'None of the above' as options. 3) Including grammatical clues that eliminate options. 4) Using negatively worded stems without emphasis. 5) Creating options that overlap in meaning. 6) Testing trivial or irrelevant knowledge. 7) Using absolute terms that make options obviously wrong.",
                "source": "builtin_mcq_guide",
                "chunk_id": 3
            },

            # Item Response Theory
            {
                "text": "Item Response Theory (IRT) provides a framework for analyzing and developing assessments. The key parameters are: Difficulty (b-parameter) represents the ability level at which 50% of students answer correctly. Discrimination (a-parameter) indicates how well the item differentiates between high and low ability students. Guessing (c-parameter) represents the probability of answering correctly by chance.",
                "source": "builtin_irt",
                "chunk_id": 0
            },
            {
                "text": "Difficulty Index (P-value) interpretation: P > 0.75 indicates an easy item that most students answer correctly. P between 0.30-0.75 indicates moderate difficulty, which is ideal for most assessments. P < 0.30 indicates a difficult item that few students answer correctly. Items with extreme P-values (very high or very low) generally have poor discrimination.",
                "source": "builtin_irt",
                "chunk_id": 1
            },
            {
                "text": "Discrimination Index interpretation: Values above 0.40 indicate excellent discrimination—the item clearly separates students who know the material from those who don't. Values between 0.30-0.39 indicate good discrimination. Values between 0.20-0.29 indicate fair discrimination and may need improvement. Values below 0.20 indicate poor discrimination—the item should be revised or removed.",
                "source": "builtin_irt",
                "chunk_id": 2
            },
            {
                "text": "To improve a question with poor discrimination: 1) Ensure the question aligns with the learning objectives. 2) Check that distractors target specific misconceptions. 3) Verify the correct answer is unambiguously correct. 4) Review the stem for clarity and remove extraneous information. 5) Adjust difficulty level—items that are too easy or too hard tend to have poor discrimination. 6) Consider increasing the cognitive level using Bloom's Taxonomy.",
                "source": "builtin_irt",
                "chunk_id": 3
            },

            # Assessment Design
            {
                "text": "Principles of effective assessment design: 1) Alignment: Ensure questions match learning objectives. 2) Validity: Questions should measure what they intend to measure. 3) Reliability: Consistent results across different administrations. 4) Fairness: Questions should not advantage or disadvantage any group. 5) Authenticity: Questions should relate to real-world applications when possible.",
                "source": "builtin_assessment_design",
                "chunk_id": 0
            },
            {
                "text": "Reducing LaTeX complexity in mathematical questions: 1) Use simple notation where possible. 2) Break complex expressions into smaller parts with intermediate steps. 3) Use words instead of symbols when they convey meaning more clearly. 4) Provide a figure or diagram alongside complex formulas. 5) Ensure all symbols are defined in the stem. High LaTeX density can confuse students who struggle with notation rather than concepts.",
                "source": "builtin_assessment_design",
                "chunk_id": 1
            },
            {
                "text": "Converting questions between Bloom's Taxonomy levels: To increase cognitive level from 'Remember' to 'Apply', change 'What is X?' to 'Given this scenario, use X to solve...'. To move from 'Apply' to 'Analyze', change 'Calculate X' to 'Given these results, determine why X differs from expected'. To reach 'Evaluate', ask students to judge the appropriateness of a given solution or method.",
                "source": "builtin_assessment_design",
                "chunk_id": 2
            },
            {
                "text": "Testing for critical thinking with MCQs: 1) Present novel scenarios that require applying known concepts. 2) Include data interpretation (graphs, tables). 3) Ask students to identify assumptions or limitations. 4) Present multiple valid approaches and ask which is most appropriate. 5) Use case studies or real-world contexts. 6) Require multi-step reasoning where each step builds on the previous.",
                "source": "builtin_assessment_design",
                "chunk_id": 3
            },

            # Student Misconceptions
            {
                "text": "Common student misconceptions in STEM: In physics, students often confuse velocity with acceleration, or believe heavier objects fall faster. In chemistry, students may think atoms are visible under a microscope. In mathematics, common errors include distributing exponents over addition, confusing correlation with causation, and misapplying the chain rule.",
                "source": "builtin_misconceptions",
                "chunk_id": 0
            },
            {
                "text": "Designing distractors based on misconceptions: Each distractor should represent a specific, documented misconception that students commonly hold. For mathematical questions, include answers that result from common procedural errors (e.g., forgetting to distribute a negative sign, incorrect order of operations). For conceptual questions, include options that represent commonly held but incorrect beliefs about the topic.",
                "source": "builtin_misconceptions",
                "chunk_id": 1
            },
        ]
        return knowledge
