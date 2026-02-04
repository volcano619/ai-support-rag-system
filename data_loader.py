"""
Data Loader Module for RAG System

This module handles loading and preprocessing the KIS Q&A dataset.
Each function is documented with its purpose and reasoning.

WHY THIS MODULE EXISTS:
- Centralizes all data loading logic
- Provides clean, validated data to other components
- Handles edge cases (missing values, malformed data)
- Creates structured objects that are easy to work with

Author: RAG System
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import logging

from config import (
    DATASET_PATH,
    COL_TOPIC,
    COL_TEXT,
    COL_QUESTION,
    COL_GROUND_TRUTH
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeItem:
    """
    Represents a single knowledge base article
    
    WHY THIS CLASS:
    - Provides a clean, typed interface for knowledge items
    - Easier to work with than raw dictionaries
    - Can add methods for processing/validation
    """
    
    def __init__(self, topic: str, text: str, metadata: Dict = None):
        self.topic = topic
        self.text = text
        self.metadata = metadata or {}
        
        # Generate a unique ID based on topic
        # WHY: Needed for tracking which document was retrieved
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID from topic"""
        # Simple slug generation: lowercase, replace spaces with underscores
        return self.topic.lower().replace(" ", "_")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "topic": self.topic,
            "text": self.text,
            "metadata": self.metadata
        }
    
    def __repr__(self):
        return f"KnowledgeItem(id='{self.id}', topic='{self.topic[:50]}...')"


class QAPair:
    """
    Represents a question-answer pair for evaluation
    
    WHY THIS CLASS:
    - Keeps test data organized
    - Links questions to their expected answers
    - Tracks which KI article should be retrieved
    """
    
    def __init__(self, question: str, ground_truth: str, expected_ki_id: str):
        self.question = question
        self.ground_truth = ground_truth
        self.expected_ki_id = expected_ki_id
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "expected_ki_id": self.expected_ki_id
        }
    
    def __repr__(self):
        return f"QAPair(question='{self.question[:50]}...', expected_ki='{self.expected_ki_id}')"


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    WHY MINIMAL CLEANING:
    - The dataset is already high-quality corporate documentation
    - Over-aggressive cleaning can remove important information
    - Embeddings models handle varied text well
    
    Args:
        text: Raw text string
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    # WHY: Multiple spaces/newlines don't add semantic value
    text = " ".join(text.split())
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # That's it! We want to preserve:
    # - Punctuation (important for step-by-step instructions)
    # - Numbers (version numbers, step numbers)
    # - Special characters (e.g., file paths, commands)
    
    return text


def load_knowledge_base() -> List[KnowledgeItem]:
    """
    Load all knowledge base articles from CSV with CHUNKING
    
    WHY CHUNKING:
    - Small models (Flan-T5) have limited context (512 tokens)
    - Original documents are ~1300 tokens, causing truncation
    - We split docs into ~300 token chunks so the model sees the answer
    """
    logger.info(f"Loading knowledge base from {DATASET_PATH}")
    
    # Check file exists
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    # Load CSV
    df = pd.read_csv(DATASET_PATH)
    
    knowledge_items = []
    
    # Configure chunking
    # 600 chars ~= 150 tokens. Allows retrieving 2-3 chunks.
    CHUNK_SIZE = 600  
    CHUNK_OVERLAP = 100
    
    # Group by topic
    unique_kis = df.groupby(COL_TOPIC).first().reset_index()
    
    for idx, row in unique_kis.iterrows():
        topic = str(row[COL_TOPIC])
        text = clean_text(str(row[COL_TEXT]))
        
        # Skip empty
        if not topic or not text:
            continue
            
        # Create chunks
        chunks = []
        if len(text) > CHUNK_SIZE:
            start = 0
            while start < len(text):
                end = min(start + CHUNK_SIZE, len(text))
                
                # Attempt to split on space to avoid cutting words
                if end < len(text):
                    last_space = text.rfind(' ', start, end)
                    if last_space != -1 and last_space > start + (CHUNK_SIZE // 2):
                        end = last_space
                
                chunk_text = text[start:end]
                chunks.append(chunk_text)
                
                # Move start forward by stride (size - overlap)
                start += (CHUNK_SIZE - CHUNK_OVERLAP)
        else:
            chunks = [text]
            
        # Create KnowledgeItem for each chunk
        for i, chunk in enumerate(chunks):
            # Create ID that links back to original topic but is unique for chunk
            # e.g., "email_setup_chunk_0"
            chunk_id = f"{topic.lower().replace(' ', '_')}_{i}"
            
            # Add context to text so model knows what this chunk is about
            # WHY: Isolated chunks might lose context (e.g., "Step 5: Click OK")
            # Adding title helps: "Email Setup (Part 1): Step 5: Click OK"
            chunk_text_with_context = f"{topic} (Part {i+1}):\n{chunk}"
            
            ki = KnowledgeItem(
                topic=topic,
                text=chunk_text_with_context, 
                metadata={
                    "source_row": int(idx),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "is_chunk": True,
                    "original_id": topic.lower().replace(" ", "_")
                }
            )
            # Override ID manually to ensure uniqueness
            ki.id = chunk_id
            
            knowledge_items.append(ki)
    
    logger.info(f"Loaded {len(knowledge_items)} chunks from {len(unique_kis)} original documents")
    return knowledge_items


def load_qa_pairs() -> List[QAPair]:
    """
    Load question-answer pairs for evaluation
    
    WHY THIS FUNCTION:
    - Creates test set for accuracy measurement
    - Links each question to its expected KI article
    
    Returns:
        List of QAPair objects
    
    Raises:
        FileNotFoundError: If dataset doesn't exist
        ValueError: If dataset is malformed
    """
    logger.info(f"Loading Q&A pairs from {DATASET_PATH}")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    
    # Validate required columns
    required_cols = [COL_TOPIC, COL_QUESTION, COL_GROUND_TRUTH]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    qa_pairs = []
    
    for idx, row in df.iterrows():
        topic = row[COL_TOPIC]
        question = row[COL_QUESTION]
        ground_truth = row[COL_GROUND_TRUTH]
        
        # Clean text
        clean_question = clean_text(str(question))
        clean_ground_truth = clean_text(str(ground_truth))
        
        # Skip if any field is empty
        if not clean_question or not clean_ground_truth:
            logger.warning(f"Skipping incomplete Q&A pair at row {idx}")
            continue
        
        # Generate expected KI ID from topic
        # WHY: This links the question to which document should be retrieved
        expected_ki_id = topic.lower().replace(" ", "_")
        
        qa_pair = QAPair(
            question=clean_question,
            ground_truth=clean_ground_truth,
            expected_ki_id=expected_ki_id
        )
        
        qa_pairs.append(qa_pair)
    
    logger.info(f"Loaded {len(qa_pairs)} Q&A pairs for evaluation")
    
    return qa_pairs


def load_all_data() -> Tuple[List[KnowledgeItem], List[QAPair]]:
    """
    Load both knowledge base and Q&A pairs
    
    WHY THIS CONVENIENCE FUNCTION:
    - Single function to load everything needed
    - Ensures consistent loading
    
    Returns:
        Tuple of (knowledge_items, qa_pairs)
    """
    knowledge_items = load_knowledge_base()
    qa_pairs = load_qa_pairs()
    
    logger.info(f"Data loading complete: {len(knowledge_items)} KIs, {len(qa_pairs)} Q&A pairs")
    
    return knowledge_items, qa_pairs


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the data loader
    
    WHY: Validates that data loading works correctly before using in pipeline
    """
    print("=" * 80)
    print("TESTING DATA LOADER")
    print("=" * 80)
    
    try:
        # Load data
        knowledge_items, qa_pairs = load_all_data()
        
        # Display sample knowledge item
        print("\n📚 SAMPLE KNOWLEDGE ITEM:")
        print("-" * 80)
        sample_ki = knowledge_items[0]
        print(f"ID: {sample_ki.id}")
        print(f"Topic: {sample_ki.topic}")
        print(f"Text (first 200 chars): {sample_ki.text[:200]}...")
        print(f"Metadata: {sample_ki.metadata}")
        
        # Display sample Q&A pair
        print("\n❓ SAMPLE Q&A PAIR:")
        print("-" * 80)
        sample_qa = qa_pairs[0]
        print(f"Question: {sample_qa.question}")
        print(f"Expected KI: {sample_qa.expected_ki_id}")
        print(f"Ground Truth (first 200 chars): {sample_qa.ground_truth[:200]}...")
        
        # Statistics
        print("\n📊 DATASET STATISTICS:")
        print("-" * 80)
        print(f"Total Knowledge Items: {len(knowledge_items)}")
        print(f"Total Q&A Pairs: {len(qa_pairs)}")
        print(f"Average text length: {sum(ki.metadata['character_count'] for ki in knowledge_items) / len(knowledge_items):.0f} chars")
        print(f"Average question length: {sum(len(qa.question) for qa in qa_pairs) / len(qa_pairs):.0f} chars")
        
        # Verify all expected KI IDs exist in knowledge base
        ki_ids = {ki.id for ki in knowledge_items}
        missing_kis = [qa.expected_ki_id for qa in qa_pairs if qa.expected_ki_id not in ki_ids]
        
        if missing_kis:
            print(f"\n⚠️  WARNING: {len(missing_kis)} Q&A pairs reference missing KIs:")
            for ki_id in set(missing_kis):
                print(f"  - {ki_id}")
        else:
            print("\n✅ All Q&A pairs have corresponding knowledge items")
        
        print("\n✅ DATA LOADER TEST PASSED")
        
    except Exception as e:
        print(f"\n❌ DATA LOADER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
