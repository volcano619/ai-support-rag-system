"""
Configuration Module for RAG System

This module centralizes all configuration parameters for the RAG system.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
RESULTS_DIR = DATA_DIR / "results"

for dir_path in [DATA_DIR, FAISS_INDEX_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

DATASET_PATH = PROJECT_ROOT / "rag_sample_qas_from_kis.csv"

# ============================================================================
# EMBEDDING MODEL CONFIGURATION
# ============================================================================
# Small, efficient model (384 dims, ~80MB)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ============================================================================
# RETRIEVAL CONFIGURATION
# ============================================================================
RETRIEVAL_TOP_K = 5
SIMILARITY_THRESHOLD = 0.3

# ============================================================================
# RERANKING CONFIGURATION
# ============================================================================
# Small cross-encoder (~80MB)
# Small cross-encoder (~80MB)
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 2  # Increased to 2 (with smaller chunks) to capture more context

# ============================================================================
# LLM CONFIGURATION (LOCAL FLAN-T5)
# ============================================================================
# Changed to local model for CPU compatibility
USE_LOCAL_LLM = True
LLM_MODEL_NAME = "google/flan-t5-base"  # ~250MB size, perfectly fits in 1.4GB RAM

# Generation parameters
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 512

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
RETRIEVAL_RECALL_TARGET = 0.95
RESPONSE_SIMILARITY_TARGET = 0.90
ROUGE_L_TARGET = 0.70

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================
APP_TITLE = "🤖 RAG-Based IT Support (Local)"
APP_LAYOUT = "wide"
DEBUG_MODE = True

# ============================================================================
# DATA PROCESSING CONFIGURATION
# ============================================================================
COL_TOPIC = "ki_topic"
COL_TEXT = "ki_text"
COL_QUESTION = "sample_question"
COL_GROUND_TRUTH = "sample_ground_truth"
CHUNKING_ENABLED = False
CHUNK_OVERLAP = 50

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    if not DATASET_PATH.exists():
        raise ValueError(f"Dataset not found: {DATASET_PATH}")
    return True
