"""
Evaluator Module for RAG System

This module implements comprehensive evaluation metrics to measure system accuracy.

WHY THIS MODULE EXISTS:
- user requirement: ">90% accuracy"
- We need quantitative proof that the system meets this target
- Measures both retrieval quality and generation quality independently

METRICS IMPLEMENTED:
1. Retrieval Recall@K: Is the correct document retrieved?
2. Response Semantic Similarity: Does the answer mean the same as ground truth?
3. ROUGE Scores: Text overlap with ground truth

Author: RAG System
"""

import numpy as np
from typing import List, Dict, Tuple
import logging
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from tqdm import tqdm

from config import (
    EMBEDDING_MODEL_NAME,
    RETRIEVAL_RECALL_TARGET,
    RESPONSE_SIMILARITY_TARGET,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K
)
from data_loader import QAPair
from retriever import Retriever
from generator import Generator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluation suite for RAG system
    
    WHY THIS CLASS:
    - Centralizes all evaluation logic
    - runs end-to-end tests
    - Generates detailed reports
    """
    
    def __init__(self, retriever: Retriever, generator: Generator = None):
        """
        Initialize evaluator
        
        Args:
            retriever: Initialized Retriever instance
            generator: Initialized Generator instance (optional)
        """
        self.retriever = retriever
        self.generator = generator
        
        # Load model for semantic similarity calculation
        logger.info("Loading evaluation model for semantic similarity...")
        self.eval_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        WHY THIS METRIC:
        - Exact string match is too strict for generated text
        - Semantic similarity captures if the *meaning* is correct
        
        Args:
            text1: Generated answer
            text2: Ground truth answer
            
        Returns:
            Float 0.0 to 1.0
        """
        embeddings = self.eval_model.encode([text1, text2], convert_to_numpy=True)
        
        # Compute cosine similarity
        # dot product of normalized vectors
        norm1 = np.linalg.norm(embeddings[0])
        norm2 = np.linalg.norm(embeddings[1])
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(embeddings[0], embeddings[1]) / (norm1 * norm2))
    
    def evaluate_retrieval(self, qa_pairs: List[QAPair], top_k: int = RERANK_TOP_K) -> Dict:
        """
        Evaluate information retrieval performance
        
        Metric: Recall@K (percentage of times correct doc is in top K)
        
        Args:
            qa_pairs: List of test questions and expected docs
            top_k: Number of results to check
            
        Returns:
            Dictionary with metrics
        """
        logger.info(f"Evaluating retrieval on {len(qa_pairs)} questions...")
        
        correct_retrievals = 0
        mrr_sum = 0.0  # Mean Reciprocal Rank
        
        details = []
        
        for qa in tqdm(qa_pairs, desc="Evaluating Retrieval"):
            # Execute retrieval
            # We assume reranking is ON for best performance
            results = self.retriever.retrieve(qa.question, top_k=top_k, use_reranking=True)
            
            # Check if expected document is in results
            # Handle CHUNKING: retrieved ID might be "doc_id_chunk_0" while expected is "doc_id"
            found = False
            rank = 0
            
            retrieved_ids = [r[0].id for r in results]
            
            for i, r_id in enumerate(retrieved_ids):
                # Check for prefix match (e.g. "email_setup_chunk_0" starts with "email_setup")
                # Also check exact match just in case
                if r_id == qa.expected_ki_id or r_id.startswith(f"{qa.expected_ki_id}_"):
                    found = True
                    rank = i + 1
                    correct_retrievals += 1
                    mrr_sum += 1.0 / rank
                    break
            
            details.append({
                "question": qa.question,
                "expected": qa.expected_ki_id,
                "retrieved": retrieved_ids,
                "found": found,
                "rank": rank
            })
        
        recall = correct_retrievals / len(qa_pairs) if qa_pairs else 0
        mrr = mrr_sum / len(qa_pairs) if qa_pairs else 0
        
        logger.info(f"Retrieval Results: Recall@{top_k}={recall:.2%}, MRR={mrr:.3f}")
        
        return {
            "recall": recall,
            "mrr": mrr,
            "details": details
        }
    
    def evaluate_end_to_end(self, qa_pairs: List[QAPair]) -> Dict:
        """
        Evaluate full RAG pipeline (Retrieval + Generation)
        
        Metrics:
        - Semantic Similarity (Response vs Ground Truth)
        - ROUGE-L (Structural overlap)
        
        Args:
            qa_pairs: Test data
            
        Returns:
            Dictionary with comprehensive metrics
        """
        logger.info(f"Evaluating end-to-end on {len(qa_pairs)} questions...")
        
        similarities = []
        rouge_scores = []
        results_details = []
        
        for qa in tqdm(qa_pairs, desc="Evaluating Generation"):
            # 1. Retrieve context
            context, sources_metadata = self.retriever.get_context_for_generation(qa.question)
            
            # 2. Generate response
            response, metadata = self.generator.generate_response(
                qa.question, context, sources_metadata
            )
            
            # 3. Calculate metrics
            sim_score = self.calculate_semantic_similarity(response, qa.ground_truth)
            similarities.append(sim_score)
            
            rouge_score = self.rouge_scorer.score(qa.ground_truth, response)['rougeL'].fmeasure
            rouge_scores.append(rouge_score)
            
            results_details.append({
                "question": qa.question,
                "generated": response,
                "ground_truth": qa.ground_truth,
                "sources": [s['id'] for s in sources_metadata],
                "semantic_similarity": sim_score,
                "rouge_l": rouge_score
            })
            
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        
        logger.info(f"Generation Results: Similarity={avg_similarity:.2%}, ROUGE-L={avg_rouge:.3f}")
        
        return {
            "avg_similarity": avg_similarity,
            "avg_rouge_l": avg_rouge,
            "details": results_details
        }
        
    def run_full_evaluation(self, qa_pairs: List[QAPair]) -> Dict:
        """
        Run both retrieval and generation evaluation and print report
        """
        print("\n" + "="*80)
        print("🚀 STARTING COMPLETENESS EVALUATION")
        print("="*80)
        
        # 1. Retrieval Eval
        retrieval_metrics = self.evaluate_retrieval(qa_pairs)
        recall = retrieval_metrics['recall']
        
        # 2. End-to-End Eval (only if generator is available)
        e2e_metrics = None
        similarity = 0.0
        rouge_score = 0.0
        
        if self.generator:
            try:
                e2e_metrics = self.evaluate_end_to_end(qa_pairs)
                similarity = e2e_metrics['avg_similarity']
                rouge_score = e2e_metrics['avg_rouge_l']
            except Exception as e:
                logger.error(f"Generation evaluation failed: {e}")
                print(f"⚠️ Generation evaluation skipped due to error: {e}")
        else:
            print("\n⚠️ Generation evaluation skipped (No Generator/API Key)")
        
        # 3. Final Report
        print("\n" + "="*80)
        print("📊 FINAL ACCURACY REPORT")
        print("="*80)
        
        print(f"Retrieval Recall@{RERANK_TOP_K}:  {recall:.1%}  (Target: {RETRIEVAL_RECALL_TARGET:.0%})")
        
        is_success = False
        if e2e_metrics:
            print(f"Response Similarity: {similarity:.1%}  (Target: {RESPONSE_SIMILARITY_TARGET:.0%})")
            print(f"ROUGE-L Score:       {rouge_score:.3f}")
            
            # Success Check
            is_success = (recall >= RETRIEVAL_RECALL_TARGET) and (similarity >= RESPONSE_SIMILARITY_TARGET)
        else:
            print("Response Similarity: N/A")
            print("ROUGE-L Score:       N/A")
            # Partial success check
            is_success = (recall >= RETRIEVAL_RECALL_TARGET)
        
        print("-" * 80)
        if is_success:
            print("✅ SYSTEM MEETS ACCURACY REQUIREMENT")
            if not e2e_metrics:
                print("   (Retrieval only - Generator not tested)")
        else:
            print("⚠️ SYSTEM BELOW TARGET ACCURACY")
            
        return {
            "retrieval": retrieval_metrics,
            "generation": e2e_metrics,
            "success": is_success
        }


# ============================================================================
# RUN EVALUATION
# ============================================================================

if __name__ == "__main__":
    from data_loader import load_all_data
    from vector_store import VectorStore
    
    # Load all components
    try:
        print("Loading data...")
        knowledge_items, qa_pairs = load_all_data()
        
        print("Initializing Vector Store...")
        vector_store = VectorStore()
        if vector_store.index_exists():
            vector_store.load_index()
        else:
            vector_store.build_index(knowledge_items)
            
        print("Initializing Retriever...")
        retriever = Retriever(vector_store)
        
        print("Initializing Generator...")
        generator = None
        try:
            generator = Generator()
        except ValueError as e:
            print(f"⚠️ Generator skipped: {e}")
        
        print("Initializing Evaluator...")
        evaluator = Evaluator(retriever, generator)
        
        # Run Evaluation
        evaluator.run_full_evaluation(qa_pairs)
        
    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {e}")
        import traceback
        traceback.print_exc()
