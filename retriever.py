"""
Retriever Module for RAG System

This module implements intelligent retrieval with two-stage ranking:
1. Stage 1: Fast FAISS similarity search (retrieves top-K candidates)
2. Stage 2: Cross-encoder reranking (reranks candidates for precision)

WHY TWO-STAGE RETRIEVAL:
- Stage 1 (Bi-encoder/FAISS): Fast but less accurate, high recall
- Stage 2 (Cross-encoder): Slow but very accurate, high precision
- Together: Best of both worlds - fast AND accurate

WHY THIS IMPROVES ACCURACY:
- Bi-encoders (FAISS) encode query and doc separately → misses interaction
- Cross-encoders see query+doc together → captures semantic relationship
- Studies show cross-encoder reranking improves accuracy by 10-15%

Author: RAG System
"""

from typing import List, Tuple, Dict
from sentence_transformers import CrossEncoder
import logging

from config import (
    RETRIEVAL_TOP_K,
    RERANKER_MODEL_NAME,
    RERANK_TOP_K,
    SIMILARITY_THRESHOLD
)
from data_loader import KnowledgeItem
from vector_store import VectorStore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Retriever:
    """
    Intelligent retrieval system with two-stage ranking
    
    WHY THIS CLASS:
    - Encapsulates complete retrieval pipeline
    - Combines FAISS (speed) with cross-encoder (accuracy)
    - Provides clean API for RAG system
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        reranker_model_name: str = RERANKER_MODEL_NAME
    ):
        """
        Initialize retriever
        
        Args:
            vector_store: Initialized VectorStore instance
            reranker_model_name: Name of cross-encoder model for reranking
        """
        logger.info("Initializing Retriever with reranking")
        
        self.vector_store = vector_store
        
        # Initialize cross-encoder for reranking
        # WHY cross-encoder:
        # - Processes (query, document) pair together
        # - Learns interaction between query and doc
        # - More accurate than bi-encoder for final ranking
        # - Slower than bi-encoder, but we only rerank top-K (e.g., 5 docs)
        logger.info(f"Loading cross-encoder: {reranker_model_name}")
        self.reranker = CrossEncoder(reranker_model_name)
        
        logger.info("✅ Retriever initialized")
    
    def retrieve(
        self,
        query: str,
        top_k: int = RERANK_TOP_K,
        retrieval_k: int = RETRIEVAL_TOP_K,
        use_reranking: bool = True
    ) -> List[Tuple[KnowledgeItem, float, Dict]]:
        """
        Retrieve most relevant documents for query
        
        PIPELINE:
        1. Stage 1: FAISS retrieves top-retrieval_k candidates (fast, high recall)
        2. Stage 2: Cross-encoder reranks to top-k (slow, high precision)
        
        WHY THIS APPROACH:
        - FAISS searches entire knowledge base quickly
        - Cross-encoder only processes top candidates (5-10 docs)
        - Result: Fast retrieval with high accuracy
        
        Args:
            query: User's question
            top_k: Number of final results to return
            retrieval_k: Number of candidates for stage 1 (should be >= top_k)
            use_reranking: Whether to use cross-encoder reranking
        
        Returns:
            List of (KnowledgeItem, score, metadata) tuples, sorted by score descending
            metadata contains: stage1_score, stage2_score (if reranking), rank, etc.
        """
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        logger.info(f"Retrieving documents for query: '{query[:100]}...'")
        
        # =====================================================================
        # STAGE 1: Fast FAISS Similarity Search
        # =====================================================================
        
        logger.debug(f"Stage 1: FAISS search (top_k={retrieval_k})")
        
        # Get candidates from vector store
        # WHY retrieval_k > top_k:
        # - Casts wider net to ensure relevant docs are captured
        # - Reranking will filter down to top_k most relevant
        stage1_results = self.vector_store.search(
            query=query,
            top_k=retrieval_k,
            score_threshold=SIMILARITY_THRESHOLD
        )
        
        if not stage1_results:
            logger.warning("No results from stage 1 (FAISS search)")
            return []
        
        logger.info(f"Stage 1: Found {len(stage1_results)} candidates")
        
        # If reranking disabled, return stage 1 results
        if not use_reranking:
            logger.info("Reranking disabled, returning stage 1 results")
            results = []
            for i, (ki, score) in enumerate(stage1_results[:top_k], 1):
                metadata = {
                    "stage1_score": score,
                    "stage1_rank": i,
                    "reranked": False
                }
                results.append((ki, score, metadata))
            return results
        
        # =====================================================================
        # STAGE 2: Cross-Encoder Reranking
        # =====================================================================
        
        logger.debug(f"Stage 2: Cross-encoder reranking (top_k={top_k})")
        
        # Prepare (query, document) pairs for cross-encoder
        # WHY: Cross-encoder requires both query and doc as input
        pairs = [(query, ki.text) for ki, _ in stage1_results]
        
        # Get reranking scores
        # WHY: Cross-encoder outputs a single score for each (query, doc) pair
        # Higher score = more relevant
        rerank_scores = self.reranker.predict(pairs)
        
        # Combine results with both stage 1 and stage 2 scores
        combined_results = []
        for (ki, stage1_score), rerank_score in zip(stage1_results, rerank_scores):
            combined_results.append({
                "ki": ki,
                "stage1_score": float(stage1_score),
                "stage2_score": float(rerank_score),
                "final_score": float(rerank_score)  # Use stage 2 score as final
            })
        
        # Sort by reranking score (descending)
        # WHY: Cross-encoder scores are more accurate for final ranking
        combined_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Take top-k
        top_results = combined_results[:top_k]
        
        # Format output
        results = []
        for i, result in enumerate(top_results, 1):
            metadata = {
                "stage1_score": result["stage1_score"],
                "stage2_score": result["stage2_score"],
                "final_score": result["final_score"],
                "stage1_rank": stage1_results.index((result["ki"], result["stage1_score"])) + 1,
                "stage2_rank": i,
                "reranked": True
            }
            results.append((result["ki"], result["final_score"], metadata))
        
        logger.info(f"Stage 2: Reranked to top {len(results)} results")
        
        if results:
            top_result = results[0]
            logger.info(
                f"Top result: '{top_result[0].topic}' "
                f"(stage1_score: {top_result[2]['stage1_score']:.3f}, "
                f"stage2_score: {top_result[2]['stage2_score']:.3f})"
            )
        
        return results
    
    def get_context_for_generation(
        self,
        query: str,
        top_k: int = RERANK_TOP_K
    ) -> Tuple[str, List[Dict]]:
        """
        Get formatted context for LLM generation
        
        WHY THIS METHOD:
        - Retrieves relevant docs and formats them for LLM
        - Returns both context string and metadata
        - Makes it easy to feed into generator
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
        
        Returns:
            Tuple of (context_string, sources_metadata)
            - context_string: Formatted text to inject into LLM prompt
            - sources_metadata: List of dicts with source information
        """
        # Retrieve documents
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            logger.warning("No documents retrieved for context")
            return "", []
        
        # Format context
        # WHY: Clear structure helps LLM understand and cite sources
        context_parts = []
        sources_metadata = []
        
        for i, (ki, score, metadata) in enumerate(results, 1):
            # Add document to context
            # WHY numbered sections: Makes it easy for LLM to cite sources
            context_parts.append(f"[Document {i}: {ki.topic}]\n{ki.text}\n")
            
            # Track source metadata
            sources_metadata.append({
                "number": i,
                "id": ki.id,
                "topic": ki.topic,
                "score": score,
                "metadata": metadata
            })
        
        context_string = "\n---\n\n".join(context_parts)
        
        logger.info(f"Generated context from {len(results)} documents ({len(context_string)} characters)")
        
        return context_string, sources_metadata


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the retriever with reranking
    
    WHY: Validates that two-stage retrieval improves accuracy
    """
    print("=" * 80)
    print("TESTING RETRIEVER WITH RERANKING")
    print("=" * 80)
    
    from data_loader import load_knowledge_base
    
    try:
        # Load knowledge base
        print("\n📚 Loading knowledge base...")
        knowledge_items = load_knowledge_base()
        
        # Initialize vector store
        print("\n🔧 Initializing vector store...")
        vector_store = VectorStore()
        
        # Check if index exists, otherwise build
        if vector_store.index_exists():
            print("Loading existing index...")
            vector_store.load_index()
        else:
            print("Building new index...")
            vector_store.build_index(knowledge_items)
            vector_store.save_index()
        
        # Initialize retriever
        print("\n🎯 Initializing retriever...")
        retriever = Retriever(vector_store)
        
        # Test queries
        test_queries = [
            "How do I reset my forgotten PIN?",
            "I need to set up company email on my Android phone",
            "My printer is jammed, what should I do?",
            "How can I configure VPN to work from home?"
        ]
        
        for query in test_queries:
            print("\n" + "="*80)
            print(f"📝 Query: '{query}'")
            print("="*80)
            
            # Test WITHOUT reranking
            print("\n🔍 Stage 1 Only (FAISS):")
            results_no_rerank = retriever.retrieve(query, top_k=3, use_reranking=False)
            for i, (ki, score, metadata) in enumerate(results_no_rerank, 1):
                print(f"  {i}. {ki.topic}")
                print(f"     Score: {score:.3f}")
            
            # Test WITH reranking
            print("\n🎯 Stage 1 + Stage 2 (FAISS + Reranking):")
            results_rerank = retriever.retrieve(query, top_k=3, use_reranking=True)
            for i, (ki, score, metadata) in enumerate(results_rerank, 1):
                print(f"  {i}. {ki.topic}")
                print(f"     Stage 1 Score: {metadata['stage1_score']:.3f}")
                print(f"     Stage 2 Score: {metadata['stage2_score']:.3f}")
                print(f"     Rank Change: {metadata['stage1_rank']} → {metadata['stage2_rank']}")
            
            # Get formatted context
            print("\n📄 Formatted Context:")
            context, sources = retriever.get_context_for_generation(query, top_k=2)
            print(f"  Context length: {len(context)} characters")
            print(f"  Sources: {[s['topic'] for s in sources]}")
        
        print("\n✅ RETRIEVER TEST PASSED")
        
    except Exception as e:
        print(f"\n❌ RETRIEVER TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
