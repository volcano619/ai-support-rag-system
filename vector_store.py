"""
Vector Store Module for RAG System

This module handles:
1. Generating embeddings for knowledge base documents
2. Building and managing the FAISS vector index
3. Performing similarity search

WHY THIS MODULE EXISTS:
- Centralizes all vector database operations
- Provides fast semantic search over knowledge base
- Handles index persistence (save/load)

WHY FAISS:
- Extremely fast similarity search (optimized by Facebook AI)
- Can scale to millions of documents
- No external dependencies (runs locally)
- Industry standard for vector search

Author: RAG System
"""

import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
import logging

from config import (
    EMBEDDING_MODEL_NAME,
    FAISS_INDEX_DIR,
    RETRIEVAL_TOP_K,
    SIMILARITY_THRESHOLD
)
from data_loader import KnowledgeItem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector embeddings and FAISS index for semantic search
    
    WHY THIS CLASS:
    - Encapsulates all vector database logic
    - Provides clean API for retrieval
    - Handles embedding generation and caching
    """
    
    def __init__(self, embedding_model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize vector store
        
        Args:
            embedding_model_name: Name of the sentence-transformers model
        """
        logger.info(f"Initializing VectorStore with model: {embedding_model_name}")
        
        # Initialize embedding model
        # WHY sentence-transformers:
        # - Pre-trained on semantic similarity tasks
        # - Produces high-quality embeddings for Q&A matching
        # - Easy to use, well-maintained library
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Get embedding dimension
        # WHY: Need this to initialize FAISS index with correct dimension
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize FAISS index (will be built when add_documents is called)
        self.index = None
        
        # Store knowledge items for retrieval
        # WHY: After FAISS returns document IDs, we need to map back to original docs
        self.knowledge_items: List[KnowledgeItem] = []
        
        # Store embeddings for potential reuse
        self.embeddings: np.ndarray = None
    
    def _encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for texts
        
        WHY SEPARATE METHOD:
        - Can be used for both documents and queries
        - Centralizes encoding logic
        - Easy to add batching, caching, etc.
        
        Args:
            texts: List of text strings to encode
            show_progress: Show progress bar
        
        Returns:
            Numpy array of embeddings, shape (n_texts, embedding_dim)
        """
        logger.info(f"Encoding {len(texts)} texts...")
        
        # Generate embeddings
        # WHY normalize_embeddings=True:
        # - Normalized vectors allow using cosine similarity via dot product
        # - Faster computation and better numerical stability
        # - FAISS IndexFlatIP (inner product) can be used for cosine similarity
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Critical for cosine similarity
        )
        
        return embeddings
    
    def build_index(self, knowledge_items: List[KnowledgeItem]):
        """
        Build FAISS index from knowledge base documents
        
        WHY THIS METHOD:
        - Creates the searchable vector database
        - Must be called before search can be performed
        
        Args:
            knowledge_items: List of KnowledgeItem objects to index
        """
        logger.info(f"Building FAISS index for {len(knowledge_items)} documents")
        
        if not knowledge_items:
            raise ValueError("No knowledge items provided for indexing")
        
        # Store knowledge items
        self.knowledge_items = knowledge_items
        
        # Extract texts for embedding
        # WHY: We embed the full text of each KI article
        # The topic is learned implicitly in the text content
        texts = [ki.text for ki in knowledge_items]
        
        # Generate embeddings
        self.embeddings = self._encode_texts(texts)
        
        logger.info(f"Generated embeddings with shape: {self.embeddings.shape}")
        
        # Build FAISS index
        # WHY IndexFlatIP (Inner Product):
        # - "Flat" = exhaustive search (guaranteed to find best matches)
        # - "IP" = inner product (equivalent to cosine similarity for normalized vectors)
        # - For small datasets (<10k docs), flat search is fast enough
        # - For larger datasets, could use IndexIVFFlat for approximate search
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        # WHY: FAISS requires float32 numpy arrays
        self.index.add(self.embeddings.astype('float32'))
        
        logger.info(f"✅ FAISS index built with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        score_threshold: float = SIMILARITY_THRESHOLD
    ) -> List[Tuple[KnowledgeItem, float]]:
        """
        Search for most similar documents to query
        
        WHY THIS METHOD:
        - Core retrieval function for RAG system
        - Returns ranked results with similarity scores
        
        Args:
            query: User's question/query text
            top_k: Number of results to return
            score_threshold: Minimum similarity score (0-1)
        
        Returns:
            List of (KnowledgeItem, similarity_score) tuples, sorted by score descending
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first")
        
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Encode query
        # WHY: Query needs to be in same embedding space as documents
        query_embedding = self._encode_texts([query], show_progress=False)
        
        # Search FAISS index
        # WHY top_k: We retrieve slightly more candidates for reranking
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Process results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Skip invalid indices (can happen if top_k > index size)
            if idx < 0 or idx >= len(self.knowledge_items):
                continue
            
            # Filter by score threshold
            # WHY: Removes completely irrelevant results
            if score < score_threshold:
                logger.debug(f"Skipping result with score {score:.3f} < threshold {score_threshold}")
                continue
            
            ki = self.knowledge_items[idx]
            results.append((ki, float(score)))
        
        logger.info(f"Found {len(results)} results for query (top_k={top_k}, threshold={score_threshold})")
        
        if results:
            logger.debug(f"Top result: {results[0][0].topic} (score: {results[0][1]:.3f})")
        
        return results
    
    def save_index(self, index_dir: Path = FAISS_INDEX_DIR):
        """
        Save FAISS index and metadata to disk
        
        WHY THIS METHOD:
        - Avoids rebuilding index every time (expensive for large datasets)
        - Enables loading pre-built index quickly
        
        Args:
            index_dir: Directory to save index files
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first")
        
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        # WHY: FAISS provides optimized serialization
        index_path = index_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save knowledge items
        # WHY: Need to map FAISS indices back to original documents
        ki_path = index_dir / "knowledge_items.pkl"
        with open(ki_path, 'wb') as f:
            pickle.dump(self.knowledge_items, f)
        logger.info(f"Saved knowledge items to {ki_path}")
        
        # Save embeddings (optional, for analysis)
        # WHY: Useful for debugging and analysis
        emb_path = index_dir / "embeddings.npy"
        np.save(emb_path, self.embeddings)
        logger.info(f"Saved embeddings to {emb_path}")
        
        # Save metadata
        metadata = {
            "model_name": EMBEDDING_MODEL_NAME,
            "embedding_dim": self.embedding_dim,
            "num_documents": len(self.knowledge_items)
        }
        metadata_path = index_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")
        
        logger.info("✅ Index saved successfully")
    
    def load_index(self, index_dir: Path = FAISS_INDEX_DIR):
        """
        Load FAISS index and metadata from disk
        
        WHY THIS METHOD:
        - Fast loading of pre-built index
        - Skips expensive embedding generation
        
        Args:
            index_dir: Directory containing index files
        
        Raises:
            FileNotFoundError: If index files don't exist
        """
        index_dir = Path(index_dir)
        
        if not index_dir.exists():
            raise FileNotFoundError(f"Index directory not found: {index_dir}")
        
        # Load FAISS index
        index_path = index_dir / "faiss_index.bin"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load knowledge items
        ki_path = index_dir / "knowledge_items.pkl"
        with open(ki_path, 'rb') as f:
            self.knowledge_items = pickle.load(f)
        logger.info(f"Loaded {len(self.knowledge_items)} knowledge items")
        
        # Load embeddings
        emb_path = index_dir / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
            logger.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
        
        logger.info("✅ Index loaded successfully")
    
    def index_exists(self, index_dir: Path = FAISS_INDEX_DIR) -> bool:
        """
        Check if a saved index exists
        
        WHY THIS METHOD:
        - Allows conditional loading vs building
        - Useful for caching logic
        
        Args:
            index_dir: Directory to check
        
        Returns:
            True if index files exist
        """
        index_dir = Path(index_dir)
        required_files = ["faiss_index.bin", "knowledge_items.pkl"]
        return all((index_dir / f).exists() for f in required_files)


# ============================================================================
# TESTING & VALIDATION
# ============================================================================

if __name__ == "__main__":
    """
    Test the vector store
    
    WHY: Validates that embedding and search work correctly
    """
    print("=" * 80)
    print("TESTING VECTOR STORE")
    print("=" * 80)
    
    from data_loader import load_knowledge_base
    
    try:
        # Load knowledge base
        print("\n📚 Loading knowledge base...")
        knowledge_items = load_knowledge_base()
        
        # Initialize vector store
        print("\n🔧 Initializing vector store...")
        vector_store = VectorStore()
        
        # Build index
        print("\n🏗️  Building FAISS index...")
        vector_store.build_index(knowledge_items)
        
        # Test search
        print("\n🔍 Testing search...")
        test_queries = [
            "How do I reset my PIN?",
            "Setting up email on my phone",
            "VPN not connecting"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = vector_store.search(query, top_k=3)
            
            if results:
                for i, (ki, score) in enumerate(results, 1):
                    print(f"  {i}. {ki.topic} (score: {score:.3f})")
            else:
                print("  No results found")
        
        # Test save/load
        print("\n💾 Testing save/load...")
        vector_store.save_index()
        
        # Create new instance and load
        vector_store2 = VectorStore()
        vector_store2.load_index()
        
        # Verify loaded index works
        print("\n🔍 Testing loaded index...")
        results = vector_store2.search("How do I reset my PIN?", top_k=3)
        print(f"Found {len(results)} results with loaded index")
        
        print("\n✅ VECTOR STORE TEST PASSED")
        
    except Exception as e:
        print(f"\n❌ VECTOR STORE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
