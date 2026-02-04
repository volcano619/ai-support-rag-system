"""
Generator Module for RAG System

This module handles Response generation using a LOCAL LLM (Flan-T5).

WHY LOCAL LLM:
- User requirement: "No API Key" architecture
- Hardware constraint: Low RAM (<2GB)
- "google/flan-t5-base" is chosen for efficiency and robustness

Author: RAG System
"""

import logging
from typing import List, Dict, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import LLM_MODEL_NAME, LLM_MAX_TOKENS, LLM_TEMPERATURE

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Generator:
    """
    Local LLM response generator using HuggingFace Transformers
    
    WHY THIS IMPLEMENTATION:
    - Runs entirely offline
    - Uses Seq2Seq model (T5) which is great for "text-to-text" tasks like QA
    """
    
    def __init__(self, model_name: str = LLM_MODEL_NAME):
        """
        Initialize local generator
        
        Args:
            model_name: HuggingFace model name (e.g., "google/flan-t5-base")
        """
        logger.info(f"Initializing Local Generator with model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model
            # WHY device_map="cpu":
            # - We know system is RAM constrained and lacks NVIDIA GPU
            # - "auto" might try to use GPU and fail
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                device_map="cpu", 
                torch_dtype=torch.float32 # FP32 is safer for CPU
            )
            
            logger.info("✅ Local Generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise RuntimeError(f"Could not load model {model_name}. Check internet connection or RAM.") from e

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt formatted for Flan-T5 (Question First Strategy)
        
        WHY QUESTION FIRST:
        - T5 has a 512 token limit.
        - If we put Context first, the Question at the end gets truncated.
        - By putting Question first, the model always knows WHAT to do, even if context is cut.
        """
        # T5 prefers: "question: ... context: ..."
        # Using standard T5 prefix format
        prompt = (
            f"question: {query} "
            f"context: {context}"
        )
        return prompt
    
    def generate_response(
        self,
        query: str,
        context: str,
        sources_metadata: List[Dict],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS
    ) -> Tuple[str, Dict]:
        """
        Generate response using local LLM
        """
        if not context:
            return "I don't have enough information to answer that question.", {"error": "no_context"}

        logger.info(f"Generating response for query: '{query[:50]}...'")
        
        # Prepare prompt
        prompt = self._create_prompt(query, context)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            # Generate
            # WHY generation parameters:
            # - do_sample=False: Deterministic greedy decoding (temperature ignored)
            # - max_length: Limit response size
            outputs = self.model.generate(
                **inputs,
                max_length=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
            )
            
            # Decode
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Metadata
            metadata = {
                "model": LLM_MODEL_NAME,
                "sources_used": [s['topic'] for s in sources_metadata]
            }
            
            logger.info(f"Generated response: {response_text[:50]}...")
            return response_text, metadata

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "Error generating response.", {"error": str(e)}

if __name__ == "__main__":
    print("Testing Local Generator...")
    gen = Generator()
    ctx = "To reset your password, visit password.corp.com and enter your employee ID."
    q = "Where do I go to reset my password?"
    res, _ = gen.generate_response(q, ctx, [])
    print(f"Query: {q}\nResponse: {res}")
