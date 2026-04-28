"""
Streamlit App for RAG System

This module provides an interactive web interface for the RAG system.
It allows users to:
1. Ask questions naturally
2. See the retrieved documents and their scores
3. View the generated Answer
4. See citations and source metadata

WHY THIS APP EXISTS:
- Provides a user-friendly way to interact with the system
- Allows manual verification of responses
- Demonstrates the system capabilities to stakeholders

Author: RAG System
"""

import random
import streamlit as st
import time
from pathlib import Path
import logging

from config import APP_TITLE, APP_LAYOUT, DEBUG_MODE
from data_loader import load_all_data
from vector_store import VectorStore
from retriever import Retriever
from generator import Generator
import shared_ui

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT, page_icon="🤖")

# Apply global theme
shared_ui.apply_global_theme()

# Project-specific CSS for chat and buttons
st.markdown("""
<style>
    div.stButton > button:first-child {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        background-color: white;
        transition: all 0.2s;
    }
    div.stButton > button:hover {
        border-color: #1E88E5;
        color: #1E88E5;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZATION (Cached)
# ============================================================================

@st.cache_resource
def initialize_system():
    """
    Initialize all system components (cached to avoid reloading on every interaction)
    """
    with st.spinner("Initializing RAG System... (Loading Knowledge Base & Models)"):
        logger.info("Initializing system...")
        
        # 1. Load Data
        knowledge_items, qa_pairs = load_all_data()
        
        # 2. Vector Store
        vector_store = VectorStore()
        if vector_store.index_exists():
            vector_store.load_index()
        else:
            vector_store.build_index(knowledge_items)
            # Auto-save for next time
            vector_store.save_index()
            
        # 3. Retriever
        retriever = Retriever(vector_store)
        
        # 4. Generator
        generator = Generator()
        
        return retriever, generator, qa_pairs

# Initialize
try:
    retriever, generator, qa_pairs = initialize_system()
    st.toast("System Initialized Successfully!", icon="✅")
except Exception as e:
    st.error(f"System Initialization Failed: {e}")
    st.stop()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ RAG Settings")
    
    st.markdown("### Retrieval")
    if "top_k" not in st.session_state:
        from config import RETRIEVAL_TOP_K
        st.session_state.top_k = RETRIEVAL_TOP_K
    top_k = st.slider("Documents to Retrieve", min_value=1, max_value=5, key="top_k")
    
    st.markdown("### Generation")
    if "temperature" not in st.session_state:
        from config import LLM_TEMPERATURE
        st.session_state.temperature = LLM_TEMPERATURE
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, key="temperature", step=0.1)
    
    st.markdown("---")
    st.markdown("### 🛰️ System Status")
    status_cols = st.columns(2)
    with status_cols[0]:
        st.caption("Vector Store")
        st.success("Connected")
    with status_cols[1]:
        st.caption("LLM Engine")
        st.success("Ready")
    
    st.markdown("---")
    st.markdown("### Debug Info")
    if st.checkbox("Show Debug Details", value=DEBUG_MODE):
        st.info("Debug mode enabled. Detailed metadata will be shown.")

# ============================================================================
# MAIN INTERFACE
# ============================================================================

# Header
shared_ui.add_header(
    APP_TITLE,
    "AI-powered Q&A using Retrieval-Augmented Generation | *Reducing IT support costs by $20-25 per ticket*"
)

# Suggested Questions (Random on Refresh)
if "random_questions" not in st.session_state:
    # Pick 3 random questions from the dataset
    all_questions = [qa.question for qa in qa_pairs]
    # Ensure we have enough questions
    if len(all_questions) >= 3:
        st.session_state.random_questions = random.sample(all_questions, 3)
    else:
        st.session_state.random_questions = all_questions

st.markdown("### 🎲 Try an example:")
cols = st.columns(3)
selected_question = None

for i, col in enumerate(cols):
    if i < len(st.session_state.random_questions):
        if col.button(st.session_state.random_questions[i]):
            selected_question = st.session_state.random_questions[i]

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Logic
user_input = st.chat_input("How do I reset my PIN?")
query = user_input or selected_question

if query:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    # Generate Response
    with st.chat_message("assistant"):
        start_time = time.time()
        
        with st.status("Thinking...", expanded=True) as status:
            # 1. Retrieval
            st.write("🔍 Retrieving documents...")
            context, sources = retriever.get_context_for_generation(query, top_k=top_k)
            
            if not sources:
                st.error("No relevant documents found.")
                status.update(label="Retrieval Failed", state="error")
                st.stop()
            
            # Show retrieved docs
            st.write(f"✅ Found {len(sources)} relevant documents")
            for s in sources:
                st.write(f"- **{s['topic']}** (Score: {s['score']:.3f})")
            
            # 2. Generation
            st.write("🧠 Generating response...")
            response, metadata = generator.generate_response(
                query, context, sources, temperature=temperature
            )
            
            status.update(label="Complete!", state="complete", expanded=False)
        
        # Display Answer
        st.markdown(response)
        
        # Display Sources/Citations
        st.markdown("### 📚 Sources Used")
        for s in sources:
            with st.expander(f"Document {s['number']}: {s['topic']} (Relevance Score: {s['score']:.3f})"):
                st.markdown(f"**From ID:** `{s['id']}`")
                if 'metadata' in s and 'stage1_score' in s['metadata']:
                    st.caption(f"Stage 1 Score (FAISS): {s['metadata']['stage1_score']:.3f}")
                    st.caption(f"Stage 2 Score (Reranker): {s['metadata']['stage2_score']:.3f}")
                
                # Show snippet
                st.markdown(f"**Knowledge Content:**\n{s['text']}")        
        # Debug Metadata
        if st.checkbox("Show Generation Metadata"):
            st.json(metadata)
        
        # Metrics row
        col1, col2 = st.columns(2)
        with col1:
            shared_ui.create_metric_card("Retrieved Docs", str(len(sources)), delta="Contextual Support", delta_pos=True)
        with col2:
            shared_ui.create_metric_card("Latency", f"{time.time() - start_time:.2f}s", delta="Real-time")
        
        # Save interaction
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; font-size: 0.875rem; padding: 2rem 0;">
    🤖 RAG Based Support System | Built with FAISS + LlamaIndex<br>
    <span style="font-family: 'Roboto Mono', monospace;">Version 1.1.0-Premium</span>
</div>
""", unsafe_allow_html=True)
