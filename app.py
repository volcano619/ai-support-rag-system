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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(page_title=APP_TITLE, layout=APP_LAYOUT)

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
    st.title("⚙️ RAG Settings")
    
    st.markdown("### Retrieval")
    top_k = st.slider("Documents to Retrieve", min_value=1, max_value=5, value=3)
    
    st.markdown("### Generation")
    temperature = st.slider("Temperature (Creativity)", 0.0, 1.0, 0.0, 0.1)
    
    st.markdown("---")
    st.markdown("### Debug Info")
    if st.checkbox("Show Debug Details", value=DEBUG_MODE):
        st.info("Debug mode enabled. Detailed metadata will be shown.")

# ============================================================================
# MAIN INTERFACE
# ============================================================================

st.title(APP_TITLE)
st.markdown("""
Ask any question about IT support topics (e.g., VPN, Email, Printer, PIN reset).
The system will retrieve relevant knowledge items and generate a grounded answer.
""")

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

# Custom CSS to make buttons full width and look clickable
st.markdown("""
<style>
div.stButton > button:first-child {
    width: 100%;
    height: 100%;
    white-space: normal;
    word-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)

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
            with st.expander(f"Document {s['number']}: {s['topic']} (Relevance: {s['score']:.2%})"):
                st.markdown(f"**From ID:** `{s['id']}`")
                if 'metadata' in s and 'stage1_score' in s['metadata']:
                    st.caption(f"Stage 1 Score (FAISS): {s['metadata']['stage1_score']:.3f}")
                    st.caption(f"Stage 2 Score (Reranker): {s['metadata']['stage2_score']:.3f}")
                
                # Show snippet
                # Find matching Knowledge Item text from context
                # (Simple text display)
                pass 
        
        # Debug Metadata
        if st.checkbox("Show Generation Metadata"):
            st.json(metadata)
        
        # Save interaction
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Latency metric
        latency = time.time() - start_time
        st.caption(f"Response generated in {latency:.2f} seconds")
