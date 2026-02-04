
import nbformat as nbf
from pathlib import Path
import re

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # helper to read file
    def read_file(fname):
        with open(fname, 'r') as f:
            content = f.read()
            # Remove __file__ reliance
            content = content.replace('Path(__file__).parent', 'Path(".")')
            # Remove imports of locals to avoid errors in cells
            content = re.sub(r'from (config|data_loader|vector_store|retriever|generator) import .*', '', content)
            return content

    # 1. Setup
    nb.cells.append(nbf.v4.new_markdown_cell("# RAG System for IT Support (Local CPU Version)\n\nThis notebook implements a complete Retrieve-Augmented Generation system using Flan-T5 and FAISS."))
    
    nb.cells.append(nbf.v4.new_code_cell("""
# Install Dependencies
!pip install -q transformers sentence-transformers faiss-cpu pandas numpy torch accelerate sentencepiece
"""))
    
    # 2. Config
    config_code = read_file("config.py")
    nb.cells.append(nbf.v4.new_markdown_cell("## 1. Configuration"))
    nb.cells.append(nbf.v4.new_code_cell(config_code))
    
    # 3. Data Loader
    loader_code = read_file("data_loader.py")
    nb.cells.append(nbf.v4.new_markdown_cell("## 2. Data Loading & Chunking"))
    nb.cells.append(nbf.v4.new_code_cell(loader_code))
    
    # 4. Vector Store
    vs_code = read_file("vector_store.py")
    nb.cells.append(nbf.v4.new_markdown_cell("## 3. Vector Database (FAISS)"))
    nb.cells.append(nbf.v4.new_code_cell(vs_code))

    # 5. Retriever
    ret_code = read_file("retriever.py")
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. Retriever Logic"))
    nb.cells.append(nbf.v4.new_code_cell(ret_code))
    
    # 6. Generator
    gen_code = read_file("generator.py")
    nb.cells.append(nbf.v4.new_markdown_cell("## 5. Generator (Flan-T5)"))
    nb.cells.append(nbf.v4.new_code_cell(gen_code))
    
    # 7. Main Execution
    nb.cells.append(nbf.v4.new_markdown_cell("## 6. Execution & Testing"))
    
    main_code = """
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize System
print("Initializing System...")

# 1. Load Data
try:
    knowledge_items = load_knowledge_base()
except FileNotFoundError:
    print("Please upload 'rag_sample_qas_from_kis.csv' to the current directory.")
    knowledge_items = []

if knowledge_items:
    # 2. Build Index
    vector_store = VectorStore()
    vector_store.build_index(knowledge_items)

    # 3. Init Components
    retriever = Retriever(vector_store)
    generator = Generator()

    print("\\n✅ System Initialized!\\n")

    # Interactive Loop
    def ask(query):
        print(f"❓ Question: {query}")
        print("🔍 Retrieving...")
        context, sources = retriever.get_context_for_generation(query, top_k=2)
        
        for s in sources:
            print(f"  - Found: {s['topic']} (Score: {s['score']:.3f})")

        print("🧠 Generating...")
        response, _ = generator.generate_response(query, context, sources)
        print(f"🤖 Answer: {response}\\n")
        print("-" * 50)

    # Examples
    ask("How do I reset my PIN?")
    ask("How do I configure VPN?")
else:
    print("Using dummy data or failed to load.")
"""
    nb.cells.append(nbf.v4.new_code_cell(main_code))

    with open('RAG_Solution.ipynb', 'w') as f:
        nbf.write(nb, f)
    
    print("Notebook 'RAG_Solution.ipynb' created successfully.")

if __name__ == "__main__":
    create_notebook()
