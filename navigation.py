import streamlit as st

def add_portfolio_navigation(current_project):
    """
    Adds a navigation dropdown in the sidebar to jump between Hugging Face spaces.
    """
    projects = {
        "RAGbasedsolution": {
            "name": "🤖 RAG-Based IT Support",
            "url": "https://huggingface.co/spaces/vnicks177/RAGbasedsolution-demo"
        },
        "RecommenderSystem": {
            "name": "🎓 Career Recommender",
            "url": "https://huggingface.co/spaces/vnicks177/RecommenderSystem-demo"
        },
        "TimeSeriesForecasting": {
            "name": "⚡ Energy Forecasting",
            "url": "https://huggingface.co/spaces/vnicks179/TimeSeriesForecasting-demo"
        },
        "ComputerVision": {
            "name": "🔍 Defect Detection",
            "url": "https://huggingface.co/spaces/vnicks179/ComputerVision-demo"
        },
        "SentimentAnalysis": {
            "name": "🎯 VoC Sentiment Analysis",
            "url": "https://huggingface.co/spaces/vnicks177/SentimentAnalysis-demo"
        }
    }
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🌐 Portfolio Navigation")
    
    options = list(projects.keys())
    
    def format_proj(key):
        if key == current_project:
            return f"{projects[key]['name']} (Current)"
        return projects[key]['name']
        
    selected = st.sidebar.selectbox(
        "Explore Other Products:",
        options=options,
        index=options.index(current_project),
        format_func=format_proj,
        key="portfolio_nav_selector"
    )
    
    if selected != current_project:
        st.sidebar.info(f"Navigate to {projects[selected]['name']}:")
        st.sidebar.link_button("Open Application 🚀", projects[selected]["url"], use_container_width=True)
