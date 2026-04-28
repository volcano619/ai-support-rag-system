import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

def apply_global_theme():
    """Apply global CSS and Plotly theme to the Streamlit app."""
    
    # 1. Inject Global CSS
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;800&family=Roboto+Mono&display=swap');

        /* Base App Styling */
        .stApp {
            background-color: #F8FAFC;
            font-family: 'Inter', sans-serif;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0F172A !important;
            color: #F8FAFC !important;
        }
        
        [data-testid="stSidebar"] .stMarkdown, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stNumberInput label,
        [data-testid="stSidebar"] .stDateInput label {
            color: #CBD5E1 !important;
        }

        /* Sidebar Input Contrast */
        [data-testid="stSidebar"] input {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border-color: #334155 !important;
        }

        /* Sidebar Button Styling (Legibility) */
        [data-testid="stSidebar"] button {
            background-color: #1E293B !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
        }
        
        [data-testid="stSidebar"] button:hover {
            background-color: #334155 !important;
            border-color: #475569 !important;
        }
        
        [data-testid="stSidebar"] button p {
            color: #F8FAFC !important;
        }

        /* Metric Cards */
        .metric-card {
            background: white;
            padding: 1.25rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid #E2E8F0;
            margin-bottom: 1rem;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: #64748B;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        
        .metric-value {
            font-size: 1.5rem;
            color: #0F172A;
            font-weight: 700;
        }
        
        .metric-delta {
            font-size: 0.875rem;
            font-weight: 600;
        }
        
        .delta-pos { color: #10B981; }
        .delta-neg { color: #EF4444; }

        /* Headers */
        .main-header {
            font-size: 2.5rem;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.025em;
            margin-bottom: 0.5rem;
        }
        
        .sub-header {
            font-size: 1.125rem;
            color: #64748B;
            margin-bottom: 2rem;
            line-height: 1.5;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 0px;
            color: #64748B;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            color: #1E88E5 !important;
            border-bottom-color: #1E88E5 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # 2. Apply Plotly Theme
    theme = go.layout.Template()
    theme.layout.paper_bgcolor = 'rgba(0,0,0,0)'
    theme.layout.plot_bgcolor = 'rgba(0,0,0,0)'
    theme.layout.font = dict(family="Inter, sans-serif", size=13, color="#1E293B")
    theme.layout.colorway = ['#1E88E5', '#FF7043', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    theme.layout.margin = dict(t=40, b=40, l=40, r=40)
    theme.layout.xaxis = dict(showgrid=True, gridcolor='#F1F5F9', zeroline=False)
    theme.layout.yaxis = dict(showgrid=True, gridcolor='#F1F5F9', zeroline=False)
    
    pio.templates["portfolio_theme"] = theme
    pio.templates.default = "portfolio_theme"

def create_metric_card(label, value, delta=None, delta_pos=True):
    """Create a custom styled metric card."""
    delta_html = ""
    if delta:
        delta_class = "delta-pos" if delta_pos else "delta-neg"
        prefix = "+" if delta_pos and not str(delta).startswith('+') else ""
        delta_html = f'<div class="metric-delta {delta_class}">{prefix}{delta}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def add_header(title, subtitle):
    """Add a consistent header to the app."""
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{subtitle}</div>', unsafe_allow_html=True)
