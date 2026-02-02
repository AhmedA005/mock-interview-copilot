"""
Mock Interview Copilot - Streamlit Frontend

A clean, modern UI for generating personalized interview questions
using AI-powered analysis of your resume and job description.
"""

import streamlit as st

from config import config
from api_client import APIClient
from components import (
    render_header,
    render_tech_stack,
    render_question,
    render_tips,
    render_footer,
)

# Page Configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout=config.LAYOUT,
)

# Initialize API client
api_client = APIClient()

# Render Header
render_header()
render_tech_stack()

# Sidebar - Configuration
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    api_url = st.text_input(
        "API URL",
        value=config.API_URL,
        help="Backend API endpoint URL",
    )
    
    api_key = st.text_input(
        "API Key",
        value=config.API_KEY,
        type="password",
        help="Your API authentication key",
    )
    
    # Update client with new values
    if api_url != config.API_URL or api_key != config.API_KEY:
        api_client = APIClient(api_url=api_url, api_key=api_key)
    
    st.divider()
    
    # Connection status
    st.markdown("### 🔌 Connection Status")
    if st.button("Test Connection", use_container_width=True):
        with st.spinner("Checking..."):
            is_healthy, message = api_client.check_health()
            if is_healthy:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
    
    st.divider()
    
    st.markdown("### 📖 Tips")
    st.caption("""
    - Upload a PDF resume (max 3 pages recommended)
    - Paste the full job description for better results
    - Generation takes 1-3 minutes
    - Questions are tailored to your experience
    """)

# Main Content
st.markdown("---")

# Input Section
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📄 Your Resume")
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        help="Upload your resume in PDF format",
        label_visibility="collapsed",
    )
    
    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
    else:
        st.info("Please upload your resume (PDF format)")

with col2:
    st.subheader("💼 Job Description")
    job_description = st.text_area(
        "Enter job description",
        placeholder="Paste the job posting or describe the key requirements, responsibilities, and tech stack...",
        height=200,
        help="The more detailed the job description, the better the questions",
        label_visibility="collapsed",
    )
    
    if job_description.strip():
        st.success(f"✅ {len(job_description)} characters entered")
    else:
        st.info("Please enter the target job description")

st.markdown("---")

# Generate Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    generate_button = st.button(
        "🚀 Generate Interview Questions",
        use_container_width=True,
        type="primary",
        disabled=not (uploaded_file and job_description.strip()),
    )

# Handle Generation
if generate_button:
    # Validation
    if uploaded_file is None:
        st.error("❌ Please upload your resume")
        st.stop()
    
    if len(job_description.strip()) < 20:
        st.error("❌ Please enter a more detailed job description (at least 20 characters)")
        st.stop()
    
    # Generate questions
    with st.spinner("🤖 AI is analyzing your resume and generating personalized questions... This may take 1-3 minutes"):
        success, result = api_client.generate_questions(
            uploaded_file,
            job_description,
        )
    
    if success:
        st.session_state["interview_result"] = result
        st.success("✅ Interview questions generated successfully!")
        st.balloons()
    else:
        st.error(f"❌ {result}")
        st.stop()

# Display Results
if "interview_result" in st.session_state:
    result = st.session_state["interview_result"]
    
    st.markdown("---")
    st.markdown("## 📝 Your Personalized Interview Questions")
    
    # Summary metrics
    tech_questions = result.get("technical_questions", [])
    behavioral_questions = result.get("behavioral_questions", [])
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Technical Questions", len(tech_questions))
    col2.metric("Behavioral Questions", len(behavioral_questions))
    
    all_scores = [
        q.get("relevance_score", 0)
        for q in tech_questions + behavioral_questions
    ]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    col3.metric("Avg Relevance", f"{avg_score:.0%}")
    
    st.markdown("---")
    
    # Questions tabs
    tab1, tab2 = st.tabs(["🛠️ Technical Questions", "💬 Behavioral Questions"])
    
    with tab1:
        if tech_questions:
            for i, q in enumerate(tech_questions, 1):
                render_question(q, i, "technical")
        else:
            st.info("No technical questions generated")
    
    with tab2:
        if behavioral_questions:
            for i, q in enumerate(behavioral_questions, 1):
                render_question(q, i, "behavioral")
        else:
            st.info("No behavioral questions generated")
    
    st.markdown("---")
    render_tips()

# Footer
render_footer()
