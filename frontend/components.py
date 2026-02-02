"""
UI components and styling for the Streamlit app.
"""


def get_custom_css() -> str:
    """Return custom CSS styles for the app."""
    return """
    <style>
        /* Main Layout */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .sub-header {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
            font-size: 1.1rem;
        }
        
        /* Cards */
        .info-card {
            background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border: 1px solid #667eea30;
            margin-bottom: 1rem;
        }
        
        /* Score Badges */
        .score-badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.875rem;
            font-weight: bold;
        }
        
        .score-high {
            background-color: #d4edda;
            color: #155724;
        }
        
        .score-medium {
            background-color: #fff3cd;
            color: #856404;
        }
        
        .score-low {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        /* Question Styling */
        .question-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
        
        .question-card.behavioral {
            border-left-color: #764ba2;
        }
        
        /* Pills/Tags */
        .tech-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.8rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            background: #667eea20;
            border: 1px solid #667eea40;
            color: #667eea;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            color: #999;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #eee;
        }
    </style>
    """


def render_header():
    """Render the app header."""
    import streamlit as st
    
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    st.markdown(
        '<div class="main-header">🤖 Mock Interview Copilot</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-header">'
        'Upload your resume and enter a job description to generate '
        'personalized interview questions with AI'
        '</div>',
        unsafe_allow_html=True
    )


def render_tech_stack():
    """Render the technology stack badges."""
    import streamlit as st
    
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <span class="tech-pill">Qwen2.5-7B</span>
            <span class="tech-pill">FastAPI</span>
            <span class="tech-pill">FAISS</span>
            <span class="tech-pill">Sentence Transformers</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_score_class(score: float) -> str:
    """Return CSS class based on relevance score."""
    if score >= 0.75:
        return "score-high"
    elif score >= 0.5:
        return "score-medium"
    return "score-low"


def render_question(question: dict, index: int, question_type: str = "technical"):
    """Render a single question card."""
    import streamlit as st
    
    q_text = question.get("question", "No question")
    answer = question.get("suggested_answer", "No suggested answer")
    score = question.get("relevance_score", 0.0)
    
    score_class = get_score_class(score)
    card_class = "question-card" if question_type == "technical" else "question-card behavioral"
    
    st.markdown(f"**Question {index}:**")
    st.markdown(f"*{q_text}*")
    st.markdown(
        f'<span class="score-badge {score_class}">Relevance: {score:.0%}</span>',
        unsafe_allow_html=True
    )
    
    with st.expander("💡 View suggested answer"):
        st.write(answer)
    
    st.markdown("---")


def render_tips():
    """Render interview tips section."""
    import streamlit as st
    
    st.markdown("## 💡 Interview Tips")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Technical Interview Tips:**
        - Explain your thought process clearly
        - Use examples from your actual projects
        - Discuss trade-offs and alternatives
        - Show your problem-solving approach
        """)
    
    with col2:
        st.info("""
        **Behavioral Interview Tips:**
        - Use the STAR method (Situation, Task, Action, Result)
        - Be specific with examples
        - Focus on your personal contributions
        - Highlight lessons learned
        """)


def render_footer():
    """Render the app footer."""
    import streamlit as st
    
    st.markdown(
        '<div class="footer">Built with ❤️ for confident interviews</div>',
        unsafe_allow_html=True
    )
