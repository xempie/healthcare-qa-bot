"""
Streamlit web application for Healthcare Q&A Bot.
"""

import streamlit as st
import asyncio
import sys
from pathlib import Path
import time
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import initialize_bot, get_bot
from config.settings import settings, MEDICAL_DISCLAIMER, EMERGENCY_MESSAGE


# Page configuration
st.set_page_config(
    page_title="Healthcare Q&A Bot",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #4CAF50, #45a049);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        background-color: #f0f8f0;
    }
    
    .emergency-alert {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .warning-alert {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_initialized_bot():
    """Get initialized bot instance with caching."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        bot = loop.run_until_complete(initialize_bot())
        return bot
    except Exception as e:
        st.error(f"Failed to initialize bot: {str(e)}")
        return None


def display_chat_message(message: str, is_user: bool = False, message_type: str = "normal"):
    """Display a chat message with appropriate styling."""
    
    if is_user:
        with st.container():
            st.markdown(f"""
            <div style="text-align: right; margin: 1rem 0;">
                <div style="display: inline-block; padding: 0.5rem 1rem; background-color: #2196f3; 
                           color: white; border-radius: 15px; max-width: 70%;">
                    {message}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        if message_type == "emergency":
            st.markdown(f"""
            <div class="emergency-alert">
                <strong>EMERGENCY ALERT</strong><br>
                {message}
            </div>
            """, unsafe_allow_html=True)
        elif message_type == "warning":
            st.markdown(f"""
            <div class="warning-alert">
                <strong>Medical Consultation Required</strong><br>
                {message}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message">
                <strong>Healthcare Assistant:</strong><br>
                {message}
            </div>
            """, unsafe_allow_html=True)


def display_sources(sources: list):
    """Display source information."""
    if sources:
        with st.expander(f"Sources ({len(sources)} documents)", expanded=False):
            for i, source in enumerate(sources, 1):
                st.markdown(f"""
                **Source {i}:**
                - **Type:** {source['metadata'].get('document_type', 'Unknown')}
                - **Specialty:** {source['metadata'].get('medical_specialty', 'General')}
                - **Content:** {source['content']}
                """)


def display_follow_up_questions(questions: list):
    """Display follow-up questions."""
    if questions:
        st.markdown("### You might also ask:")
        for question in questions:
            if st.button(question, key=f"followup_{hash(question)}"):
                st.session_state.current_question = question
                st.experimental_rerun()


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Healthcare Q&A Bot</h1>
        <p>Your AI-powered healthcare information assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'bot' not in st.session_state:
        with st.spinner("Initializing Healthcare Q&A Bot..."):
            st.session_state.bot = get_initialized_bot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    
    # Check if bot is initialized
    if st.session_state.bot is None:
        st.error("Failed to initialize the Healthcare Q&A Bot. Please check your configuration and try again.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Controls")
        
        # System status
        with st.container():
            status = st.session_state.bot.get_system_status()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", "Ready" if status['is_initialized'] else "Error")
            with col2:
                documents_count = status.get('vectorstore_stats', {}).get('total_documents', 'N/A')
                st.metric("Documents", documents_count)
        
        st.divider()
        
        # Settings
        st.subheader("Settings")
        
        use_agent = st.checkbox("Use Medical Agent", value=True, 
                               help="Use advanced agent-based processing for better results")
        
        include_sources = st.checkbox("Show Sources", value=True,
                                    help="Display source documents for answers")
        
        st.divider()
        
        # Quick actions
        st.subheader("Quick Actions")
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.bot.clear_conversation()
            st.success("Chat history cleared!")
            st.experimental_rerun()
        
        if st.button("System Status"):
            status = st.session_state.bot.get_system_status()
            st.json(status)
        
        st.divider()
        
        # Medical disclaimer
        with st.expander("Important Medical Disclaimer"):
            st.markdown(MEDICAL_DISCLAIMER)
        
        # Emergency information
        with st.expander("Emergency Information"):
            st.markdown(EMERGENCY_MESSAGE)
    
    # Main chat interface
    st.header("Ask Your Healthcare Question")
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message['content'], message['is_user'], message.get('type', 'normal'))
        
        if not message['is_user'] and message.get('sources'):
            display_sources(message['sources'])
        
        if not message['is_user'] and message.get('follow_up_questions'):
            display_follow_up_questions(message['follow_up_questions'])
    
    # Input form
    with st.form("question_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            question = st.text_area(
                "Enter your healthcare question:",
                value=st.session_state.current_question,
                height=100,
                placeholder="e.g., What are the symptoms of diabetes? How can I prevent heart disease?"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            submit_button = st.form_submit_button("Ask Question", type="primary")
    
    # Reset current question after displaying
    if st.session_state.current_question:
        st.session_state.current_question = ""
    
    # Process question
    if submit_button and question.strip():
        # Add user message to chat
        st.session_state.chat_history.append({
            'content': question,
            'is_user': True
        })
        
        # Display user message immediately
        display_chat_message(question, is_user=True)
        
        # Process the question
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.bot.ask_question(
                    question,
                    use_agent=use_agent,
                    include_sources=include_sources
                )
                
                # Determine message type based on safety classification
                message_type = "normal"
                if result.get('safety_status') == 'emergency' or result.get('safety_classification') == 'EMERGENCY':
                    message_type = "emergency"
                elif result.get('safety_status') == 'requires_consultation' or result.get('safety_classification') == 'RESTRICTED':
                    message_type = "warning"
                
                # Add bot response to chat
                bot_message = {
                    'content': result['response'],
                    'is_user': False,
                    'type': message_type,
                    'sources': result.get('sources', []) if include_sources else [],
                    'follow_up_questions': result.get('follow_up_questions', [])
                }
                
                st.session_state.chat_history.append(bot_message)
                
                # Display bot response
                display_chat_message(result['response'], is_user=False, message_type=message_type)
                
                # Display sources if available
                if include_sources and result.get('sources'):
                    display_sources(result['sources'])
                
                # Display follow-up questions
                if result.get('follow_up_questions'):
                    display_follow_up_questions(result['follow_up_questions'])
                
                # Show token usage in sidebar (debug mode)
                if settings.debug and result.get('token_usage'):
                    with st.sidebar:
                        with st.expander(" Debug Info"):
                            st.json(result.get('token_usage'))
                            if result.get('tool_calls'):
                                st.subheader("Tool Calls:")
                                for tool_call in result['tool_calls']:
                                    st.write(f"**{tool_call['tool']}:** {tool_call['tool_input']}")
                
            except Exception as e:
                error_message = f"I apologize, but I encountered an error while processing your question: {str(e)}"
                
                st.session_state.chat_history.append({
                    'content': error_message,
                    'is_user': False,
                    'type': 'warning'
                })
                
                display_chat_message(error_message, message_type="warning")
    
    # Quick example questions
    st.header(" Example Questions")
    
    example_questions = [
        "What are the symptoms of high blood pressure?",
        "How can I maintain a healthy diet?",
        "What should I know about diabetes prevention?",
        "What are the side effects of common pain relievers?",
        "How much exercise do I need per week?",
        "What are the warning signs of a heart attack?"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(example_questions):
        col = cols[i % 2]
        with col:
            if st.button(example, key=f"example_{i}"):
                st.session_state.current_question = example
                st.experimental_rerun()
    
    # Document upload section
    st.header(" Document Management")
    
    with st.expander("Upload Medical Documents"):
        uploaded_files = st.file_uploader(
            "Upload medical documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload medical guidelines, FAQs, or patient information documents"
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        # Save uploaded files temporarily
                        import tempfile
                        import os
                        
                        temp_paths = []
                        for uploaded_file in uploaded_files:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                                tmp_file.write(uploaded_file.getvalue())
                                temp_paths.append(tmp_file.name)
                        
                        # Add documents to the system
                        result = st.session_state.bot.add_documents(temp_paths)
                        
                        # Clean up temporary files
                        for temp_path in temp_paths:
                            try:
                                os.unlink(temp_path)
                            except:
                                pass
                        
                        if result['success']:
                            st.success(f" {result['message']}")
                            # Clear cache to reload bot with new documents
                            st.cache_resource.clear()
                        else:
                            st.error(f" {result['message']}")
                            
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Version:** {settings.app_version}")
    
    with col2:
        st.markdown(f"**Model:** {settings.openai_model}")
    
    with col3:
        st.markdown(f"**Vector Store:** {settings.vector_store_type}")
    
    # Important disclaimers at bottom
    st.markdown("""
    ---
    ###  Important Reminders
    
    - This AI assistant provides **general health information only**
    - **Always consult healthcare professionals** for medical advice, diagnosis, or treatment
    - **In emergencies**, call 911 or your local emergency services immediately
    - This tool **cannot replace professional medical care**
    """)


if __name__ == "__main__":
    main()
