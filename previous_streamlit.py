"""
Professional Podcast Generator App
Converts articles to conversational podcasts using OpenAI and ElevenLabs
"""

import streamlit as st
import requests
import json
import time
import io
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import base64

# Import utility modules
from utils.scrape import scrape_and_clean
from utils.script_prompt import build_messages
from utils.audio_streamlit import synthesize_episode, get_available_voices, preview_voice

# Page configuration
st.set_page_config(
    page_title="AI Podcast Generator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for n8n.io inspired professional design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    .section-header {
        background: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .api-section {
        background: #fff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .voice-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'voices_loaded' not in st.session_state:
        st.session_state.voices_loaded = False
    if 'available_voices' not in st.session_state:
        st.session_state.available_voices = []
    if 'script_generated' not in st.session_state:
        st.session_state.script_generated = False
    if 'generated_script' not in st.session_state:
        st.session_state.generated_script = []
    if 'audio_generated' not in st.session_state:
        st.session_state.audio_generated = False

def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ AI Podcast Generator</h1>
        <p>Transform any article into an engaging conversational podcast</p>
    </div>
    """, unsafe_allow_html=True)

def render_api_section():
    """Render API configuration section"""
    st.markdown('<div class="section-header"><h3>🔑 API Configuration</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="api-section">', unsafe_allow_html=True)
        st.subheader("OpenAI API")
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Required for script generation",
            key="openai_key"
        )
        openai_model = st.selectbox(
            "Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            help="Choose the OpenAI model for script generation"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="api-section">', unsafe_allow_html=True)
        st.subheader("ElevenLabs API")
        elevenlabs_api_key = st.text_input(
            "ElevenLabs API Key",
            type="password",
            help="Required for voice synthesis",
            key="elevenlabs_key"
        )
        
        if elevenlabs_api_key and not st.session_state.voices_loaded:
            if st.button("Load Voices", key="load_voices"):
                with st.spinner("Loading available voices..."):
                    try:
                        voices = get_available_voices(elevenlabs_api_key)
                        st.session_state.available_voices = voices
                        st.session_state.voices_loaded = True
                        st.success(f"Loaded {len(voices)} voices successfully!")
                    except Exception as e:
                        st.error(f"Failed to load voices: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return openai_api_key, elevenlabs_api_key, openai_model

def render_voice_selection():
    """Render voice selection interface"""
    if not st.session_state.voices_loaded:
        st.info("🎵 Load voices first to configure podcast speakers")
        return None, None, None, None
    
    st.markdown('<div class="section-header"><h3>🎭 Speaker Configuration</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    voice_options = [(v['name'], v['voice_id']) for v in st.session_state.available_voices]
    
    with col1:
        st.markdown('<div class="voice-card">', unsafe_allow_html=True)
        st.subheader("🎤 Host")
        host_name = st.text_input("Host Name", value="Alex", key="host_name")
        host_voice = st.selectbox(
            "Host Voice",
            voice_options,
            format_func=lambda x: x[0],
            key="host_voice"
        )
        
        if host_voice and st.button("Preview Host Voice", key="preview_host"):
            with st.spinner("Generating preview..."):
                try:
                    audio_url = preview_voice(
                        st.session_state.elevenlabs_key,
                        host_voice[1],
                        f"G'day! I'm {host_name}, your podcast host."
                    )
                    st.audio(audio_url)
                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="voice-card">', unsafe_allow_html=True)
        st.subheader("👥 Guest")
        guest_name = st.text_input("Guest Name", value="Sarah", key="guest_name")
        guest_voice = st.selectbox(
            "Guest Voice",
            voice_options,
            format_func=lambda x: x[0],
            key="guest_voice"
        )
        
        if guest_voice and st.button("Preview Guest Voice", key="preview_guest"):
            with st.spinner("Generating preview..."):
                try:
                    audio_url = preview_voice(
                        st.session_state.elevenlabs_key,
                        guest_voice[1],
                        f"Hello! I'm {guest_name}, excited to be here!"
                    )
                    st.audio(audio_url)
                except Exception as e:
                    st.error(f"Preview failed: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return host_name, host_voice, guest_name, guest_voice

def render_article_section():
    """Render article input section"""
    st.markdown('<div class="section-header"><h3>📰 Article Input</h3></div>', unsafe_allow_html=True)
    
    article_url = st.text_input(
        "Article URL",
        placeholder="https://example.com/article",
        help="Paste the URL of the article you want to convert to a podcast"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        pause_duration = st.slider(
            "Pause between speakers (ms)",
            min_value=200,
            max_value=2000,
            value=800,
            step=100,
            help="Duration of silence between speaker turns"
        )
    
    with col2:
        aussie_style = st.checkbox(
            "Australian Style",
            value=True,
            help="Generate script in Australian conversational style"
        )
    
    return article_url, pause_duration, aussie_style

def render_script_generation(openai_api_key, openai_model, article_url, host_name, guest_name, aussie_style):
    """Render script generation section"""
    st.markdown('<div class="section-header"><h3>📝 Script Generation</h3></div>', unsafe_allow_html=True)
    
    if not all([openai_api_key, article_url, host_name, guest_name]):
        st.warning("Please fill in all required fields above to generate script")
        return
    
    if st.button("🚀 Generate Podcast Script", key="generate_script"):
        with st.spinner("Scraping article and generating script..."):
            try:
                # Step 1: Scrape article
                st.info("📖 Scraping article content...")
                article = scrape_and_clean(article_url)
                
                # Step 2: Generate script
                st.info("🤖 Generating conversational script...")
                
                # Import OpenAI here to avoid issues if not installed
                from openai import OpenAI
                client = OpenAI(api_key=openai_api_key)
                
                messages = build_messages(
                    article_title=article["title"],
                    article_text=article["text"],
                    host_name=host_name,
                    guest_name=guest_name,
                    aussie=aussie_style
                )
                
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=messages,
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                
                script_content = json.loads(response.choices[0].message.content)
                st.session_state.generated_script = script_content.get("script", [])
                st.session_state.script_generated = True
                
                st.markdown('<div class="success-box">✅ Script generated successfully!</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
    
    # Display generated script
    if st.session_state.script_generated and st.session_state.generated_script:
        st.subheader("Generated Script Preview")
        
        with st.expander("View Full Script", expanded=True):
            for i, turn in enumerate(st.session_state.generated_script, 1):
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                
                if speaker.lower() == 'host':
                    st.markdown(f"**🎤 {host_name} (Host):** {text}")
                else:
                    st.markdown(f"**👥 {guest_name} (Guest):** {text}")
                
                if i < len(st.session_state.generated_script):
                    st.markdown("---")

def render_audio_generation(elevenlabs_api_key, host_voice, guest_voice, pause_duration):
    """Render audio generation section"""
    if not st.session_state.script_generated:
        return
    
    st.markdown('<div class="section-header"><h3>🎵 Audio Generation</h3></div>', unsafe_allow_html=True)
    
    if not all([elevenlabs_api_key, host_voice, guest_voice]):
        st.warning("Please configure voices and load them first")
        return
    
    if st.button("🎧 Generate Podcast Audio", key="generate_audio"):
        with st.spinner("Generating audio... This may take a few minutes."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Generate audio with progress updates
                audio_bytes, filename = synthesize_episode(
                    script=st.session_state.generated_script,
                    pause_ms=pause_duration,
                    host_voice_id=host_voice[1],
                    guest_voice_id=guest_voice[1],
                    eleven_key=elevenlabs_api_key,
                    progress_callback=lambda p, s: (
                        progress_bar.progress(p),
                        status_text.text(s)
                    )
                )
                
                st.session_state.audio_generated = True
                st.session_state.audio_bytes = audio_bytes
                st.session_state.audio_filename = filename
                
                progress_bar.progress(100)
                status_text.text("✅ Audio generation complete!")
                
                st.markdown('<div class="success-box">🎉 Podcast audio generated successfully!</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Audio generation failed: {str(e)}</div>', unsafe_allow_html=True)
    
    # Display audio player and download
    if st.session_state.audio_generated:
        st.subheader("🎧 Your Podcast")
        
        # Audio player
        st.audio(st.session_state.audio_bytes, format='audio/mp3')
        
        # Download button
        st.download_button(
            label="📥 Download Podcast MP3",
            data=st.session_state.audio_bytes,
            file_name=st.session_state.audio_filename,
            mime="audio/mp3",
            key="download_audio"
        )

def main():
    """Main application function"""
    initialize_session_state()
    render_header()
    
    # API Configuration
    openai_api_key, elevenlabs_api_key, openai_model = render_api_section()
    
    # Voice Selection
    host_name, host_voice, guest_name, guest_voice = render_voice_selection()
    
    # Article Input
    article_url, pause_duration, aussie_style = render_article_section()
    
    # Script Generation
    render_script_generation(openai_api_key, openai_model, article_url, host_name, guest_name, aussie_style)
    
    # Audio Generation
    render_audio_generation(elevenlabs_api_key, host_voice, guest_voice, pause_duration)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, OpenAI, and ElevenLabs")

if __name__ == "__main__":
    main()
