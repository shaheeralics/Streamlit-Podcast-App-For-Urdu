"""
Professional Podcast Generator App - Production Version
Converts articles to conversational podcasts using OpenAI and ElevenLabs
Uses Streamlit secrets for API keys (no manual input required)

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
from utils.script_prompt import build_messages, validate_script_response

# Try to import audio modules with graceful fallback
_AUDIO_AVAILABLE = True
_AUDIO_ERROR = ""
try:
    from utils.audio_streamlit import synthesize_episode, get_available_voices, preview_voice
except Exception as e:
    _AUDIO_AVAILABLE = False
    _AUDIO_ERROR = str(e)
    # Define dummy functions to prevent import errors
    def synthesize_episode(*args, **kwargs):
        raise Exception(f"Audio synthesis not available: {_AUDIO_ERROR}")
    def get_available_voices(*args, **kwargs):
        return []
    def preview_voice(*args, **kwargs):
        return None

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
        background: linear-gradient(135deg, #00c9a7 0%, #ff6a88 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white !important;
        border: none;
        box-shadow: 0 2px 10px rgba(0,201,167,0.15);
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
    
    .info-box {
        background: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
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

def get_api_keys():
    """Get API keys from Streamlit secrets"""
    try:
        openai_api_key = st.secrets["openaiapi"]
        elevenlabs_api_key = st.secrets["elevenlabsapi"]
        return openai_api_key, elevenlabs_api_key
    except KeyError as e:
        st.error(f"Missing API key in secrets: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error accessing secrets: {e}")
        st.stop()

def generate_script_text_file(script_turns, article_title):
    """
    Generate a formatted text file from the podcast script
    
    Args:
        script_turns: List of script turns with speaker and text
        article_title: Title of the article
    
    Returns:
        Formatted text content for download
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PODCAST SCRIPT")
    lines.append("=" * 60)
    lines.append(f"Title: {article_title}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append("")
    
    for i, turn in enumerate(script_turns, 1):
        speaker = turn.get('speaker', 'Unknown').upper()
        text = turn.get('text', '')
        
        lines.append(f"[{i:02d}] {speaker}:")
        lines.append(f"    {text}")
        lines.append("")
    
    lines.append("=" * 60)
    lines.append("END OF SCRIPT")
    lines.append("=" * 60)
    
    return "\n".join(lines)

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
    if 'api_keys_loaded' not in st.session_state:
        st.session_state.api_keys_loaded = False

def render_header():
    """Render the main application header"""
    st.markdown("""
    <div class="main-header">
        <h1>🎙️ AI Podcast Generator</h1>
    <p>Turn long-form articles into ready-to-record conversational podcast scripts in minutes.</p>
    <small>Save research time • Consistent tone & style • Instant dialogue structure</small>
    </div>
    """, unsafe_allow_html=True)

def render_api_status(openai_api_key, elevenlabs_api_key):
    """Render configuration: full-width model row, full-width voice row."""
    st.markdown('<div class="section-header"><h3>⚙️ Core Configuration</h3></div>', unsafe_allow_html=True)

    # Row 1: Model selection
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    openai_model = st.selectbox(
        "Select OpenAI Model",
        ["gpt-5", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
        index=1 if "gpt-5" not in (st.session_state.get('selected_model') or '') else 0,
        help="Choose the model that will convert article content into a structured, conversational script."
    )
    st.session_state.selected_model = openai_model
    st.caption("Model loaded securely (keys never exposed). Higher-tier models may improve nuance.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Row 2: Voice loading
    st.markdown('<div class="info-box" style="margin-top:0.75rem;">', unsafe_allow_html=True)
    if not st.session_state.voices_loaded:
        st.write("Optionally load ElevenLabs voices for later audio synthesis (if supported).")
        if st.button("🎵 Load Voices", key="load_voices"):
            with st.spinner("Loading voices..."):
                try:
                    voices = get_available_voices(elevenlabs_api_key)
                    st.session_state.available_voices = voices
                    st.session_state.voices_loaded = True
                    st.success(f"Loaded {len(voices)} voices")
                    st.rerun()
                except Exception as e:
                    st.error(f"Voice load failed: {str(e)}")
    else:
        st.success(f"Voices loaded: {len(st.session_state.available_voices)}")
        cols = st.columns([1,1,2])
        with cols[0]:
            if st.button("🔄 Refresh", key="refresh_voices"):
                with st.spinner("Refreshing voices..."):
                    try:
                        voices = get_available_voices(elevenlabs_api_key)
                        st.session_state.available_voices = voices
                        st.success(f"Updated: {len(voices)} voices")
                    except Exception as e:
                        st.error(f"Refresh failed: {str(e)}")
        with cols[1]:
            st.write(f"Default host voices: {min(3, len(st.session_state.available_voices))} shown in selector later")
        with cols[2]:
            st.caption("You can adjust host/guest voices in the Speaker Configuration section once loaded.")
    st.caption("Skip this step if you only need the script.")
    st.markdown('</div>', unsafe_allow_html=True)

    return openai_model

def render_voice_selection():
    """Render voice selection interface"""
    # Check if audio is available
    if not _AUDIO_AVAILABLE:
        st.markdown(f'<div class="error-box">⚠️ {_AUDIO_ERROR}</div>', unsafe_allow_html=True)
        st.markdown('<div class="info-box">Audio synthesis features are currently unavailable. Script generation will still work.</div>', unsafe_allow_html=True)
        return None, None, None, None
    
    if not st.session_state.voices_loaded:
        st.markdown('<div class="info-box">🎵 Please load voices first to configure podcast speakers</div>', unsafe_allow_html=True)
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
        
        if host_voice and st.button("🔊 Preview Host Voice", key="preview_host"):
            with st.spinner("Generating preview..."):
                try:
                    _, elevenlabs_api_key = get_api_keys()
                    audio_data = preview_voice(
                        elevenlabs_api_key,
                        host_voice[1],
                        f"G'day! I'm {host_name}, your podcast host."
                    )
                    st.audio(audio_data)
                except Exception as e:
                    st.error(f"❌ Preview failed: {str(e)}")
        
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
        
        if guest_voice and st.button("🔊 Preview Guest Voice", key="preview_guest"):
            with st.spinner("Generating preview..."):
                try:
                    _, elevenlabs_api_key = get_api_keys()
                    audio_data = preview_voice(
                        elevenlabs_api_key,
                        guest_voice[1],
                        f"Hello! I'm {guest_name}, excited to be here!"
                    )
                    st.audio(audio_data)
                except Exception as e:
                    st.error(f"❌ Preview failed: {str(e)}")
        
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
    
    col1, col2, col3 = st.columns(3)
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
    
    with col3:
        st.metric("📊 Status", "Ready" if article_url else "Waiting for URL")
    
    return article_url, pause_duration, aussie_style

def render_script_generation(openai_model, article_url, host_name, guest_name, aussie_style):
    """Render script generation section"""
    st.markdown('<div class="section-header"><h3>📝 Script Generation</h3></div>', unsafe_allow_html=True)
    
    if not all([article_url, host_name, guest_name]):
        st.warning("⚠️ Please fill in all required fields above to generate script")
        return
    
    if st.button("🚀 Generate Podcast Script", key="generate_script"):
        with st.spinner("🔄 Processing article and generating script..."):
            try:
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Scrape article
                status_text.text("📖 Scraping article content...")
                progress_bar.progress(20)
                article = scrape_and_clean(article_url)
                
                # Step 2: Generate script
                status_text.text("🤖 Generating conversational script...")
                progress_bar.progress(50)
                
                openai_api_key, _ = get_api_keys()
                
                # Import OpenAI with error handling - using older version for compatibility
                try:
                    import openai
                    openai.api_key = openai_api_key
                    
                    # Test the API key with a simple request
                    test_response = openai.ChatCompletion.create(
                        model=openai_model,
                        messages=[{"role": "user", "content": "Say 'test' as JSON: {\"test\": \"success\"}"}],
                        max_tokens=50,
                        temperature=0.1
                    )
                    st.write(f"Debug: API test successful - {len(test_response.choices[0].message.content)} chars")
                    
                except ImportError:
                    st.error("❌ OpenAI package not installed. Please ensure 'openai==0.28.1' is in requirements.txt")
                    st.stop()
                except Exception as e:
                    st.error(f"❌ Error with OpenAI API: {str(e)}")
                    st.write("This could be due to:")
                    st.write("- Invalid API key")
                    st.write("- Insufficient OpenAI credits")
                    st.write("- Model not available")
                    st.write("- Rate limiting")
                    st.stop()
                
                messages = build_messages(
                    article_title=article["title"],
                    article_text=article["text"],
                    host_name=host_name,
                    guest_name=guest_name,
                    aussie=aussie_style
                )
                
                progress_bar.progress(80)
                
                # Debug: Show API key status
                # st.write(f"Debug: Using OpenAI API key ending in: ...{openai_api_key[-4:]}")
                # st.write(f"Debug: Model: {openai_model}")
                # st.write(f"Debug: Messages count: {len(messages)}")
                
                try:
                    response = openai.ChatCompletion.create(
                        model=openai_model,
                        messages=messages,
                        temperature=0.7
                    )
                    
                    # Debug: Show response structure
                    # st.write(f"Debug: Response type: {type(response)}")
                    # st.write(f"Debug: Response keys: {list(response.keys()) if hasattr(response, 'keys') else 'No keys'}")
                    
                    # Get the response content
                    response_content = response.choices[0].message.content
                    
                    # Debug: Show response length and first 100 chars
                    # st.write(f"Debug: Response length: {len(response_content) if response_content else 0}")
                    # st.write(f"Debug: First 100 chars: {response_content[:100] if response_content else 'EMPTY RESPONSE'}...")
                    # st.write(f"Debug: Last 50 chars: ...{response_content[-50:] if response_content and len(response_content) > 50 else response_content}")
                    
                    # Check for empty response
                    if not response_content or not response_content.strip():
                        raise Exception("OpenAI returned an empty response. This could be due to API limits, invalid API key, or model issues.")
                    
                    # Validate and parse the response
                    script_content = validate_script_response(response_content, host_name, guest_name)
                    st.session_state.generated_script = script_content.get("script", [])
                    st.session_state.script_generated = True
                    st.session_state.article_title = article["title"]
                    
                except Exception as api_error:
                    st.error(f"❌ OpenAI API Error: {str(api_error)}")
                    # st.write(f"Debug: API Error type: {type(api_error)}")
                    raise api_error
                
                progress_bar.progress(100)
                status_text.text("✅ Script generation complete!")
                
                st.markdown('<div class="success-box">🎉 Script generated successfully!</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Error: {str(e)}</div>', unsafe_allow_html=True)
    
    # Display generated script
    if st.session_state.script_generated and st.session_state.generated_script:
        st.subheader("📋 Generated Script Preview")
        
        # Script statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Turns", len(st.session_state.generated_script))
        with col2:
            total_words = sum(len(turn.get('text', '').split()) for turn in st.session_state.generated_script)
            st.metric("📝 Total Words", total_words)
        with col3:
            estimated_duration = total_words / 150  # ~150 words per minute
            st.metric("⏱️ Est. Duration", f"{estimated_duration:.1f} min")
        
        with st.expander("🔍 View Full Script", expanded=True):
            for i, turn in enumerate(st.session_state.generated_script, 1):
                speaker = turn.get('speaker', 'Unknown')
                text = turn.get('text', '')
                
                if speaker.lower() == 'host':
                    st.markdown(f"**🎤 {host_name} (Host):** {text}")
                else:
                    st.markdown(f"**👥 {guest_name} (Guest):** {text}")
                
                if i < len(st.session_state.generated_script):
                    st.markdown("---")

def render_audio_generation(host_voice, guest_voice, pause_duration):
    """Render audio generation section"""
    if not st.session_state.script_generated:
        return
    
    st.markdown('<div class="section-header"><h3>🎵 Audio Generation</h3></div>', unsafe_allow_html=True)
    
    # Check if audio synthesis is available
    if not _AUDIO_AVAILABLE:
        st.warning(f"⚠️ {_AUDIO_ERROR}")
        st.info("💡 The app will generate a downloadable script text file instead of audio.")
        
        # Provide script download option
        if st.button("📄 Generate Script Text File", key="generate_script_file"):
            script_text = generate_script_text_file(st.session_state.generated_script, st.session_state.article_title)
            st.download_button(
                label="📥 Download Script as Text File",
                data=script_text,
                file_name=f"podcast_script_{st.session_state.article_title.replace(' ', '_')[:50]}.txt",
                mime="text/plain",
                key="download_script"
            )
            st.success("✅ Script text file generated! Click the download button above.")
        return
    
    if not all([host_voice, guest_voice]):
        st.warning("⚠️ Please configure voices and load them first")
        return
    
    if st.button("🎧 Generate Podcast Audio", key="generate_audio"):
        with st.spinner("🎵 Generating high-quality audio... This may take a few minutes."):
            try:
                _, elevenlabs_api_key = get_api_keys()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                audio_bytes = None
                filename = None
                format_label = "MP3"
                mime_type = "audio/mp3"
                playback_format = 'audio/mp3'

                try:
                    # Primary (advanced) path using pydub utilities
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
                except Exception as advanced_err:
                    # Fallback to basic WAV synthesis that doesn't rely on pydub/audioop
                    try:
                        from utils.audio_basic import synthesize_episode_basic, BasicAudioError
                        status_text.text("⚙️ Falling back to basic WAV synthesis...")
                        progress_bar.progress(10)
                        audio_bytes, filename = synthesize_episode_basic(
                            script=st.session_state.generated_script,
                            host_voice_id=host_voice[1],
                            guest_voice_id=guest_voice[1],
                            eleven_key=elevenlabs_api_key,
                            pause_ms=pause_duration,
                            progress_callback=lambda p, s: (
                                progress_bar.progress(min(90, p)),
                                status_text.text(s)
                            )
                        )
                        if filename.lower().endswith('.mp3'):
                            format_label = "MP3 (basic concat)"
                            mime_type = "audio/mp3"
                            playback_format = 'audio/mp3'
                        else:
                            format_label = "WAV (basic)"
                            mime_type = "audio/wav"
                            playback_format = 'audio/wav'
                    except Exception as basic_err:
                        raise Exception(f"Advanced synthesis failed ({advanced_err}); basic fallback failed ({basic_err})")
                
                st.session_state.audio_generated = True
                st.session_state.audio_bytes = audio_bytes
                st.session_state.audio_filename = filename
                st.session_state.audio_mime = mime_type
                st.session_state.audio_format_label = format_label
                st.session_state.audio_playback_format = playback_format
                
                progress_bar.progress(100)
                status_text.text("✅ Audio generation complete!")
                
                st.markdown('<div class="success-box">🎉 Podcast audio generated successfully!</div>', unsafe_allow_html=True)
                
                # Audio statistics
                audio_size_mb = len(audio_bytes) / (1024 * 1024)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📁 File Size", f"{audio_size_mb:.1f} MB")
                with col2:
                    st.metric("🎵 Format", format_label)
                
            except Exception as e:
                st.markdown(f'<div class="error-box">❌ Audio generation failed: {str(e)}</div>', unsafe_allow_html=True)
    
    # Display audio player and download
    if st.session_state.audio_generated:
        st.subheader("🎧 Your Podcast")
        
        # Audio player
        playback_fmt = st.session_state.get('audio_playback_format', 'audio/mp3')
        st.audio(st.session_state.audio_bytes, format=playback_fmt)
        
        # Download section
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label=f"📥 Download Podcast {st.session_state.get('audio_format_label','Audio')}",
                data=st.session_state.audio_bytes,
                file_name=st.session_state.audio_filename,
                mime=st.session_state.get('audio_mime','audio/mp3'),
                key="download_audio"
            )
        
        with col2:
            if st.button("🔄 Generate New Podcast", key="reset_app"):
                # Reset session state for new podcast
                st.session_state.script_generated = False
                st.session_state.audio_generated = False
                st.session_state.generated_script = []
                st.success("Ready for new podcast generation!")
                st.rerun()

def check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    # Check OpenAI
    try:
        import openai
        # Test if it's the older version we need
        if not hasattr(openai, 'ChatCompletion'):
            missing_deps.append("openai (need version 0.28.1)")
    except ImportError:
        missing_deps.append("openai")
    
    # Check requests
    try:
        import requests
    except ImportError:
        missing_deps.append("requests")
    
    if missing_deps:
        st.error(f"❌ Missing required packages: {', '.join(missing_deps)}")
        st.error("Please ensure these packages are listed in requirements.txt:")
        for dep in missing_deps:
            if "openai" in dep:
                st.code("openai==0.28.1")
            else:
                st.code(dep)
        st.stop()

def main():
    """Main application function"""
    # Check dependencies first
    check_dependencies()
    
    initialize_session_state()
    render_header()
    
    # Get API keys from secrets
    openai_api_key, elevenlabs_api_key = get_api_keys()
    
    # API Status Display
    openai_model = render_api_status(openai_api_key, elevenlabs_api_key)
    
    # Voice Selection
    host_name, host_voice, guest_name, guest_voice = render_voice_selection()
    
    # Article Input
    article_url, pause_duration, aussie_style = render_article_section()
    
    # Script Generation
    render_script_generation(openai_model, article_url, host_name, guest_name, aussie_style)
    
    # Audio Generation
    render_audio_generation(host_voice, guest_voice, pause_duration)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; color: #666;">
            Built with ❤️ using Streamlit, OpenAI, and ElevenLabs<br>
            <small>🔒 Production Ready | 🚀 Auto-Configured APIs</small>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
