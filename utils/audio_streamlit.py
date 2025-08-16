"""
Audio synthesis utilities for Streamlit app using ElevenLabs.
Enhanced version with progress callbacks and better error handling.
"""

import requests
import time
from io import BytesIO
from typing import List, Dict, Tuple, Optional, Callable
import streamlit as st

# Check for audio dependencies
_AUDIO_DISABLED = False
_AUDIO_IMPORT_ERROR = ""
try:
    # Try to import audioop first (missing in Python 3.13+)
    try:
        import audioop
    except ImportError:
        # audioop not available, pydub will have limited functionality
        pass
    
    from pydub import AudioSegment
    from pydub.generators import Silence
except Exception as e:
    _AUDIO_DISABLED = True
    _AUDIO_IMPORT_ERROR = f"Audio synthesis not available: {str(e)}"

def get_available_voices(elevenlabs_api_key: str) -> List[Dict]:
    """
    Fetch available voices from ElevenLabs API
    
    Args:
        elevenlabs_api_key: ElevenLabs API key
        
    Returns:
        List of voice dictionaries with name, voice_id, and other metadata
    """
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers={
                "xi-api-key": elevenlabs_api_key,
                "accept": "application/json"
            },
            timeout=10
        )
        response.raise_for_status()
        
        voices_data = response.json()
        return voices_data.get("voices", [])
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch voices: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing voices: {str(e)}")

def preview_voice(elevenlabs_api_key: str, voice_id: str, text: str = None) -> str:
    """
    Generate a voice preview using ElevenLabs TTS
    
    Args:
        elevenlabs_api_key: ElevenLabs API key
        voice_id: Voice ID to preview
        text: Text to synthesize (optional)
        
    Returns:
        Audio data as bytes that can be used with st.audio()
    """
    if text is None:
        text = "Hello! This is a voice preview for the podcast generator."
    
    try:
        tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        
        response = requests.post(
            tts_url,
            headers={
                "xi-api-key": elevenlabs_api_key,
                "accept": "audio/mpeg",
                "content-type": "application/json",
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.4,
                    "similarity_boost": 0.8,
                    "style": 0.2,
                    "use_speaker_boost": True
                }
            },
            timeout=30
        )
        response.raise_for_status()
        
        return response.content
        
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to generate voice preview: {str(e)}")
    except Exception as e:
        raise Exception(f"Error in voice preview: {str(e)}")

def _synthesize_single_line(
    text: str, 
    voice_id: str, 
    elevenlabs_api_key: str,
    max_retries: int = 3
) -> bytes:
    """
    Synthesize a single line of text to audio
    
    Args:
        text: Text to synthesize
        voice_id: ElevenLabs voice ID
        elevenlabs_api_key: ElevenLabs API key
        max_retries: Maximum number of retry attempts
        
    Returns:
        Audio data as bytes
    """
    tts_url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                tts_url,
                headers={
                    "xi-api-key": elevenlabs_api_key,
                    "accept": "audio/mpeg",
                    "content-type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": {
                        "stability": 0.4,
                        "similarity_boost": 0.8,
                        "style": 0.2,
                        "use_speaker_boost": True
                    }
                },
                timeout=60
            )
            response.raise_for_status()
            return response.content
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to synthesize audio after {max_retries} attempts: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff
    
    raise Exception("Unexpected error in audio synthesis")

def synthesize_episode(
    script: List[Dict],
    pause_ms: int,
    host_voice_id: str,
    guest_voice_id: str,
    eleven_key: str,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[bytes, str]:
    """
    Synthesize a complete podcast episode from script
    
    Args:
        script: List of script turns with speaker and text
        pause_ms: Pause duration between speakers in milliseconds
        host_voice_id: Voice ID for host
        guest_voice_id: Voice ID for guest
        eleven_key: ElevenLabs API key
        progress_callback: Optional callback for progress updates (progress_percent, status_message)
        
    Returns:
        Tuple of (audio_bytes, filename)
    """
    if _AUDIO_DISABLED:
        raise Exception(f"Audio synthesis unavailable: {_AUDIO_IMPORT_ERROR}")
    
    if not script:
        raise Exception("No script provided for audio synthesis")
    
    # Initialize progress
    total_turns = len(script)
    if progress_callback:
        progress_callback(0, "Starting audio synthesis...")
    
    # Create silence for pauses
    pause_audio = Silence(duration=max(0, pause_ms))
    
    # Initialize the final audio track
    final_audio = AudioSegment.empty()
    
    # Process each turn in the script
    for i, turn in enumerate(script):
        speaker = turn.get("speaker", "").lower()
        text = turn.get("text", "").strip()
        
        if not text:
            continue
        
        # Determine voice ID based on speaker
        voice_id = host_voice_id if speaker == "host" else guest_voice_id
        
        # Update progress
        progress_percent = int((i / total_turns) * 90)  # Reserve 10% for final processing
        speaker_name = "Host" if speaker == "host" else "Guest"
        
        if progress_callback:
            progress_callback(progress_percent, f"Synthesizing {speaker_name} line {i+1}/{total_turns}...")
        
        try:
            # Synthesize the audio for this line
            audio_data = _synthesize_single_line(text, voice_id, eleven_key)
            
            # Convert to AudioSegment
            audio_segment = AudioSegment.from_file(BytesIO(audio_data), format="mp3")
            
            # Add to final audio with pause
            final_audio += audio_segment
            
            # Add pause after each line (except the last one)
            if i < total_turns - 1:
                final_audio += pause_audio
            
            # Rate limiting to avoid API limits
            time.sleep(0.3)
            
        except Exception as e:
            error_msg = f"Failed to synthesize line {i+1}: {str(e)}"
            if progress_callback:
                progress_callback(progress_percent, error_msg)
            raise Exception(error_msg)
    
    # Final processing
    if progress_callback:
        progress_callback(95, "Finalizing audio file...")
    
    # Export to MP3
    output_buffer = BytesIO()
    final_audio.export(
        output_buffer,
        format="mp3",
        bitrate="192k",
        parameters=["-ar", "44100"]  # Ensure consistent sample rate
    )
    output_buffer.seek(0)
    
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"podcast_episode_{timestamp}.mp3"
    
    audio_bytes = output_buffer.read()
    
    if progress_callback:
        progress_callback(100, f"Audio synthesis complete! Generated {len(audio_bytes)} bytes")
    
    return audio_bytes, filename
    
    if not script:
        raise Exception("No script provided for audio synthesis")
    
    # Initialize progress
    total_turns = len(script)
    if progress_callback:
        progress_callback(0, "Starting audio synthesis...")
    
    # Create silence for pauses
    pause_audio = Silence(duration=max(0, pause_ms))
    
    # Initialize the final audio track
    final_audio = AudioSegment.empty()
    
    # Process each turn in the script
    for i, turn in enumerate(script):
        speaker = turn.get("speaker", "").lower()
        text = turn.get("text", "").strip()
        
        if not text:
            continue
        
        # Determine voice ID based on speaker
        voice_id = host_voice_id if speaker == "host" else guest_voice_id
        
        # Update progress
        progress_percent = int((i / total_turns) * 90)  # Reserve 10% for final processing
        speaker_name = "Host" if speaker == "host" else "Guest"
        
        if progress_callback:
            progress_callback(progress_percent, f"Synthesizing {speaker_name} line {i+1}/{total_turns}...")
        
        try:
            # Synthesize the audio for this line
            audio_data = _synthesize_single_line(text, voice_id, eleven_key)
            
            # Convert to AudioSegment
            audio_segment = AudioSegment.from_file(BytesIO(audio_data), format="mp3")
            
            # Add to final audio with pause
            final_audio += audio_segment
            
            # Add pause after each line (except the last one)
            if i < total_turns - 1:
                final_audio += pause_audio
            
            # Rate limiting to avoid API limits
            time.sleep(0.3)
            
        except Exception as e:
            error_msg = f"Failed to synthesize line {i+1}: {str(e)}"
            if progress_callback:
                progress_callback(progress_percent, error_msg)
            raise Exception(error_msg)
    
    # Final processing
    if progress_callback:
        progress_callback(95, "Finalizing audio file...")
    
    # Export to MP3
    output_buffer = BytesIO()
    final_audio.export(
        output_buffer,
        format="mp3",
        bitrate="192k",
        parameters=["-ar", "44100"]  # Ensure consistent sample rate
    )
    output_buffer.seek(0)
    
    # Generate filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"podcast_episode_{timestamp}.mp3"
    
    audio_bytes = output_buffer.read()
    
    if progress_callback:
        progress_callback(100, f"Audio synthesis complete! Generated {len(audio_bytes)} bytes")
    
    return audio_bytes, filename

def test_audio_setup() -> bool:
    """
    Test if audio synthesis dependencies are available
    
    Returns:
        True if audio synthesis is available, False otherwise
    """
    return not _AUDIO_DISABLED

def get_audio_error() -> Optional[str]:
    """
    Get the audio setup error message if any
    
    Returns:
        Error message if audio is disabled, None otherwise
    """
    return _AUDIO_IMPORT_ERROR if _AUDIO_DISABLED else None
