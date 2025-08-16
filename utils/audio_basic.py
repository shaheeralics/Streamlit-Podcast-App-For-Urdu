"""Basic ElevenLabs multi-turn audio synthesis without pydub.
Builds a single WAV file by concatenating per-turn WAV responses.
Works even when audioop/pydub are unavailable (Python 3.13+).

Limitations:
- Relies on ElevenLabs returning standard 44-byte PCM WAV headers.
- Inserts silence by padding zeroed PCM frames (configurable).
- Does NOT perform volume normalization or cross-fades.

API:
    synthesize_episode_basic(script, host_voice_id, guest_voice_id, eleven_key,
                             pause_ms=300, progress_callback=None) -> (bytes, filename)
"""
from __future__ import annotations
import io
import struct
import time
from typing import List, Dict, Callable, Optional, Tuple
import requests
from datetime import datetime

ELEVEN_API_TTS = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
MODEL_ID = "eleven_multilingual_v2"
DEFAULT_VOICE_SETTINGS = {"stability": 0.5, "similarity_boost": 0.75}

class BasicAudioError(Exception):
    pass

def _extract_wav_pcm(payload: bytes) -> Tuple[bytes, int, int, int]:
    """Extract raw PCM data and format info from a simple PCM WAV buffer.

    Returns: (pcm_bytes, sample_rate, channels, bits_per_sample)
    """
    if len(payload) < 44 or payload[0:4] != b'RIFF' or payload[8:12] != b'WAVE':
        # Provide first 12 bytes hex for diagnostics
        preview = payload[:12].hex()
        raise BasicAudioError(f"Unexpected WAV header from ElevenLabs (first bytes: {preview})")

    # Parse fmt chunk (assume at 12)
    # Offset 22: channels (2 bytes), 24: sample rate (4), 34: bits per sample (2)
    channels = struct.unpack('<H', payload[22:24])[0]
    sample_rate = struct.unpack('<I', payload[24:28])[0]
    bits_per_sample = struct.unpack('<H', payload[34:36])[0]

    # Find data chunk (may not be at fixed offset if there are extra chunks)
    offset = 12
    data_chunk_start = None
    while offset < len(payload) - 8:
        chunk_id = payload[offset:offset+4]
        chunk_size = struct.unpack('<I', payload[offset+4:offset+8])[0]
        if chunk_id == b'data':
            data_chunk_start = offset + 8
            pcm = payload[data_chunk_start:data_chunk_start+chunk_size]
            break
        offset += 8 + chunk_size

    if data_chunk_start is None:
        raise BasicAudioError("No data chunk found in WAV")

    return pcm, sample_rate, channels, bits_per_sample

def _build_wav(pcm: bytes, sample_rate: int, channels: int, bits_per_sample: int) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = len(pcm)
    riff_chunk_size = 36 + data_size
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', riff_chunk_size, b'WAVE',
        b'fmt ', 16, 1, channels, sample_rate,
        byte_rate, block_align, bits_per_sample,
        b'data', data_size
    )
    return header + pcm

def _tts_turn(text: str, voice_id: str, api_key: str, want_wav: bool = True) -> bytes:
    """Request a single TTS turn. Try WAV if requested; fallback handled by caller."""
    headers = {
        'xi-api-key': api_key,
        'accept': 'audio/wav' if want_wav else 'audio/mpeg',
        'content-type': 'application/json'
    }
    json_payload = {
        'text': text,
        'model_id': MODEL_ID,
        'voice_settings': DEFAULT_VOICE_SETTINGS,
        # Hint desired output format (not always required but explicit)
        'output_format': 'wav' if want_wav else 'mp3_44100_128'
    }
    r = requests.post(ELEVEN_API_TTS.format(voice_id=voice_id), headers=headers, json=json_payload, timeout=90)
    if r.status_code != 200:
        raise BasicAudioError(f"ElevenLabs TTS failed ({r.status_code}): {r.text[:160]}")
    return r.content

def synthesize_episode_basic(
    script: List[Dict[str, str]],
    host_voice_id: str,
    guest_voice_id: str,
    eleven_key: str,
    pause_ms: int = 300,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    prefer_wav: bool = True,
) -> Tuple[bytes, str]:
    if not script:
        raise BasicAudioError("Empty script")

    # We support two aggregation modes:
    # 1. WAV merge (uncompressed) when WAV returned
    # 2. MP3 concatenation fallback when only MP3 available
    pcm_chunks: List[bytes] = []
    mp3_segments: List[bytes] = []
    sr = channels = bps = None
    using_mp3 = False

    silence_frames = None  # computed after first WAV segment

    total_turns = len(script)
    for idx, turn in enumerate(script, 1):
        speaker = turn.get('speaker', 'host')
        text = turn.get('text', '').strip()
        if not text:
            continue
        voice_id = host_voice_id if speaker == 'host' else guest_voice_id

        if progress_callback:
            progress_callback(int(((idx-1)/total_turns)*90), f"Synthesizing turn {idx}/{total_turns} ({speaker})")

        # Attempt preferred WAV first (if still in wav mode)
        if prefer_wav and not using_mp3:
            try:
                wav_bytes = _tts_turn(text, voice_id, eleven_key, want_wav=True)
                pcm, srate, ch, bits = _extract_wav_pcm(wav_bytes)
                if sr is None:
                    sr, channels, bps = srate, ch, bits
                    bytes_per_sample = bps // 8
                    frame_size = channels * bytes_per_sample
                    silence_samples = int(sr * (pause_ms / 1000.0))
                    silence_frames = b'\x00' * (silence_samples * frame_size)
                else:
                    if (srate, ch, bits) != (sr, channels, bps):
                        raise BasicAudioError("Inconsistent audio format returned across turns")
                pcm_chunks.append(pcm)
                if idx != total_turns:
                    pcm_chunks.append(silence_frames)
                continue
            except BasicAudioError:
                # Switch to MP3 fallback
                using_mp3 = True
                if progress_callback:
                    progress_callback(int(((idx-1)/total_turns)*90), f"Switching to MP3 fallback (turn {idx})")

        # MP3 path
        mp3_bytes = _tts_turn(text, voice_id, eleven_key, want_wav=False)
        # Basic validation: check for MP3 frame or ID3
        if not (mp3_bytes.startswith(b'ID3') or mp3_bytes[:2] in (b'\xff\xfb', b'\xff\xf3', b'\xff\xf2')):
            preview = mp3_bytes[:16].hex()
            raise BasicAudioError(f"Unexpected MP3 fallback bytes (first 16: {preview})")
        mp3_segments.append(mp3_bytes)

    if using_mp3:
        if progress_callback:
            progress_callback(95, "Merging MP3 segments")
        merged = b''.join(mp3_segments)
        filename = f"podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        if progress_callback:
            progress_callback(100, "Done")
        return merged, filename
    else:
        merged_pcm = b''.join(pcm_chunks)
        final_wav = _build_wav(merged_pcm, sr, channels, bps)
        if progress_callback:
            progress_callback(95, "Finalizing WAV file")
        filename = f"podcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        if progress_callback:
            progress_callback(100, "Done")
        return final_wav, filename

__all__ = ["synthesize_episode_basic", "BasicAudioError"]
