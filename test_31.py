"""
============================================================
  🤖 Speech-to-Speech AI Chatbot — Powered by OpenAI
============================================================
  Stack:
    STT  → OpenAI Whisper (whisper-1)
    LLM  → OpenAI Chat model (streamed, sentence-by-sentence)
    TTS  → Microsoft Edge TTS (edge-tts, streamed + cached)
    Offline fallback → espeak (no internet required)

  Language Support:
    → Speak English → RoboBot replies & speaks in English
    → Speak Hindi   → RoboBot replies & speaks in Hindi
    → Switches instantly every message — no confusion

  Language Detection (3-layer):
    1. Whisper language tag  (fast, sometimes wrong)
    2. Script scan of transcript  (ground truth — never lies)
    3. Default → English

  Optimisations in this version:
    1. Streaming input  — speech is captured continuously and each
       sentence-sized chunk (separated by a short mid-turn pause) is
       transcribed in the background WHILE the user keeps talking.
    2. Streaming output — the LLM reply is streamed token-by-token;
       each finished sentence is sent to TTS and played immediately
       while the next sentence is still being generated/synthesised.
    3. Prompt caching   — (a) exact-match reply cache so a repeated
       question skips the LLM call entirely, (b) a disk-backed audio
       cache so a phrase is never re-synthesised twice, and (c) the
       system+history prefix is kept stable so OpenAI's own automatic
       prompt caching (models ≥1024 prompt tokens) can kick in.
    4. Empty strings are filtered out at every hand-off point before
       they ever reach an API (STT segment, combined transcript,
       individual streamed sentences).
    5/6/7. Layered, offline-safe error handling — see `handle_error()`.

  State Machine:
    IDLE ──(wake word)──► LISTENING ──(speech)──► THINKING ──► SPEAKING
      ▲                        │                                  │
      └──────(10s silence)─────┘◄─────────────────────────────────┘

  Wake word: "Hello" / "Hey"
============================================================
"""

import os
import io
import re
import time
import queue
import socket
import hashlib
import threading
import subprocess
from enum import Enum
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
import soundfile as sf
import edge_tts
import pygame
import requests
from dotenv import load_dotenv

from openai import (
    OpenAI,
    APIError,          # base class — catches every OpenAI SDK error
                        # (APIConnectionError, APITimeoutError, RateLimitError,
                        #  AuthenticationError, APIStatusError, InternalServerError...)
)

load_dotenv()

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

NYLA_AUTH_URL       = os.getenv("NYLA_AUTH_URL", "http://ai.salesninjacrm.com/org/auth-robot")
NYLA_ASK_URL        = os.getenv("NYLA_ASK_URL", "http://ai.salesninjacrm.com/nyla-report/ask")
NYLA_API_KEY        = os.getenv("NYLA_API_KEY")
NYLA_ENCRYPTED_DATA = os.getenv("NYLA_ENCRYPTED_DATA")

STT_MODEL  = "whisper-1"
CHAT_MODEL = "gpt-4o"   # unused for chat — replies come from Nyla /ask; Whisper still uses OpenAI.

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16000
CHANNELS    = 1
MAX_TOKENS  = 200

# ── VAD tuning ─────────────────────────────────────────────
ENERGY_THRESHOLD     = 0.10
# Was 1.2s. Every turn pays this in full before STT even starts, so it is
# one of the biggest single latency knobs. 0.8s is a common sweet spot for
# conversational assistants -- if users start getting cut off mid-thought,
# raise it back up in 0.1s steps; if RoboBot still feels slow to react and
# nobody is getting cut off, it can go a little lower (try 0.6-0.7s).
SILENCE_AFTER_SPEECH = 1.0
# Was 0.45s. Governs how quickly a mid-turn chunk gets shipped to Whisper
# in the background. Lower = STT gets a head start sooner, at the cost of
# slightly choppier segment boundaries. 0.35s tested fine; recalibrate if
# segments start splitting mid-word too often.
SENTENCE_PAUSE       = 0.35
# Adaptive silence detection: once at least one mid-turn segment has
# already been flushed to the STT pool (see `any_mid_flush` below), the
# background transcription for the bulk of the utterance has a head
# start, so we don't need to wait as long to conclude the turn -- only
# the short trailing remainder is still on the critical path.
SILENCE_AFTER_SPEECH_FOLLOWUP = 0.6
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.1

IDLE_TIMEOUT      = 10.0
IDLE_POLL_TIMEOUT = 30.0

# ── Perf / debug toggles ───────────────────────────────────
# Print the per-turn latency breakdown (Speech End → STT → GPT First
# Token → Sentence Complete → TTS Start → Playback Started → Total).
# Safe to leave on in production; it's just terminal output.
PRINT_LATENCY_TIMINGS = True
# Stream the FIRST spoken sentence of every turn straight into `mpg123`'s
# stdin as edge-tts produces it, instead of waiting for the whole
# sentence to finish synthesising before playback starts. This is the
# single biggest win for "time to first sound" because every later
# sentence already overlaps its synthesis with the previous sentence's
# playback (see StreamingSpeaker) -- only sentence #1 has nothing to
# overlap with. Requires `mpg123` (sudo apt install mpg123). If it's
# missing, or streaming fails for any reason, RoboBot automatically
# falls back to the original synth-then-play path, so this is safe to
# leave on even if mpg123 isn't installed yet.
STREAM_TTS_PLAYBACK = True

WAKE_WORDS = ["hello", "hey"]

# NOTE: SYSTEM_EN / SYSTEM_HI / MAX_TOKENS are no longer used now that
# Nyla handles all chat replies (Nyla has its own persona/behavior
# server-side). Left in place in case you ever want a GPT-4o fallback.
SYSTEM_EN = (
    "Your name is RoboBot. You are the helpful AI assistant . "
    "Keep responses concise and conversational."
    "No bullet points or markdown."
    "you are created by Robotwala."
    "dont use bullet points."
    "answer only in 2 to 3 sentences"
    "If the user's language is Hindi, always respond in Hindi written in Devanagari script, "
    "even if the user's input is written in Urdu (Perso-Arabic) script or Roman Hindi."
    "Never respond in the Urdu script."
)

SYSTEM_HI = (
    "Aapka naam RoboBot hai. Aap helpful AI assistant hain. "
    "Apne uttar chhote aur batcheet ke andaz mein rakhein. "
    "Koi bullet points ya markdown nahi."
    "tumhein robotwala ne banaya hein."
    "bullet points ka use nahi karna he."
    "sirf 2 se 3 sentence me jawab dena he"
    "Agar user ki language Hindi hai, to hamesha Hindi mein sirf Devanagari script ka use karke ""reply do. Agar user Hindi ko Roman Hindi ya Urdu (Perso-Arabic) script mein likhe, tab bhi ""hamesha Hindi ki Devanagari script mein hi jawab do. Kabhi bhi Urdu (Perso-Arabic) script ""mein reply mat dena."
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।\n])\s+")

# ── Fixed offline error strings (never sent to any network TTS) ──
MSG_NO_INTERNET = "Can't connect to the internet."
MSG_NO_SERVER   = "Can't connect to the server."
MSG_TRY_AGAIN   = "Please try again."

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ──────────────────────────────────────────────
#  LATENCY INSTRUMENTATION
# ──────────────────────────────────────────────
# One shared dict per turn: speech_end → stt_complete → gpt_start →
# gpt_first_token → first_sentence → tts_start → playback_start.
# Reset at the start of every real (non-idle) listening turn.
TURN_TIMINGS: dict = {}


def _reset_timings():
    TURN_TIMINGS.clear()


def _mark(stage: str):
    if stage not in TURN_TIMINGS:   # first write wins (e.g. first token, first sentence)
        TURN_TIMINGS[stage] = time.time()


def print_turn_timings():
    """Prints the requested per-turn latency breakdown, skipping any stage
    that didn't fire this turn (e.g. GPT stages on a reply-cache hit)."""
    t = TURN_TIMINGS
    if "speech_end" not in t:
        return

    def gap(a, b):
        return int((t[b] - t[a]) * 1000) if a in t and b in t else None

    rows = [
        ("Speech End",         0),
        ("STT",                gap("speech_end", "stt_complete")),
        ("GPT First Token",    gap("gpt_start", "gpt_first_token")),
        ("Sentence Complete",  gap("gpt_first_token", "first_sentence")),
        ("TTS Start",          gap("first_sentence", "tts_start")),
        ("Playback Started",   gap("tts_start", "playback_start")),
    ]

    print("\n   ── Latency breakdown ──────────────")
    for label, ms in rows:
        shown = f"{ms} ms" if ms is not None else "n/a (skipped this turn)"
        print(f"   {label:<20} {shown}")
    if "playback_start" in t:
        total = int((t["playback_start"] - t["speech_end"]) * 1000)
        print(f"   {'Total Latency':<20} {total} ms")
    print("   ────────────────────────────────────\n")


# ──────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"

STATE_LABEL = {
    State.IDLE:      "😴 IDLE",
    State.LISTENING: "👂 LISTENING",
    State.THINKING:  "🤔 THINKING",
    State.SPEAKING:  "🔊 SPEAKING",
}


def show_state(state: State, note: str = ""):
    """Single always-visible line so the current state is obvious in the terminal."""
    line = f"\n[{STATE_LABEL[state]}]"
    if note:
        line += f" {note}"
    print(line)


# ──────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY, timeout=15.0, max_retries=1)

# Separate history per language so the model never sees cross-language
# context and stays in the right language naturally. Kept as a STABLE
# ordered prefix (system + history) on every call so OpenAI's automatic
# prompt caching can match the repeated prefix.
history: dict = {"en": [], "hi": []}

# Exact-match reply cache: skips the LLM entirely for a repeated question.
reply_cache: dict = {}

# In-memory (voice, text) → path cache in front of the on-disk audio cache.
# The disk cache still does the heavy lifting (survives restarts), this
# just avoids a redundant os.path.exists() stat() call for phrases that
# repeat often within one run (greetings, idle prompts, common replies).
_session_audio_cache: dict = {}

pygame.mixer.init()

# Background workers: one pool for STT segments, one for TTS synthesis.
stt_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="stt")
tts_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")


# ──────────────────────────────────────────────
#  CONNECTIVITY / ERROR HANDLING  (features 5, 6, 7)
# ──────────────────────────────────────────────

def is_internet_available(timeout: float = 2.0) -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def speak_offline(text: str):
    """
    Offline, dependency-free TTS via the local `espeak` binary.
    Used ONLY for error announcements, since it never touches the
    network — unlike edge-tts, which would itself fail if the
    internet or the API is the actual problem.
    """
    print(f"   🔇 (offline) {text}")
    try:
        subprocess.run(
            ["espeak", text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        print("   ⚠️  espeak is not installed — install it for spoken error messages "
              "(e.g. `sudo apt install espeak`).")


def handle_error(e: Exception, where: str):
    """
    Central error classifier. Order matters:
      1. No internet at all               → "Can't connect to the internet."
      2. Internet is fine but the OpenAI
         API call itself failed           → "Can't connect to the server."
      3. Anything else                    → "Please try again."
    """
    print(f"\n❌ Error in {where}: {type(e).__name__}: {e}")

    if not is_internet_available():
        speak_offline(MSG_NO_INTERNET)
        return

    if isinstance(e, (APIError, requests.RequestException, RuntimeError)):
        speak_offline(MSG_NO_SERVER)
        return

    speak_offline(MSG_TRY_AGAIN)


# ──────────────────────────────────────────────
#  AUDIO CACHE  (feature 2 — part of "prompt caching")
# ──────────────────────────────────────────────

def _cache_path(text: str, voice: str) -> str:
    key = hashlib.sha256(f"{voice}::{text}".encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{key}.mp3")


async def _tts_to_file(text: str, path: str, voice: str):
    await edge_tts.Communicate(text, voice=voice).save(path)


def synthesize(text: str, voice: str) -> str:
    """
    Returns a path to an mp3 for `text`. Disk-cached by (voice, text) hash
    so any phrase — greetings, idle prompts, or a repeated LLM sentence —
    is only ever sent to edge-tts once.
    """
    session_key = (voice, text)
    cached = _session_audio_cache.get(session_key)
    if cached:
        return cached

    path = _cache_path(text, voice)
    if os.path.exists(path):
        _session_audio_cache[session_key] = path
        return path

    import asyncio
    tmp_path = path + ".tmp"
    asyncio.run(_tts_to_file(text, tmp_path, voice))
    os.replace(tmp_path, path)
    _session_audio_cache[session_key] = path
    return path


def play(path: str):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(50)
    pygame.mixer.music.unload()


# ──────────────────────────────────────────────
#  VOICE SELECTION
# ──────────────────────────────────────────────

def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


async def _tts_stream_to_mpg123(text: str, voice: str, tmp_path: str):
    """
    Streams edge-tts audio chunks straight into `mpg123`'s stdin as they
    arrive, AND writes the same bytes to `tmp_path` so the normal disk
    cache still gets populated exactly as before. Raises on any problem
    (mpg123 missing, stream error, ...) — caller is responsible for
    falling back to the non-streaming path.
    """
    proc = subprocess.Popen(
        ["mpg123", "-q", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    first_chunk = True
    try:
        with open(tmp_path, "wb") as f:
            communicate = edge_tts.Communicate(text, voice=voice)
            async for chunk in communicate.stream():
                if chunk.get("type") != "audio":
                    continue
                data = chunk["data"]
                f.write(data)
                proc.stdin.write(data)
                if first_chunk:
                    proc.stdin.flush()
                    _mark("playback_start")   # audio is now reaching the speaker
                    first_chunk = False
    finally:
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        proc.wait()

    if first_chunk:
        # No audio chunks at all — treat as a failure so the caller falls back.
        raise RuntimeError("edge-tts produced no audio chunks")


def _synth_and_play_streaming(text: str, voice: str):
    """
    Used ONLY for the first spoken sentence of a turn (see StreamingSpeaker).
    Starts audio playing as soon as the first chunk of synthesised speech
    exists, instead of waiting for the whole sentence to finish. Falls back
    to the plain synthesize()+play() path if mpg123 isn't installed or
    streaming fails for any other reason — never worse than the original
    behaviour, just not faster.
    """
    _mark("tts_start")

    session_key = (voice, text)
    cached = _session_audio_cache.get(session_key) or (
        _cache_path(text, voice) if os.path.exists(_cache_path(text, voice)) else None
    )
    if cached:
        # Already synthesised in a previous run/turn — nothing to stream.
        _mark("playback_start")
        play(cached)
        _session_audio_cache[session_key] = cached
        return

    path = _cache_path(text, voice)
    tmp_path = path + ".tmp"
    try:
        import asyncio
        asyncio.run(_tts_stream_to_mpg123(text, voice, tmp_path))
        os.replace(tmp_path, path)
        _session_audio_cache[session_key] = path
    except Exception as e:
        print(f"   ⚠️  streaming TTS playback unavailable ({e}) — falling back to normal playback")
        p = synthesize(text, voice)
        _mark("playback_start")
        play(p)


# ──────────────────────────────────────────────
#  STREAMING SPEAK  — plays sentences as they arrive (feature 1, output half)
# ──────────────────────────────────────────────

class StreamingSpeaker:
    """
    Consumer that plays synthesised sentences in order while the producer
    (LLM stream) is still generating later sentences. Synthesis for
    sentence N+1 happens in the background while sentence N is playing.
    """

    def __init__(self, lang: str):
        self.lang = lang
        self._q: "queue.Queue" = queue.Queue()
        self._thread = threading.Thread(target=self._consume, daemon=True)
        self._thread.start()
        self._announced = False
        self._sentence_index = 0

    def _consume(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            mode, future = item
            if mode == "sync":
                # Streaming path already played the audio itself inside the
                # worker thread — just block here until it's actually done,
                # so sentence order is still preserved.
                future.result()
            else:
                path = future.result()
                play(path)

    def say(self, sentence: str):
        sentence = sentence.strip()
        if not sentence:                       # feature 4 — never pass empty text
            return
        if not self._announced:
            show_state(State.SPEAKING)
            self._announced = True
        print(f"   💬 {sentence}")
        voice = pick_voice(sentence, self.lang)
        self._sentence_index += 1

        if STREAM_TTS_PLAYBACK and self._sentence_index == 1:
            # Only sentence #1: every later sentence already overlaps its
            # synthesis with the previous sentence's playback below, so
            # streaming only matters for the one sentence with nothing to
            # overlap with. Ordering stays safe because this future does
            # its own playback synchronously — the consumer just waits on it.
            future = tts_executor.submit(_synth_and_play_streaming, sentence, voice)
            self._q.put(("sync", future))
        else:
            future = tts_executor.submit(synthesize, sentence, voice)
            self._q.put(("play", future))

    def finish(self):
        self._q.put(None)
        self._thread.join()


def speak_blocking(text: str, lang: str = "en"):
    """Simple one-shot speak for short fixed prompts (greeting, idle, wake-ack)."""
    text = text.strip()
    if not text:
        return
    voice = pick_voice(text, lang)
    path = synthesize(text, voice)
    play(path)


# ──────────────────────────────────────────────
#  VAD RECORDING WITH MID-TURN SENTENCE SEGMENTATION  (feature 1, input half)
# ──────────────────────────────────────────────

def transcribe_segment(audio: np.ndarray) -> Tuple[str, str]:
    """Whisper call for a single audio segment. Returns (text, lang).

    Writes the WAV to an in-memory buffer instead of a temp file on disk —
    avoids a file create + write + unlink round trip (and SD-card wear on
    a Raspberry Pi) on every segment, background or trailing.
    """
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV")
    buf.seek(0)
    buf.name = "segment.wav"   # SDK uses this for the upload filename

    result = client.audio.transcriptions.create(
        model=STT_MODEL,
        file=buf,
        response_format="verbose_json",
    )

    text = (result.text or "").strip()
    lang = (result.language or "en").strip().lower()
    if lang == "ur":
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"

    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            lang = "hi"
            break

    return text, lang


def capture_and_transcribe(timeout: float, track_timing: bool = False) -> Tuple[Optional[str], str]:
    """
    Records one full user turn via VAD. Whenever a short mid-turn pause
    (SENTENCE_PAUSE) is detected, the audio collected so far is sent off
    for transcription in the background immediately — while the user is
    still talking — instead of waiting for the whole turn to end.

    `track_timing=True` resets and records the per-turn latency dict used
    by print_turn_timings() — only set for real listening turns, not idle
    wake-word polling.

    Returns (combined_text_or_None, lang).
    """
    if track_timing:
        _reset_timings()

    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
        blocksize=blocksize, callback=callback,
    )
    stream.start()

    pre_buffer: list = []
    segment_buffer: list = []
    recording = False
    silence_start: Optional[float] = None
    segment_split_done = False
    any_mid_flush = False   # True once ≥1 background segment has been sent for STT
    idle_clock = time.time()
    futures: List = []

    def flush_segment():
        nonlocal segment_buffer
        if not segment_buffer:
            return
        audio = np.concatenate(segment_buffer, axis=0)
        segment_buffer = []
        if len(audio) < SAMPLE_RATE * 0.15:   # too short to bother
            return
        futures.append(stt_executor.submit(transcribe_segment, audio))
        print("   ✂️  segment captured → transcribing in background...")

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                if not recording and time.time() - idle_clock >= timeout:
                    stream.stop(); stream.close()
                    return None, "en"
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:
                idle_clock = time.time()
                silence_start = None
                segment_split_done = False
                if not recording:
                    recording = True
                    segment_buffer = list(pre_buffer)
                segment_buffer.append(chunk)

            elif recording:
                segment_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                    continue
                elapsed = time.time() - silence_start
                required_silence = SILENCE_AFTER_SPEECH_FOLLOWUP if any_mid_flush else SILENCE_AFTER_SPEECH
                if elapsed >= required_silence:
                    if track_timing:
                        _mark("speech_end")
                    break  # end of the whole turn
                if elapsed >= SENTENCE_PAUSE and not segment_split_done:
                    segment_split_done = True
                    any_mid_flush = True
                    flush_segment()

            else:
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)
                if time.time() - idle_clock >= timeout:
                    stream.stop(); stream.close()
                    return None, "en"
    finally:
        stream.stop()
        stream.close()

    flush_segment()  # final trailing segment

    if not futures:
        return None, "en"

    pieces = []
    lang = "en"
    for fut in futures:
        try:
            text, seg_lang = fut.result()
        except Exception:
            continue
        if text:
            pieces.append(text)
            lang = seg_lang  # last non-empty segment's script-scan wins

    if track_timing:
        _mark("stt_complete")

    combined = " ".join(p for p in pieces if p.strip()).strip()

    # Re-run the script scan over the FULL combined text so language never
    # flips mid-sentence just because one short segment mis-detected.
    for ch in combined:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            lang = "hi"
            break

    if not combined or len(combined) < 1:
        return None, lang

    return combined, lang


# ──────────────────────────────────────────────
#  NYLA REPORT API — auth-robot + /ask
# ──────────────────────────────────────────────

class NylaClient:
    """JWT via /org/auth-robot at startup; /nyla-report/ask for each question."""

    def __init__(self):
        self._token: Optional[str] = None
        self._lock = threading.Lock()

    def _require_credentials(self) -> None:
        if not NYLA_API_KEY or not NYLA_ENCRYPTED_DATA:
            raise RuntimeError(
                "Missing NYLA_API_KEY or NYLA_ENCRYPTED_DATA in .env — "
                "required for /org/auth-robot"
            )

    def _fetch_token(self, new_chat: str) -> str:
        self._require_credentials()
        resp = requests.post(
            NYLA_AUTH_URL,
            data={
                "encrypted_data": NYLA_ENCRYPTED_DATA,
                "api_key": NYLA_API_KEY,
                "new_chat": new_chat,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.text.strip().strip('"')

    def authenticate(self) -> str:
        with self._lock:
            self._token = self._fetch_token(new_chat="1")
            return self._token

    def get_token(self, force_new: bool = False) -> str:
        with self._lock:
            if self._token is None:
                self._token = self._fetch_token(new_chat="1")
            elif force_new:
                self._token = self._fetch_token(new_chat="1")
            return self._token

    def ask(self, question: str, _retried: bool = False) -> str:
        token = self.get_token()
        resp = requests.post(
            NYLA_ASK_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={"question": question},
            timeout=30,
        )
        if resp.status_code == 401 and not _retried:
            self.get_token(force_new=True)
            return self.ask(question, _retried=True)
        resp.raise_for_status()
        payload = resp.json()
        if not payload.get("success"):
            raise RuntimeError(payload.get("message") or "Nyla API returned an error")
        return payload["data"]["answer"]


nyla_client = NylaClient()


# ──────────────────────────────────────────────
#  WAKE WORD
# ──────────────────────────────────────────────

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# ──────────────────────────────────────────────
#  AI REPLY — streamed, sentence-by-sentence, with reply caching
# ──────────────────────────────────────────────

def stream_ai_reply_and_speak(user_text: str, lang: str) -> str:
    """
    Gets the reply from the Nyla Report API and speaks it sentence-by-
    sentence (Nyla's /ask endpoint is not streaming, so the full answer
    arrives at once, then gets split and spoken the same way a cache hit
    already was). Returns the full reply text.
    """
    lang_history = history[lang]

    cache_key = (lang, user_text.strip().lower())
    speaker = StreamingSpeaker(lang)

    # # ── Prompt cache hit: skip the Nyla call entirely ─────────────
    # if cache_key in reply_cache:
    #     print("   💾 cache hit — skipping Nyla call")
    #     cached_reply = reply_cache[cache_key]
    #     lang_history.append({"role": "user", "content": user_text})
    #     lang_history.append({"role": "assistant", "content": cached_reply})
    #     for sentence in SENTENCE_SPLIT_RE.split(cached_reply):
    #         speaker.say(sentence)
    #     speaker.finish()
    #     return cached_reply

    # lang_history.append({"role": "user", "content": user_text})

    _mark("gpt_start")
    try:
        full_reply = nyla_client.ask(user_text)
    except Exception:
        speaker.finish()
        raise
    # Nyla answers in one shot (no token stream), so first-token and
    # first-sentence land together — kept as separate marks purely so
    # print_turn_timings() keeps working unchanged.
    _mark("gpt_first_token")
    _mark("first_sentence")

    full_reply = (full_reply or "").strip()
    if full_reply:                        # feature 4 — don't cache/store empties
        for sentence in SENTENCE_SPLIT_RE.split(full_reply):
            speaker.say(sentence)
        lang_history.append({"role": "assistant", "content": full_reply})
        reply_cache[cache_key] = full_reply

    speaker.finish()
    return full_reply


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 56)
    print("  🤖 RoboBot — streaming speech-to-speech assistant")
    print("=" * 56)
    print("  States:")
    print("    👂 LISTENING  — auto-detects your voice")
    print(f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle")
    print("                   say 'Hello' to wake up")
    print("    🤔 THINKING   — waiting on the first tokens")
    print("    🔊 SPEAKING   — plays each sentence as it's ready")
    print("  Ctrl+C to quit")
    print("=" * 56 + "\n")


# ──────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────

def main():
    print_banner()

    print("   🔐 Authenticating with Nyla (auth-robot)...")
    try:
        nyla_client.authenticate()
        print("   ✅ Nyla session ready.\n")
    except Exception as e:
        handle_error(e, "Nyla authentication")
        return

    state = State.LISTENING
    lang  = "hi"

    try:
        speak_blocking(
            "Hello, mein RoboBot hoon . main aapki kese madad kar sakta hoon. "
            "Krapya apna sawaal poochhiye .",
            lang="hi",
        )
    except Exception as e:
        handle_error(e, "opening greeting")

    try:
        while True:

            # ════════════════════════ IDLE ════════════════════════
            if state == State.IDLE:
                show_state(state, "— say 'Hello' to activate...")
                try:
                    combined, _ = capture_and_transcribe(timeout=IDLE_POLL_TIMEOUT)
                except Exception as e:
                    handle_error(e, "idle listening")
                    continue

                if not combined:
                    continue

                print(f"   Heard: {combined!r}")
                if is_wake_word(combined):
                    state = State.LISTENING
                    print("\n✅ Wake word detected!")
                    try:
                        speak_blocking("Haan, mein sun raha hoon. Aap apna sawaal poochhiye.", lang="hi")
                    except Exception as e:
                        handle_error(e, "wake acknowledgement")
                else:
                    print("   Not a wake word — staying idle.")
                continue

            # ═════════════════════ LISTENING ═══════════════════════
            if state == State.LISTENING:
                show_state(state, f"— silence for {int(IDLE_TIMEOUT)}s → idle")
                try:
                    user_text, lang = capture_and_transcribe(timeout=IDLE_TIMEOUT, track_timing=True)
                except Exception as e:
                    handle_error(e, "listening / transcription")
                    continue

                if user_text is None:
                    state = State.IDLE
                    print(f"\n⏱️  No speech for {int(IDLE_TIMEOUT)}s — going idle.")
                    try:
                        speak_blocking(
                            "Mein abhi idle mode mein ja raha hoon. Jab zaroorat ho, 'Hello' kahiye.",
                            lang="hi",
                        )
                    except Exception as e:
                        handle_error(e, "idle announcement")
                    continue

                user_text = user_text.strip()
                if not user_text:             # feature 4 — never forward empty text
                    print("⚠️  Could not understand — listening again.")
                    continue

                print(f"   You [{lang.upper()}] › {user_text}")
                state = State.THINKING
                continue

            # ═════════════════════ THINKING ════════════════════════
            if state == State.THINKING:
                show_state(state)
                try:
                    stream_ai_reply_and_speak(user_text, lang)
                except Exception as e:
                    handle_error(e, "LLM reply / speech synthesis")
                    state = State.LISTENING
                    continue
                if PRINT_LATENCY_TIMINGS:
                    print_turn_timings()
                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")
        stt_executor.shutdown(wait=False, cancel_futures=True)
        tts_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
