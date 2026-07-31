import os
import asyncio
import tempfile
import queue
import time
import threading
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq
import edge_tts
import pygame
from dotenv import load_dotenv

# load_dotenv()

load_dotenv()

# ──────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

STT_MODEL  = "whisper-large-v3-turbo"
CHAT_MODEL = "openai/gpt-oss-120b"

TTS_VOICE_EN = "en-US-JennyNeural"
TTS_VOICE_HI = "hi-IN-SwaraNeural"

SAMPLE_RATE = 16000
CHANNELS    = 1
MAX_TOKENS  = 250  # gpt-oss-120b burns hidden "reasoning" tokens before
                    # writing content — too low a budget (e.g. 60) can use
                    # it all up on reasoning and return an empty answer.
                    # reasoning_effort="low" (set below) keeps that hidden
                    # cost small, which is what actually buys speed.

# ── VAD tuning ──────────────────────────────────────────────
ENERGY_THRESHOLD     = 0.025
SILENCE_AFTER_SPEECH = 0.6
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.3
CHUNK_SECS           = 0.5

IDLE_TIMEOUT         = 10.0
IDLE_POLL_TIMEOUT    = 30.0

# ── Wake words ──────────────────────────────────────────────
WAKE_WORDS = ["hello", "hey", "robotwala", "hey robotwala", "hello robotwala", "robo"]

# ──────────────────────────────────────────────
#  SYSTEM PROMPTS — Robotwala
# ──────────────────────────────────────────────

# Robotwala company context injected into every conversation turn
ROBOTWALA_CONTEXT = """
Robotwala is an AI, Robotics and STEM company in Indore.

Services:
AI/ML, Robotics, Drone Technology, 3D Printing, Coding,
Robotics Education and Industrial Automation.

Products:
LUCY, NYLA, NYLA 2.0 and EdTech Kits.

Audience:
Students, schools, colleges and businesses.

Location:
Old Palasia, Indore, Madhya Pradesh, India.

Website:
robotwala.org
"""

SYSTEM_EN = (
    "Your name is RoboBot. You are the official AI voice assistant of Robotwala — "
    "Indore's leading AI, Robotics, and STEM education institute. "
    "Help students, schools, colleges, and businesses learn about Robotwala's courses, "
    "products, collaborations, internships, franchise opportunities, and services. "
    "Keep responses concise and conversational. No bullet points or markdown. "
    "Speak naturally as if talking face-to-face. "
    "Here is your knowledge base:\n" + ROBOTWALA_CONTEXT
)

SYSTEM_HI = (
    "Aapka naam RoboBot hai. Aap Robotwala ke official AI voice assistant hain — "
    "Indore ka leading AI, Robotics aur STEM education institute. "
    "Students, schools, colleges aur businesses ko Robotwala ke courses, products, "
    "collaborations, internships, franchise aur services ke baare mein batayein. "
    "Jab user Hindi mein baat kare, toh aap bhi Hindi mein jawab dein — "
    "Roman script (Hinglish) mein likhein taaki TTS sahi se bol sake. "
    "Jab user English mein baat kare, toh English mein jawab dein. "
    "Apne uttar chhote aur batcheet ke andaz mein rakhein. "
    "Koi bullet points ya markdown nahi. "
    "Yahan aapka knowledge base hai:\n" + ROBOTWALA_CONTEXT
)

# ──────────────────────────────────────────────
#  STATE
# ──────────────────────────────────────────────

class State(Enum):
    IDLE       = "idle"
    LISTENING  = "listening"
    PROCESSING = "processing"
    SPEAKING   = "speaking"

# ──────────────────────────────────────────────
#  SETUP
# ──────────────────────────────────────────────

client = Groq(api_key=GROQ_API_KEY)

# ── Single unified conversation history (FIX: no more split-by-language) ──
# Stores messages in order regardless of language so context is never lost.
conversation_history: list = []

# ── Barge-in flag ────────────────────────────────────────────────────────
barge_in_detected = threading.Event()   # set by mic monitor thread

pygame.mixer.init()


# ──────────────────────────────────────────────
#  TIMING  (thread-safe — STT/LLM run on the main thread
#  while TTS + playback run on background worker threads)
# ──────────────────────────────────────────────

print_lock = threading.Lock()


def _log(msg: str):
    with print_lock:
        print(msg)


class TurnTimer:
    """
    Records a fixed set of named checkpoints for one conversation turn and
    prints a clean aligned latency table once playback of the first
    sentence starts, e.g.:

        Speech End           0 ms
        STT                  610 ms
        GPT First Token      340 ms
        Sentence Complete    380 ms
        TTS Start            5 ms
        Playback Started     260 ms
        ------------------------------
        Total Latency        1595 ms

    Each row (after "Speech End") is the time elapsed since the PREVIOUS
    checkpoint, not cumulative — so "Total Latency" is their sum, and
    equals the time from "you stopped talking" to "you hear the reply".

    Only the first sentence's journey is tracked (mark() ignores repeat
    calls for a label), since that's what determines perceived latency.
    """
    ORDER = [
        "Speech End",
        "STT",
        "GPT First Token",
        "Sentence Complete",
        "TTS Start",
        "Playback Started",
    ]

    def __init__(self):
        self._lock = threading.Lock()
        self._times: dict = {}

    def mark(self, label: str):
        with self._lock:
            if label not in self._times:
                self._times[label] = time.time()

    def report(self):
        rows = []
        prev_t = None
        total = 0.0
        for label in self.ORDER:
            t = self._times.get(label)
            if t is None:
                continue
            delta = 0.0 if prev_t is None else (t - prev_t)
            if prev_t is not None:
                total += delta
            rows.append((label, delta))
            prev_t = t

        if not rows:
            return

        width = max(len(l) for l, _ in rows + [("Total Latency", 0)]) + 4
        lines = ["", *(f"  {label:<{width}}{delta * 1000:.0f} ms" for label, delta in rows)]
        lines.append(f"  {'-' * (width + 8)}")
        lines.append(f"  {'Total Latency':<{width}}{total * 1000:.0f} ms")
        lines.append("")
        _log("\n".join(lines))


# ──────────────────────────────────────────────
#  BARGE-IN MONITOR  (runs in background thread while bot is SPEAKING)
# ──────────────────────────────────────────────

def _barge_in_monitor():
    """
    Continuously samples the microphone while the bot speaks.
    If RMS energy crosses ENERGY_THRESHOLD for > 0.3 s → sets barge_in_detected.
    Killed automatically when the caller clears/checks the event.
    """
    blocksize     = int(SAMPLE_RATE * CHUNK_SECS)
    loud_since    = None
    BARGE_IN_HOLD = 0.3   # seconds of sustained speech to trigger barge-in

    def cb(indata, frames, t, status):
        nonlocal loud_since
        rms = float(np.sqrt(np.mean(indata ** 2)))
        if rms >= ENERGY_THRESHOLD:
            if loud_since is None:
                loud_since = time.time()
            elif time.time() - loud_since >= BARGE_IN_HOLD:
                barge_in_detected.set()
        else:
            loud_since = None

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", blocksize=blocksize, callback=cb):
        # Run until the event is set (means we detected barge-in)
        # OR until _stop flag is toggled by the caller
        while not barge_in_detected.is_set():
            time.sleep(0.05)


# ──────────────────────────────────────────────
#  VAD RECORDING
# ──────────────────────────────────────────────

def capture_speech(timeout: float) -> Optional[np.ndarray]:
    """
    Listens via microphone using Voice Activity Detection.
    Returns audio ndarray or None on timeout.
    """
    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )
    stream.start()

    speech_buffer: list            = []
    pre_buffer:    list            = []
    recording                      = False
    silence_start: Optional[float] = None
    idle_clock                     = time.time()

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                if not recording and time.time() - idle_clock >= timeout:
                    return None
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:
                idle_clock    = time.time()
                silence_start = None
                if not recording:
                    recording = True
                    speech_buffer = list(pre_buffer)
                speech_buffer.append(chunk)

            elif recording:
                speech_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    break

            else:
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)
                if time.time() - idle_clock >= timeout:
                    return None

    finally:
        stream.stop()
        stream.close()

    if not speech_buffer:
        return None

    audio = np.concatenate(speech_buffer, axis=0)
    return audio if len(audio) >= SAMPLE_RATE * MIN_SPEECH_SECS else None


# ──────────────────────────────────────────────
#  TRANSCRIBE
# ──────────────────────────────────────────────

def transcribe(audio: np.ndarray, timer: "TurnTimer" = None) -> Tuple[str, str]:
    """Returns (text, lang_code) where lang_code is 'hi' or 'en'."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    sf.write(tmp_path, audio, SAMPLE_RATE)

    with open(tmp_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=STT_MODEL,
            file=f,
            response_format="verbose_json",
        )

    os.unlink(tmp_path)

    if timer:
        timer.mark("STT")

    text = (result.text or "").strip()
    lang = (result.language or "en").strip().lower()

    if lang == "ur":
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"

    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            lang = "hi"
            break
        if 0x0600 <= cp <= 0x06FF:
            lang = "hi"
            break

    return text, lang


# ──────────────────────────────────────────────
#  WAKE WORD
# ──────────────────────────────────────────────

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)


# ──────────────────────────────────────────────
#  AI REPLY  (unified history, no language split)
#  STREAMING VERSION — yields each finished sentence as
#  soon as it appears in the token stream, so downstream
#  TTS can start on sentence 1 while the model is still
#  generating sentence 2, 3, ...
# ──────────────────────────────────────────────

SENTINEL = object()   # signals "no more items" between pipeline stages


def _extract_sentences(buffer: str):
    """
    Splits buffer on sentence-ending punctuation (. ! ? or newline).
    Returns (list_of_complete_sentences, remaining_incomplete_buffer).
    """
    sentences = []
    start = 0
    for i, ch in enumerate(buffer):
        if ch in ".!?\n":
            piece = buffer[start:i + 1].strip()
            if piece:
                sentences.append(piece)
            start = i + 1
    return sentences, buffer[start:]


def get_ai_reply_stream(user_text: str, lang: str, timer: "TurnTimer" = None):
    """
    Generator: yields sentences as soon as they're complete.
    Also appends the full reply to conversation_history once done.
    """
    system = SYSTEM_HI if lang == "hi" else SYSTEM_EN

    if lang == "hi":
        instruction = (
            f"Respond in Hindi using Roman/Latin script only. "
            f"Keep the answer VERY short — one short sentence, max ~20 words. "
            f"User said: {user_text}"
        )
    else:
        instruction = (
            f"Respond in English. "
            f"Keep the answer VERY short — one short sentence, max ~20 words. "
            f"User said: {user_text}"
        )

    conversation_history.append({
        "role": "user",
        "content": instruction
    })

    buffer = ""
    full_reply_parts = []
    yielded_any = False
    first_token_seen = False

    try:
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                *conversation_history,
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.3,
            reasoning_effort="low",
            stream=True,
        )

        print("   AI [", lang.upper(), "] › ", end="", sep="")

        for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta.content:
                if not first_token_seen:
                    first_token_seen = True
                    if timer:
                        timer.mark("GPT First Token")

                token = delta.content
                print(token, end="", flush=True)
                full_reply_parts.append(token)
                buffer += token

                sentences, buffer = _extract_sentences(buffer)
                for s in sentences:
                    if not yielded_any and timer:
                        timer.mark("Sentence Complete")
                    yielded_any = True
                    yield s

        # Trailing text with no terminal punctuation still needs to be spoken
        if buffer.strip():
            if not yielded_any and timer:
                timer.mark("Sentence Complete")
            yielded_any = True
            yield buffer.strip()

        print()

    except Exception as e:
        print(f"\n   ⚠️  API error: {e}")
        fallback = (
            "Mujhe abhi network issue aa raha hai, please thoda wait karein."
            if lang == "hi"
            else "I'm experiencing a network issue. Please try again in a moment."
        )
        full_reply_parts = [fallback]
        if not yielded_any and timer:
            timer.mark("Sentence Complete")
        yielded_any = True
        yield fallback

    if not yielded_any:
        print("   ⚠️  Model returned an empty response.")
        fallback = (
            "Maaf kijiye, mujhe samajh nahi aaya. Kripya dobara poochhiye."
            if lang == "hi"
            else "Sorry, I didn't get that. Could you please ask again?"
        )
        full_reply_parts = [fallback]
        if timer:
            timer.mark("Sentence Complete")
        yield fallback

    reply = "".join(full_reply_parts).strip()

    conversation_history.append({
        "role": "assistant",
        "content": reply
    })

    if len(conversation_history) > 20:
        conversation_history[:] = conversation_history[-20:]






# ──────────────────────────────────────────────
#  VOICE SELECTION
# ──────────────────────────────────────────────

def pick_voice(text: str, lang: str) -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return TTS_VOICE_HI
        if 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


# ──────────────────────────────────────────────
#  SPEAK  (with barge-in support)
# ──────────────────────────────────────────────

async def _tts(text: str, path: str, voice: str):
    await edge_tts.Communicate(text, voice=voice).save(path)


def _tts_worker(sentence_q: "queue.Queue", audio_q: "queue.Queue", lang: str, timer: "TurnTimer" = None):
    """
    Pulls finished sentences off sentence_q, synthesizes each to a temp
    mp3, and pushes the file path onto audio_q — as soon as it's ready,
    not after the whole reply is done.
    """
    first = True
    while True:
        item = sentence_q.get()
        if item is SENTINEL:
            audio_q.put(SENTINEL)
            return

        if first:
            first = False
            if timer:
                timer.mark("TTS Start")

        voice = pick_voice(item, lang)
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
            asyncio.run(_tts(item, tmp_path, voice))
            audio_q.put(tmp_path)
        except Exception as e:
            _log(f"\n   ⚠️  TTS error: {e}")


def _player_worker(audio_q: "queue.Queue", timer: "TurnTimer" = None):
    """
    Plays synthesized mp3s in order, as soon as each becomes available.
    Never waits for later sentences to be ready before starting.
    """
    first = True
    while True:
        item = audio_q.get()
        if item is SENTINEL:
            return

        if first:
            first = False
            if timer:
                timer.mark("Playback Started")
                timer.report()

        try:
            pygame.mixer.music.load(item)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(30)
            pygame.mixer.music.unload()
        except Exception as e:
            _log(f"\n   ⚠️  Playback error: {e}")
        finally:
            try:
                os.unlink(item)
            except OSError:
                pass


def speak_reply_streaming(user_text: str, lang: str, timer: "TurnTimer" = None) -> str:
    """
    Full low-latency pipeline: streams the LLM reply sentence-by-sentence,
    synthesizes each sentence to speech as soon as it's complete, and plays
    sentences back in order — all three stages running concurrently, so
    playback starts on sentence 1 while sentence 2+ is still being
    generated/synthesized. No "wait for full answer" / "wait for full mp3".

    timer: TurnTimer used to record + print the latency table. Optional.
    """
    sentence_q: "queue.Queue" = queue.Queue()
    audio_q:    "queue.Queue" = queue.Queue()

    tts_thread    = threading.Thread(target=_tts_worker, args=(sentence_q, audio_q, lang, timer), daemon=True)
    player_thread = threading.Thread(target=_player_worker, args=(audio_q, timer), daemon=True)
    tts_thread.start()
    player_thread.start()

    full_reply = ""
    try:
        for sentence in get_ai_reply_stream(user_text, lang, timer=timer):
            full_reply += (" " if full_reply else "") + sentence
            sentence_q.put(sentence)
    finally:
        sentence_q.put(SENTINEL)

    tts_thread.join()
    player_thread.join()

    # tiny buffer so speaker echo fades before mic reopens
    time.sleep(0.1)

    return full_reply.strip()


def speak(text: str, lang: str = "en") -> bool:
    voice = pick_voice(text, lang)
    print(f"   🔊 Voice → {voice}")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    asyncio.run(_tts(text, tmp_path, voice))

    # ── Disable barge-in during playback to avoid self-echo ──
    barge_in_detected.clear()

    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()

    # Wait for playback to finish — NO barge-in monitor here
    while pygame.mixer.music.get_busy():
        pygame.time.wait(80)

    pygame.mixer.music.unload()
    os.unlink(tmp_path)

    # ── Add a cooldown buffer so speaker echo fades before mic opens ──
    time.sleep(0.15)   # 0.15 seconds of silence before listening resumes

    return False   # barge-in disabled for now


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def print_banner():
    print("\n" + "=" * 60)
    print("  RoboBot 🤖  |  Robotwala — Igniting Young Minds")
    print("=" * 60)
    print("  States:")
    print("    👂 LISTENING  — auto-detects your voice")
    print(f"    😴 IDLE       — {int(IDLE_TIMEOUT)}s silence → idle")
    print("                   say 'Hello' or 'Robotwala' to wake up")
    print("    🔊 SPEAKING   — say anything to interrupt (barge-in)")
    print("  Ctrl+C to quit")
    print("=" * 60 + "\n")


def state_label(state: State) -> str:
    return {
        State.IDLE:       "😴 IDLE",
        State.LISTENING:  "👂 LISTENING",
        State.SPEAKING:   "🔊 SPEAKING",
        State.PROCESSING: "🤔 PROCESSING",
    }[state]


# ──────────────────────────────────────────────
#  MAIN LOOP
# ──────────────────────────────────────────────

def main():
    print_banner()

    state     = State.LISTENING
    reply     = ""
    lang      = "hi"
    user_text = ""

    # ── Opening greeting ──────────────────────────────────────────
    speak(
         """
Hello , i am robobot , how can i help you.
""",
        lang="hi",
    )

    try:
        while True:

            # ════════════════════════════════════════════════════
            #  IDLE — wait for wake word
            # ════════════════════════════════════════════════════
            if state == State.IDLE:
                print(f"\n{state_label(state)}  — say 'Hello' or 'Robotwala' to activate...")

                audio = capture_speech(timeout=IDLE_POLL_TIMEOUT)

                if audio is None:
                    continue

                print("🔍 Checking for wake word...")
                wake_text, _ = transcribe(audio)
                print(f"   Heard: {wake_text!r}")

                if is_wake_word(wake_text):
                    state = State.LISTENING
                    print("\n✅ Wake word detected!")
                    speak(
                        "Haan, main sun raha hun. Aap apna sawaal poochhiye.",
                        lang="hi",
                    )
                else:
                    print("   Not a wake word — staying idle.")

                continue

            # ════════════════════════════════════════════════════
            #  LISTENING — VAD; 10s silence → IDLE
            # ════════════════════════════════════════════════════
            if state == State.LISTENING:
                print(f"\n{state_label(state)}  "
                      f"— silence for {int(IDLE_TIMEOUT)}s → idle")

                audio = capture_speech(timeout=IDLE_TIMEOUT)

                if audio is None:
                    state = State.IDLE
                    print(f"\n⏱️  No speech for {int(IDLE_TIMEOUT)}s — going idle.")
                    speak(
                        "Main abhi idle mode mein ja raha hun. "
                        "Jab zaroorat ho, 'Hello' ya 'Robotwala' kahiye.",
                        lang="hi",
                    )
                    continue

                print("🔍 Transcribing...")
                turn_timer = TurnTimer()
                turn_timer.mark("Speech End")
                user_text, lang = transcribe(audio, timer=turn_timer)

                if not user_text:
                    print("⚠️  Could not understand — listening again.")
                    continue

                print(f"   You [{lang.upper()}] › {user_text}")
                state = State.PROCESSING
                continue

            # ════════════════════════════════════════════════════
            #  PROCESSING — generate AI reply
            # ════════════════════════════════════════════════════
            if state == State.PROCESSING:
                print(f"\n{state_label(state)}")
                print("🤔 Thinking & speaking (streamed)...")

                # Generates + synthesizes + plays sentence-by-sentence.
                # Speech starts on the first sentence — no waiting for the
                # full answer or the full mp3.
                reply = speak_reply_streaming(user_text, lang, timer=turn_timer)
                print(f"   AI [{lang.upper()}] › {reply}")

                state = State.LISTENING
                continue

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down RoboBot...")


if __name__ == "__main__":
    main()
