from faster_whisper import WhisperModel
import tempfile
import os

# load once
model = WhisperModel("base", device="cpu", compute_type="int8")


def speech_to_text(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_bytes)
        path = f.name

    segments, _ = model.transcribe(path)

    text = " ".join([seg.text for seg in segments])

    os.remove(path)
    return text.strip()