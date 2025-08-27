import os
import whisper
import numpy as np
import subprocess

# ===== Config =====
AUDIO_FILE = "audio/lecture.mp3"
MODEL_SIZE = "small"       # "tiny", "base", "small", "medium", "large"
MAX_MINUTES = 2            # total audio to process (in minutes)
CHUNK_DURATION = 60        # seconds per chunk
SAMPLE_RATE = whisper.audio.SAMPLE_RATE

# ===== Load model =====
print("Loading Whisper model...")
model = whisper.load_model(MODEL_SIZE)

# ===== Load audio =====
print("Loading audio...")
audio = whisper.load_audio(AUDIO_FILE)

# Convert max minutes to samples
max_samples = MAX_MINUTES * 60 * SAMPLE_RATE
audio = audio[:max_samples]  # truncate to MAX_MINUTES

# ===== Chunked transcription =====
chunk_samples = CHUNK_DURATION * SAMPLE_RATE
transcript = ""

print("Transcribing in chunks...")
for i in range(0, len(audio), chunk_samples):
    chunk = audio[i:i+chunk_samples]
    chunk = whisper.pad_or_trim(chunk)
    mel = whisper.log_mel_spectrogram(chunk).to(model.device)
    result = model.decode(mel, whisper.DecodingOptions(fp16=False))
    transcript += result.text + " "

# ===== Output =====
print("\n--- TRANSCRIPT (First {} minutes) ---\n".format(MAX_MINUTES))
print(transcript)


# Make sure output folder exists
os.makedirs("output", exist_ok=True)

# Now write the file
with open("output/transcript.txt", "w", encoding="utf-8") as f:
    f.write(transcript)

print("\n✅ Transcript saved to output/transcript.txt")

subprocess.run(["python", "summarize.py"])