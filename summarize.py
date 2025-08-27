from transformers import pipeline
import subprocess
import os

# ===== Read transcript from file =====
transcript_file = "output/transcript.txt"
with open(transcript_file, "r", encoding="utf-8") as f:
    transcript = f.read()

# ===== Summarization using BERT =====
print("\nSummarizing transcript...")

# Load summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# BERT-based summarization works better in chunks for long text
MAX_CHARS = 1000  # split transcript if too long
summary_text = ""
for i in range(0, len(transcript), MAX_CHARS):
    chunk = transcript[i:i+MAX_CHARS]
    summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
    summary_text += summary + " "

# Save summary
os.makedirs("output", exist_ok=True)
with open("output/summary.txt", "w", encoding="utf-8") as f:
    f.write(summary_text)

print("\n✅ Summary saved to output/summary.txt")
print("\n--- SUMMARY ---\n")
print(summary_text)

# ===== Send summary via Telegram =====
print("\nSending summary via Telegram...")
subprocess.run(["python", "telegram.py"])