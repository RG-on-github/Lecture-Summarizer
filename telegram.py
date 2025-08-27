import requests

bot_token = "8436255248:AAFeNa_MTkzpealBEeyfKwXXuNuwBiEgvOg"
chat_id = "1282848803"  # The number from step 3

# Read your summary
with open("output/summary.txt", "r", encoding="utf-8") as f:
    heading = "SUMMARY\n"
    summary_text = f.read()

# Send message
url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
payload = {"chat_id": chat_id, "text": summary_text}

r = requests.post(url, data=payload)
print(r.json())