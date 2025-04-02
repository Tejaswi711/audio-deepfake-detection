# Generate more fake samples using TTS
from gtts import gTTS
import os


def create_fake_sample(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(f"data/train/spoof/{filename}")


# Create 5 more fake samples
samples = [
    ("This is a computer generated voice", "fake_4.wav"),
    ("Artificial intelligence can clone voices", "fake_5.wav"),
    ("Deepfake audio is becoming more common", "fake_6.wav"),
    ("This voice was created by a machine", "fake_7.wav"),
    ("Text to speech technology is advancing", "fake_8.wav")
]

for text, filename in samples:
    create_fake_sample(text, filename)