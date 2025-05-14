# whisper_transcription.py
from transformers import pipeline
import torch
import sys

def transcribe_audio(audio_file):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        transcriber = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=device
        )
        print(f"Transcribing {audio_file}...")
        result = transcriber(audio_file)
        transcription = result["text"]
        print("Transcription:", transcription)
        with open("transcription.txt", "w") as f:
            f.write(transcription)
        print("Transcription saved to transcription.txt")
        return transcription
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

if __name__ == "__main__":
    audio_file = "hello-46355.mp3"
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    transcribe_audio(audio_file)