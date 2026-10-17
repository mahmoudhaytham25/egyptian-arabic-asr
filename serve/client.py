"""Minimal test client for the transcription API.

Usage:
    uvicorn app:app --port 8000          # in this directory
    python client.py path/to/audio.wav
    python client.py audio.wav --url http://localhost:8000/transcribe/
"""

import argparse

import requests

parser = argparse.ArgumentParser(description="Send an audio file to the ASR API")
parser.add_argument("audio", help="Path to a .wav (or any ffmpeg-decodable) audio file")
parser.add_argument("--url", default="http://127.0.0.1:8000/transcribe/")
args = parser.parse_args()

with open(args.audio, "rb") as f:
    response = requests.post(args.url, files={"audio_file": f})
response.raise_for_status()
print(response.json())
