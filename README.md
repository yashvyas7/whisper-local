# Whisper GPU Transcription Service

Production-ready GPU transcription using OpenAI's Whisper large-v2. Auto-detects language and generates transcripts only when detected language matches intended language.

## Requirements

- Ubuntu 20.04/22.04/24.04
- NVIDIA GPU with 8GB+ VRAM
- CUDA drivers installed

## Installation

```bash
# Install system dependencies
sudo apt update
sudo apt install ffmpeg -y

# Install Python packages
pip install -r requirements.txt
```

## Usage

```bash
python whisper_transcriber.py audio.webm hi --output-dir ./transcripts
```

## Features

- GPU-only (requires CUDA)
- Automatic language detection
- Timestamped segments
- FP16 precision for speed
- Production-ready logging

Supports: mp3, wav, m4a, flac, webm, ogg, etc.
