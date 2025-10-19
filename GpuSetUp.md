# GPU Setup Guide for Ubuntu

## Prerequisites Check

### 1. Verify NVIDIA GPU
```bash
lspci | grep -i nvidia
```

You should see your GPU listed (e.g., "NVIDIA GeForce RTX 3090")

### 2. Check NVIDIA Drivers
```bash
nvidia-smi
```

**If this works**, you have drivers installed. Proceed to step 3.

**If command not found**, install drivers:
```bash
# Ubuntu will auto-detect and recommend drivers
sudo ubuntu-drivers autoinstall

# Or install specific version
sudo apt update
sudo apt install nvidia-driver-535 -y

# Reboot after installation
sudo reboot

# Verify after reboot
nvidia-smi
```

### 3. Install System Dependencies
```bash
sudo apt update
sudo apt install ffmpeg python3-pip -y
```

### 4. Install Python Packages
```bash
pip install -r requirements.txt
```

This installs:
- OpenAI Whisper (original implementation)
- PyTorch with CUDA support
- Audio processing libraries

### 5. Verify CUDA in PyTorch
```bash
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA Available: True
GPU: NVIDIA GeForce RTX 3090
```

---

## Usage

### Basic Usage
```bash
python whisper_transcriber.py audio.webm hi
```

### With Custom Output Directory
```bash
python whisper_transcriber.py interview.mp3 en --output-dir ./transcripts
```

### Specify Different Model Size
```bash
# For faster processing (less accurate)
python whisper_transcriber.py audio.wav es --model medium

# For maximum accuracy (default)
python whisper_transcriber.py audio.wav es --model large-v2
```

---

## Troubleshooting

### Issue: "CUDA is not available"

**Solution 1: Check PyTorch CUDA**
```bash
python3 -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

If CUDA version shows `None`, reinstall PyTorch with CUDA:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Solution 2: Check NVIDIA Drivers**
```bash
nvidia-smi
```

If this fails, reinstall drivers:
```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Issue: "Out of Memory" Error

Your GPU doesn't have enough VRAM. Solutions:

**Option 1: Use smaller model**
```bash
python whisper_transcriber.py audio.webm hi --model medium
```

**Option 2: Process shorter audio chunks** (split your audio file)

---

## Performance

### Expected Processing Times (large-v2 on RTX 3090)
- 1 minute audio: ~5-10 seconds
- 10 minutes audio: ~1-2 minutes
- 1 hour audio: ~6-12 minutes

### GPU Memory Usage
- **large-v2**: ~10GB VRAM
- **medium**: ~5GB VRAM
- **small**: ~2GB VRAM

---

## Supported Languages

Common language codes:
- `ar` - Arabic
- `de` - German
- `en` - English
- `es` - Spanish
- `fr` - French
- `hi` - Hindi
- `id` - Indonesian
- `ja` - Japanese
- `ko` - Korean
- `mr` - Marathi
- `pt` - Portuguese
- `ru` - Russian
- `tr` - Turkish
- `vi` - Vietnamese
- `zh` - Mandarin

Full list: [99 languages supported](https://github.com/openai/whisper#available-models-and-languages)

---

## Production Deployment

### Systemd Service Example

Create `/etc/systemd/system/whisper-transcribe.service`:
```ini
[Unit]
Description=Whisper Transcription Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/whisper
ExecStart=/usr/bin/python3 whisper_transcriber.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable whisper-transcribe
sudo systemctl start whisper-transcribe
```

### Docker Deployment

```dockerfile
FROM nvidia/cuda:12.6.0-cudnn9-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY whisper_transcriber.py .

CMD ["python3", "whisper_transcriber.py"]
```

Build and run:
```bash
docker build -t whisper-gpu .
docker run --gpus all -v $(pwd)/audio:/audio whisper-gpu audio.webm hi
```

---

## Model Download Location

First run downloads ~3GB model to:
```
~/.cache/whisper/
```

To use custom location:
```bash
export WHISPER_MODEL_DIR=/opt/whisper_models/
python whisper_transcriber.py audio.webm hi
```
