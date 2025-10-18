"""
GPU-only Whisper V2-Large transcription service.
Detects spoken language and transcribes audio when it matches the intended language.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

import torch
import whisper


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """GPU-only Whisper transcription service using OpenAI's original implementation."""
    
    def __init__(self, model_size: str = "large-v2"):
        """
        Initialize the Whisper model on GPU.
        
        Args:
            model_size: Model size (default: large-v2)
        
        Raises:
            RuntimeError: If CUDA is not available
        """
        self.model_size = model_size
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. This script requires GPU.\n"
                "Please ensure:\n"
                "1. You have an NVIDIA GPU\n"
                "2. NVIDIA drivers are installed (run 'nvidia-smi')\n"
                "3. PyTorch with CUDA is installed correctly"
            )
        
        self.device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using GPU: {gpu_name}")
        logger.info(f"Loading Whisper model: {model_size}")
        
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            logger.info("Model loaded successfully on GPU")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def transcribe_audio(
        self, 
        audio_file_path: str, 
        intended_language: str,
        output_dir: Optional[str] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Transcribe audio file and save transcript if language matches.
        
        Args:
            audio_file_path: Path to the audio file
            intended_language: Expected language code (e.g., 'en', 'es', 'fr', 'hi')
            output_dir: Directory to save transcript (default: same as audio file)
        
        Returns:
            Tuple of (detected_language, transcript_file_path or None)
        
        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file format is invalid
        """
        # Validate input
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        if not audio_path.is_file():
            raise ValueError(f"Path is not a file: {audio_file_path}")
        
        # Normalize language code
        intended_language = intended_language.lower().strip()
        
        logger.info(f"Processing audio file: {audio_file_path}")
        logger.info(f"Intended language: {intended_language}")
        
        try:
            # Transcribe with language detection
            result = self.model.transcribe(
                str(audio_path),
                language=None,  # Auto-detect language
                task="transcribe",
                fp16=True,  # Use FP16 for faster GPU inference
                verbose=False
            )
            
            detected_language = result["language"]
            transcript_text = result["text"].strip()
            
            logger.info(f"Detected language: {detected_language}")
            logger.info(f"Transcript length: {len(transcript_text)} characters")
            
            # Check if detected language matches intended language
            if detected_language == intended_language:
                logger.info("Language match! Generating transcript...")
                
                # Save transcript to file
                transcript_path = self._save_transcript(
                    transcript_text,
                    audio_path,
                    detected_language,
                    output_dir,
                    result
                )
                
                logger.info(f"Transcript saved to: {transcript_path}")
                return detected_language, transcript_path
            else:
                logger.warning(
                    f"Language mismatch: detected '{detected_language}', "
                    f"expected '{intended_language}'. Transcript not saved."
                )
                return detected_language, None
                
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def _save_transcript(
        self, 
        transcript: str, 
        audio_path: Path,
        language: str,
        output_dir: Optional[str],
        result: dict
    ) -> str:
        """Save transcript to a text file with metadata."""
        # Determine output directory
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = audio_path.parent
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = audio_path.stem
        output_filename = f"{base_name}_transcript_{language}_{timestamp}.txt"
        output_path = out_dir / output_filename
        
        # Write transcript with metadata
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Audio File: {audio_path.name}\n")
            f.write(f"Detected Language: {language}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Model: {self.model_size}\n")
            f.write(f"Device: {self.device} ({torch.cuda.get_device_name(0)})\n")
            f.write("-" * 80 + "\n\n")
            f.write(transcript)
            
            # Optional: Add segments with timestamps
            if "segments" in result and result["segments"]:
                f.write("\n\n")
                f.write("=" * 80 + "\n")
                f.write("TIMESTAMPED SEGMENTS\n")
                f.write("=" * 80 + "\n\n")
                for segment in result["segments"]:
                    start = segment["start"]
                    end = segment["end"]
                    text = segment["text"].strip()
                    f.write(f"[{start:.2f}s -> {end:.2f}s] {text}\n")
        
        return str(output_path)


def main():
    """CLI interface for Whisper transcription."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GPU-only Whisper transcription service"
    )
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("intended_language", help="Expected language code (e.g., 'en', 'hi')")
    parser.add_argument("--output-dir", help="Output directory for transcript")
    parser.add_argument("--model", default="large-v2", 
                       choices=["tiny", "base", "small", "medium", "large", "large-v2"],
                       help="Model size (default: large-v2)")
    
    args = parser.parse_args()
    
    try:
        # Initialize transcriber
        transcriber = WhisperTranscriber(model_size=args.model)
        
        # Transcribe
        detected_lang, transcript_path = transcriber.transcribe_audio(
            args.audio_file,
            args.intended_language,
            args.output_dir
        )
        
        print(f"\n{'='*60}")
        print(f"Detected Language: {detected_lang}")
        if transcript_path:
            print(f"✓ Transcript saved: {transcript_path}")
        else:
            print(f"✗ Language mismatch - no transcript saved")
            print(f"  Expected: {args.intended_language}, Got: {detected_lang}")
        print(f"{'='*60}\n")
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print("\nERROR: GPU not available or not configured properly.")
        print("Please check CUDA installation with: nvidia-smi")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
