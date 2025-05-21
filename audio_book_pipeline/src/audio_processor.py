import logging
import os
from typing import Dict, Any
import sys
import ffmpeg

logger = logging.getLogger("AudioBookPipeline.AudioProcessor")

def extract_and_prepare_audio(input_path: str, output_wav_path: str, config: Dict[str, Any]) -> bool:
    """
    Converts input video/audio to a properly formatted WAV file.
    
    Args:
        input_path: Path to the input video/audio file.
        output_wav_path: Path where the processed WAV file should be saved.
        config: Dictionary containing audio processing configuration.
        
    Returns:
        True if successful, False otherwise.
    """
    logger.info(f"Starting audio extraction from '{input_path}' to '{output_wav_path}'")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_wav_path), exist_ok=True)
    
    sample_rate = config["audio_processing"]["sample_rate"]
    audio_codec = config["audio_processing"]["audio_codec"]
    
    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                output_wav_path,
                vn=None,
                ar=sample_rate,
                acodec=audio_codec
            )
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        
        logger.info(f"Audio extraction complete. Output saved as '{output_wav_path}'")
        return True
        
    except ffmpeg.Error as e:
        logger.error("Error during FFmpeg execution")
        logger.error("FFmpeg stderr output:")
        try:
            error_message = e.stderr.decode()
            logger.error(error_message)
        except Exception as decode_err:
            logger.error(f"Could not decode stderr: {decode_err}")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error during audio extraction: {str(e)}")
        return False