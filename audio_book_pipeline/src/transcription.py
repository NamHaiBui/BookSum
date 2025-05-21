import logging
import torch
import copy
from typing import Dict, Any, List, Tuple
import srt
from datetime import timedelta
import os
import whisperx

logger = logging.getLogger("AudioBookPipeline.Transcription")

def load_whisper_model(config: Dict[str, Any]):
    """
    Loads the WhisperX model based on configuration.
    
    Args:
        config: Configuration dictionary containing model settings.
        
    Returns:
        Loaded whisper model or None if loading fails.
    """
    model_name = config["transcription"]["whisper_model_name"]
    device = config["transcription"]["device"]
    compute_type = config["transcription"]["compute_type"]
    
    try:
        logger.info(f"Loading Whisper model: {model_name}")
        
        # Use CUDA if available and requested, otherwise fall back to CPU
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
            
        # Disable TF32 to ensure consistent behavior
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        model = whisperx.load_model(model_name, device=device, compute_type=compute_type)
        logger.info(f"Successfully loaded {model_name} model")
        return model
        
    except Exception as e:
        logger.error(f"Error loading Whisper model: {str(e)}")
        return None

def transcribe_audio(model, audio_path: str, config: Dict[str, Any]):
    """
    Transcribes audio using the provided WhisperX model.
    
    Args:
        model: Loaded WhisperX model.
        audio_path: Path to the audio file.
        config: Configuration dictionary with transcription settings.
        
    Returns:
        Transcription result or None if transcription fails.
    """
    try:
        logger.info(f"Loading audio from {audio_path}")
        audio = whisperx.load_audio(audio_path)
        
        batch_size = config["transcription"]["batch_size"]
        language = config["transcription"]["language"]
        
        logger.info("Starting transcription...")
        result = model.transcribe(
            audio, 
            batch_size=batch_size, 
            language=language, 
            verbose=True, 
            print_progress=True
        )
        
        logger.info(f"Transcription complete. Detected {len(result['segments'])} segments.")
        return {"audio": audio, "result": result}
        
    except Exception as e:
        logger.error(f"Error transcribing audio: {str(e)}")
        return None

def load_alignment_model(language_code: str, device: str):
    """
    Loads the alignment model for forced alignment.
    
    Args:
        language_code: The language code (e.g., 'en').
        device: The device to load the model on ('cuda' or 'cpu').
        
    Returns:
        Tuple of (alignment_model, metadata) or (None, None) on failure.
    """
    try:
        logger.info(f"Loading alignment model for language: {language_code}")
        model_alignment, metadata = whisperx.load_align_model(
            language_code=language_code, 
            device=device
        )
        logger.info("Alignment model loaded successfully")
        return model_alignment, metadata
        
    except Exception as e:
        logger.error(f"Error loading alignment model: {str(e)}")
        return None, None

def align_transcription(segments, model_alignment, metadata_alignment, audio, device: str):
    """
    Aligns the transcription with the audio using the alignment model.
    
    Args:
        segments: Transcription segments from whisper.
        model_alignment: The alignment model.
        metadata_alignment: Metadata for the alignment model.
        audio: Audio data.
        device: The device to run alignment on.
        
    Returns:
        Aligned transcription result or None on failure.
    """
    try:
        logger.info("Starting forced alignment...")
        result_aligned = whisperx.alignment.align(
            segments, 
            model_alignment, 
            metadata_alignment, 
            audio, 
            device
        )
        logger.info(f"Alignment complete. Processed {len(result_aligned['segments'])} segments.")
        return result_aligned
        
    except Exception as e:
        logger.error(f"Error during forced alignment: {str(e)}")
        return None

def word_segment_to_sentence(segments, max_text_len=80):
    """
    Convert word segments to sentences.
    
    Args:
        segments: List of segment dictionaries.
        max_text_len: Maximum text length per segment.
        
    Returns:
        List of sentence segments.
    """
    end_of_sentence_symbols = tuple(['.', '!', '?', ',', ';', ':'])
    sentence_results = []

    current_sentence = {"text": "", "start": 0, "end": 0}
    current_sentence_template = {"text": "", "start": 0, "end": 0}

    for segment in segments:
        if current_sentence["text"] == "":
            current_sentence["start"] = segment["start"]
        current_sentence["text"] += ' ' + segment["text"] + ' '
        current_sentence["end"] = segment["end"]
        if segment["text"][-1] in end_of_sentence_symbols:
            current_sentence["text"] = current_sentence["text"].strip()
            sentence_results.append(copy.deepcopy(current_sentence))
            current_sentence = copy.deepcopy(current_sentence_template)
    
    # Add the last sentence if it exists
    if current_sentence["text"].strip():
        current_sentence["text"] = current_sentence["text"].strip()
        sentence_results.append(copy.deepcopy(current_sentence))
        
    return sentence_results

def sentence_segments_merger(segments, max_text_len=80, max_segment_interval=2):
    """
    Merge sentence segments into longer segments based on length and interval constraints.
    
    Args:
        segments: List of segment dictionaries.
        max_text_len: Maximum text length for a merged segment.
        max_segment_interval: Maximum time gap between segments to merge them.
        
    Returns:
        List of merged segments.
    """
    merged_segments = []
    current_segment = {"text": "", "start": 0, "end": 0}
    current_segment_template = {"text": "", "start": 0, "end": 0}

    for segment in segments:
        if current_segment["text"] == "":
            current_segment["start"] = segment["start"]

        if segment["start"] - current_segment["end"] < max_segment_interval and \
                len(current_segment["text"] + " " + segment['text']) < max_text_len:
            current_segment["text"] += ' ' + segment["text"]
            current_segment["end"] = segment["end"]
        else:
            if current_segment["text"]:
                current_segment["text"] = current_segment["text"].strip()
                merged_segments.append(copy.deepcopy(current_segment))
            current_segment = copy.deepcopy(segment)
    
    # Add the last segment if it exists
    if current_segment["text"].strip():
        current_segment["text"] = current_segment["text"].strip()
        merged_segments.append(copy.deepcopy(current_segment))
        
    return merged_segments

def create_srt(aligned_segments, output_path: str, config: Dict[str, Any]) -> bool:
    """
    Creates an SRT file from the aligned segments.
    
    Args:
        aligned_segments: List of aligned segment dictionaries.
        output_path: Path where the SRT file should be saved.
        config: Configuration with SRT settings.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        logger.info(f"Creating SRT file at {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        max_text_len = config["transcription"]["srt_max_text_len"]
        max_interval_sec = config["transcription"]["srt_max_interval_sec"]
        
        # Process segments
        result_merged = sentence_segments_merger(
            aligned_segments['segments'], 
            max_text_len=max_text_len, 
            max_segment_interval=max_interval_sec
        )
        
        # Create SRT subtitles
        result_srt_list = []
        for i, v in enumerate(result_merged):
            result_srt_list.append(
                srt.Subtitle(
                    index=i, 
                    start=timedelta(seconds=v['start']), 
                    end=timedelta(seconds=v['end']), 
                    content=v['text'].strip()
                )
            )
        
        composed_transcription = srt.compose(result_srt_list)
        
        # Write the SRT file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(composed_transcription)
        
        logger.info(f"SRT file created successfully with {len(result_srt_list)} subtitles")
        return True
        
    except Exception as e:
        logger.error(f"Error creating SRT file: {str(e)}")
        return False

def transcribe_pipeline(input_audio_path: str, output_srt_path: str, config: Dict[str, Any]) -> bool:
    """
    Runs the complete transcription pipeline from audio to SRT.
    
    Args:
        input_audio_path: Path to the input audio file.
        output_srt_path: Path where the SRT file should be saved.
        config: Configuration dictionary.
        
    Returns:
        True if the pipeline completes successfully, False otherwise.
    """
    # Load model
    model = load_whisper_model(config)
    if not model:
        return False
    
    # Transcribe
    transcription_result = transcribe_audio(model, input_audio_path, config)
    if not transcription_result:
        return False
    
    # Get device
    device = config["transcription"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    # Load alignment model
    model_alignment, metadata_alignment = load_alignment_model(
        transcription_result["result"]["language"], 
        device
    )
    if not model_alignment or not metadata_alignment:
        return False
    
    # Align transcription
    result_aligned = align_transcription(
        transcription_result["result"]["segments"],
        model_alignment,
        metadata_alignment,
        transcription_result["audio"],
        device
    )
    if not result_aligned:
        return False
    
    # Create SRT
    srt_success = create_srt(result_aligned, output_srt_path, config)
    if not srt_success:
        return False
    
    logger.info("Transcription pipeline completed successfully")
    return True