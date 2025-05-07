import librosa
import numpy as np

def load_utterance(audio_file_path, start_time_sec, end_time_sec, target_sr=16000):
    """
    Loads a segment (utterance) from an audio file.

    Args:
        audio_file_path (str): Path to the full audio file.
        start_time_sec (float): Start time of the utterance in seconds.
        end_time_sec (float): End time of the utterance in seconds.
        target_sr (int): The target sampling rate.

    Returns:
        np.ndarray: The audio signal of the utterance.
        int: The sampling rate of the utterance.
    """
    try:
        y_full, sr_orig = librosa.load(audio_file_path, sr=None) # Load original SR first
        
        # Resample if necessary
        if sr_orig != target_sr:
            y_full = librosa.resample(y_full, orig_sr=sr_orig, target_sr=target_sr)
        
        sr = target_sr # Update sr to target_sr after resampling

        start_sample = librosa.time_to_samples(start_time_sec, sr=sr)
        end_sample = librosa.time_to_samples(end_time_sec, sr=sr)
        
        y_utterance = y_full[start_sample:end_sample]
        
        return y_utterance, sr
    except Exception as e:
        print(f"Error loading utterance from {audio_file_path} ({start_time_sec}-{end_time_sec}s): {e}")
        return None, None

