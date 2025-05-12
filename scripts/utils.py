import librosa
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

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

def normalize_features_per_speaker(features_df, feature_columns, speaker_column='Speaker'):
    """
    Applies z-score normalization per speaker to the specified feature columns.

    Args:
        features_df (pd.DataFrame): DataFrame with features and a speaker ID column.
        feature_columns (list): List of column names to normalize.
        speaker_column (str): Name of the column containing speaker IDs.

    Returns:
        pd.DataFrame: DataFrame with normalized features.
    """
    if features_df.empty:
        return features_df

    print(f"Normalizing features per speaker using z-score for columns: {feature_columns}")
    
    # Ensure feature columns exist
    missing_cols = [col for col in feature_columns if col not in features_df.columns]
    if missing_cols:
        print(f"Warning: The following feature columns are missing and will be skipped: {missing_cols}")
        feature_columns = [col for col in feature_columns if col in features_df.columns]
        if not feature_columns:
            print("No valid feature columns to normalize.")
            return features_df

    # Create a copy to avoid modifying the original DataFrame slice in group-by operations
    normalized_df = features_df.copy()
    
    # Using StandardScaler per group
    scaler = StandardScaler()
    
    def scale_group(group):
        group[feature_columns] = scaler.fit_transform(group[feature_columns])
        return group

    if speaker_column in normalized_df.columns:
        normalized_df = normalized_df.groupby(speaker_column, group_keys=False).apply(scale_group)
        print("Normalization complete.")
    else:
        print(f"Warning: Speaker column '{speaker_column}' not found. Skipping normalization.")
        
    return normalized_df


def extract_spectrogram_utterance(utterance_audio, sr, 
                                  target_samples, # Max length in samples for padding
                                  frame_length_ms=25, 
                                  frame_hop_ms=10, 
                                  n_fft=512, 
                                  num_freq_bins=257):
    """
    Extracts a spectrogram for a single utterance with padding.

    Args:
        utterance_audio (np.ndarray): The audio signal of the utterance.
        sr (int): Sampling rate.
        target_samples (int): Target length in samples for padding.
        frame_length_ms (int): Frame length in milliseconds.
        frame_hop_ms (int): Frame hop (step) in milliseconds.
        n_fft (int): Number of DFT points.
        num_freq_bins (int): Number of frequency bins to keep (e.g., 257 for [0, 8kHz]).

    Returns:
        np.ndarray: The processed spectrogram, or None if error.
    """
    if utterance_audio is None or len(utterance_audio) == 0:
        return None

    # 1. Calculate frame and hop lengths in samples
    frame_length_samples = int(sr * frame_length_ms / 1000)
    hop_length_samples = int(sr * frame_hop_ms / 1000)

    if frame_length_samples == 0 or hop_length_samples == 0:
        print("Warning: Frame or hop length is zero for spectrogram. Skipping.")
        return None

    # 2. Pad the audio signal (right padding)
    if len(utterance_audio) < target_samples:
        padding = target_samples - len(utterance_audio)
        utterance_padded = np.pad(utterance_audio, (0, padding), 'constant')
    else: # Can also truncate if longer, or just use as is if equal
        utterance_padded = utterance_audio[:target_samples]
    
    # 3. Compute STFT
    try:
        D_stft = librosa.stft(utterance_padded, 
                              n_fft=n_fft, 
                              hop_length=hop_length_samples, 
                              win_length=frame_length_samples,
                              window='hamm') # Hamming window as recommended
    except Exception as e:
        print(f"Error during STFT: {e}")
        return None

    # 4. Get magnitude and select frequency bins
    S_mag = np.abs(D_stft)
    S_mag_selected_bins = S_mag[:num_freq_bins, :]

    # 5. Log amplitude scaling (log base 10, then multiply by 10 for dB)
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-10 
    S_log_amplitude_db = 10 * np.log10(S_mag_selected_bins + epsilon) 
    # Alternative using librosa (adjust parameters if needed for exact PDF spec):
    # S_log_amplitude_db = librosa.amplitude_to_db(S_mag_selected_bins, ref=np.max)

    return S_log_amplitude_db