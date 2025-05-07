# In scripts/extragere_trasaturi.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm # For progress bars
import utils # Or from utils import load_utterance, ...
import librosa
# --- Configuration ---
# Paths (adjust if your script is elsewhere relative to data)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assumes script is in 'scripts' dir
ANNOTATION_DIR = os.path.join(BASE_DIR, 'data/datasetAnnotation/')
AUDIO_DIR = os.path.join(BASE_DIR, 'data/extrAudio/')

# Feature extraction parameters
TARGET_SR = 16000
FRAME_LENGTH_MS = 25
FRAME_HOP_MS = 10 # Step is 10ms, so overlap is 15ms
NUM_MFCC = 13 # Minimum required

# --- Function to load all annotations (same as your notebook version) ---
def load_all_raw_annotations(annotation_dir):
    all_annotations_list = []
    if not os.path.isdir(annotation_dir):
        print(f"ERROR: Annotation directory not found: {annotation_dir}")
        return pd.DataFrame()
        
    for csv_file in os.listdir(annotation_dir):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(annotation_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                df['audio_filename'] = csv_file.replace('.csv', '.wav')
                if 'trial_lie' in csv_file:
                    df['label'] = 'deceptive'
                elif 'trial_truth' in csv_file:
                    df['label'] = 'sincere'
                else:
                    df['label'] = 'unknown' # Should not happen based on project spec
                all_annotations_list.append(df)
            except Exception as e:
                print(f"Error reading or processing CSV file {file_path}: {e}")
    
    if not all_annotations_list:
        print("No annotation files successfully loaded.")
        return pd.DataFrame()
        
    return pd.concat(all_annotations_list, ignore_index=True)

# --- Helper function for algorithmic features (you will complete this) ---
def extract_algorithmic_features_utterance(utterance_audio, sr, frame_length_ms, frame_hop_ms, num_mfcc):
    # (Implementation as previously discussed - MFCCs, F0, stats, etc.)
    # This is a placeholder - ensure you have the full logic here
    # print(f"    Extracting algorithmic features for utterance of length {len(utterance_audio)/sr:.2f}s...")
    frame_length_samples = int(sr * frame_length_ms / 1000)
    hop_length_samples = int(sr * frame_hop_ms / 1000)
    mfccs = librosa.feature.mfcc(y=utterance_audio, sr=sr, n_mfcc=num_mfcc, 
                                 n_fft=frame_length_samples, hop_length=hop_length_samples,
                                 window='hamming')
    f0, _, _ = librosa.pyin(utterance_audio, fmin=librosa.note_to_hz('C2'), 
                            fmax=librosa.note_to_hz('C7'), frame_length=frame_length_samples)
    f0 = f0[~np.isnan(f0)]
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    f0_mean = np.mean(f0) if len(f0) > 0 else 0.0
    f0_std = np.std(f0) if len(f0) > 0 else 0.0
    feature_vector = np.concatenate((mfccs_mean, mfccs_std, [f0_mean, f0_std]))
    return feature_vector


def main():
    print("Starting feature extraction...")

    # 1. Load ALL raw annotations
    raw_annotations_df = load_all_raw_annotations(ANNOTATION_DIR)

    if raw_annotations_df.empty:
        print("Failed to load raw annotations. Exiting.")
        return
    
    print(f"Loaded {len(raw_annotations_df)} total raw utterances.")
    # Verify necessary columns exist (these are from your CSVs: 'Start time', 'Stop time', 'Speaker', 'Gender')
    required_cols = ['Start time', 'Stop time', 'Speaker', 'Gender', 'audio_filename', 'label']
    if not all(col in raw_annotations_df.columns for col in required_cols):
        print(f"ERROR: Raw annotations DataFrame is missing one or more required columns. Found: {raw_annotations_df.columns.tolist()}")
        print(f"Expected columns like: {required_cols}")
        return

    # 2. Filter out 'TM' Speakers to get the working annotations_df
    annotations_df = raw_annotations_df[raw_annotations_df['Speaker'] != 'TM'].copy()
    if annotations_df.empty and not raw_annotations_df.empty :
        print("Warning: Filtering 'TM' resulted in an empty DataFrame. Check 'Speaker' column values and 'TM' string.")
    elif annotations_df.empty :
        print("Resulting annotations_df is empty (possibly due to empty raw_annotations_df).")
        return

    print(f"Using {len(annotations_df)} utterances after filtering 'TM' speakers.")

    all_algorithmic_features = []
    all_labels = []
    all_speaker_ids = []
    # all_spectrograms = [] # For later

    # 3. Iterate through each subject utterance
    for index, row in tqdm(annotations_df.iterrows(), total=annotations_df.shape[0], desc="Processing utterances"):
        audio_filename = row['audio_filename']
        start_time = row['Start time'] 
        stop_time = row['Stop time']
        label = row['label']
        speaker_id = row['Speaker']
        
        full_audio_path = os.path.join(AUDIO_DIR, audio_filename)

        utterance_audio, sr = utils.load_utterance(full_audio_path, start_time, stop_time, TARGET_SR)

        if utterance_audio is None or sr is None or len(utterance_audio) == 0:
            print(f"Skipping utterance due to loading error or empty audio: {audio_filename} ({start_time}-{stop_time})")
            continue
        
        try:
            algo_features = extract_algorithmic_features_utterance(utterance_audio, sr, 
                                                                   FRAME_LENGTH_MS, FRAME_HOP_MS, NUM_MFCC)
            all_algorithmic_features.append(algo_features)
            all_labels.append(label)
            all_speaker_ids.append(speaker_id)
        except Exception as e:
            print(f"Error extracting algorithmic features for {audio_filename} ({start_time}-{stop_time}): {e}")

    # Convert lists to NumPy arrays (be careful if errors caused Nones to be appended)
    # This simple conversion assumes all feature vectors are valid numpy arrays of the same shape
    if all_algorithmic_features:
        all_algorithmic_features_np = np.array(all_algorithmic_features)
        all_labels_np = np.array(all_labels)
        all_speaker_ids_np = np.array(all_speaker_ids)
        print(f"\nSuccessfully extracted algorithmic features for {len(all_algorithmic_features_np)} utterances.")
        print(f"Shape of features array: {all_algorithmic_features_np.shape}")

        # TODO: Save the extracted features, labels, and speaker IDs
        # Example:
        # output_path = os.path.join(BASE_DIR, 'data/algorithmic_features_data.npz')
        # np.savez(output_path, 
        #          features=all_algorithmic_features_np, 
        #          labels=all_labels_np, 
        #          speakers=all_speaker_ids_np)
        # print(f"Saved data to {output_path}")
    else:
        print("No algorithmic features were extracted.")
        
    # TODO: Implement speaker-specific z-score normalization for algorithmic features.

    print("Feature extraction script finished.")

if __name__ == '__main__':
    main()