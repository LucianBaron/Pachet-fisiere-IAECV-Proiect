# In scripts/extragere_trasaturi.py

import os
import sys
import pandas as pd
import numpy as np
import librosa 
from tqdm import tqdm # Progress bar

# --- Adjust path if utils.py is in the parent directory ---
# This assumes utils.py is in the same directory or the parent directory
try:
    import utils 
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import utils

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Project root
ANNOTATION_DIR = os.path.join(BASE_DIR, 'data/datasetAnnotation/')
AUDIO_DIR = os.path.join(BASE_DIR, 'data/extrAudio/')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data') # Directory to save processed features
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# Feature extraction parameters
TARGET_SR = 16000
FRAME_LENGTH_MS = 25
FRAME_HOP_MS = 10 # Step is 10ms, so overlap is 15ms
NUM_MFCC = 13 # Minimum required

# --- Function to load all raw annotations and standardize Speaker ID ---
def load_all_raw_annotations(annotation_dir):
    """
    Loads all CSV annotation files, standardizes 'Speaker' column to string.
    """
    all_annotations_list = []
    if not os.path.isdir(annotation_dir):
        print(f"ERROR: Annotation directory not found: {annotation_dir}")
        return pd.DataFrame()
        
    print(f"Loading annotations from: {annotation_dir}")
    for csv_file in os.listdir(annotation_dir):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(annotation_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
                
                # --- FIX: Standardize 'Speaker' column to string type ---
                if 'Speaker' in df.columns:
                    df['Speaker'] = df['Speaker'].astype(str) # this ensures all IDs are strings
                else:
                    print(f"Warning: 'Speaker' column not found in {csv_file}")
                    continue # Skip files without Speaker column if necessary
                # --- End of FIX ---

                df['audio_filename'] = csv_file.replace('.csv', '.wav')
                if 'trial_lie' in csv_file:
                    df['label'] = 'deceptive'
                elif 'trial_truth' in csv_file:
                    df['label'] = 'sincere'
                else:
                    df['label'] = 'unknown' 
                all_annotations_list.append(df)
            except Exception as e:
                print(f"Error reading or processing CSV file {file_path}: {e}")
    
    if not all_annotations_list:
        print("No annotation files successfully loaded.")
        return pd.DataFrame()
        
    print(f"Loaded data from {len(all_annotations_list)} annotation files.")
    # Use ignore_index=True for clean indexing
    return pd.concat(all_annotations_list, ignore_index=True)

# --- Helper function for algorithmic features ---
def extract_algorithmic_features_utterance(utterance_audio, sr, frame_length_ms, frame_hop_ms, num_mfcc):
    """
    Extracts algorithmic features (F0, MFCCs stats) for a single utterance.
    """
    if len(utterance_audio) == 0:
        return None # Cannot process empty audio

    # 1. Calculate frame_length and hop_length in samples
    frame_length_samples = int(sr * frame_length_ms / 1000)
    hop_length_samples = int(sr * frame_hop_ms / 1000)

    # Ensure frame length is not zero
    if frame_length_samples == 0 or hop_length_samples == 0:
        print("Warning: Frame or hop length is zero, skipping feature extraction for this utterance.")
        return None
        
    # Use a larger FFT window if frame_length is too small for some features like F0
    n_fft = max(frame_length_samples, 2048) # Example: Ensure at least 2048 for FFT if needed

    # 2. Extract MFCCs
    try:
        mfccs = librosa.feature.mfcc(y=utterance_audio, sr=sr, n_mfcc=num_mfcc, 
                                     n_fft=n_fft, # Use potentially larger n_fft
                                     win_length=frame_length_samples, # Actual window length
                                     hop_length=hop_length_samples,
                                     window='hamming')
    except Exception as e:
        print(f"Error extracting MFCCs: {e}")
        mfccs = np.full((num_mfcc, 1), np.nan) # Handle error case

    # 3. Extract Fundamental Frequency (F0) using pyin
    try:
        # pyin might require a suitable frame length, often larger than typical STFT frames
        f0, voiced_flag, voiced_probs = librosa.pyin(utterance_audio, fmin=librosa.note_to_hz('C2'), 
                                                    fmax=librosa.note_to_hz('C7'), 
                                                    sr=sr,
                                                    frame_length=2048) # Example frame length for pyin
        f0_voiced = f0[voiced_flag] # Consider only voiced frames for stats
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)] # Remove NaNs
    except Exception as e:
        print(f"Error extracting F0: {e}")
        f0_voiced = np.array([]) # Handle error case

    # 4. Apply statistical functions (mean, std dev)
    mfccs_mean = np.nanmean(mfccs, axis=1)
    mfccs_std = np.nanstd(mfccs, axis=1)
    
    f0_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0.0
    f0_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 0.0

    # Concatenate all statistical features into one vector
    # Ensure consistent shape even if F0 fails
    feature_vector = np.concatenate((mfccs_mean, mfccs_std, [f0_mean, f0_std]))
    
    # Handle potential NaNs resulting from calculations on empty/problematic frames
    feature_vector = np.nan_to_num(feature_vector) 
    
    return feature_vector

# --- Main Execution Logic ---
def main():
    print("Starting feature extraction...")

    # 1. Load ALL raw annotations and standardize Speaker ID
    raw_annotations_df = load_all_raw_annotations(ANNOTATION_DIR)

    if raw_annotations_df.empty:
        print("Failed to load raw annotations. Exiting.")
        return
    
    print(f"Loaded {len(raw_annotations_df)} total raw utterances.")
    required_cols = ['Start time', 'Stop time', 'Speaker', 'Gender', 'audio_filename', 'label']
    if not all(col in raw_annotations_df.columns for col in required_cols):
        print(f"ERROR: Raw annotations DataFrame is missing one or more required columns. Found: {raw_annotations_df.columns.tolist()}")
        return

    # 2. Filter out 'TM' Speakers
    annotations_df = raw_annotations_df[raw_annotations_df['Speaker'] != 'TM'].copy()
    if annotations_df.empty:
         if not raw_annotations_df.empty:
            print("Warning: Filtering 'TM' resulted in an empty DataFrame. Check 'Speaker' column values and 'TM' string.")
         else:
             print("Resulting annotations_df is empty.")
         return

    print(f"Using {len(annotations_df)} utterances after filtering 'TM' speakers.")
    
    # Check unique speakers again after standardization and filtering
    num_unique_subjects = annotations_df['Speaker'].nunique()
    print(f"Found {num_unique_subjects} unique subjects after filtering.")


    # --- Lists to store results ---
    all_algorithmic_features = []
    all_labels = []
    all_speaker_ids = []
    processed_indices = [] # Keep track of which rows were processed successfully

    # 3. Iterate through each subject utterance
    print("Extracting algorithmic features...")
    for index, row in tqdm(annotations_df.iterrows(), total=annotations_df.shape[0], desc="Processing utterances"):
        audio_filename = row['audio_filename']
        start_time = row['Start time'] 
        stop_time = row['Stop time']
        label = row['label']
        speaker_id = row['Speaker']
        
        full_audio_path = os.path.join(AUDIO_DIR, audio_filename)

        if not os.path.exists(full_audio_path):
            print(f"Warning: Audio file not found, skipping: {full_audio_path}")
            continue

        # Load utterance using the utility function
        utterance_audio, sr = utils.load_utterance(full_audio_path, start_time, stop_time, TARGET_SR)

        if utterance_audio is None or sr is None:
            print(f"Skipping utterance due to loading error: {audio_filename} ({start_time}-{stop_time})")
            continue
        
        # Extract algorithmic features
        try:
            algo_features = extract_algorithmic_features_utterance(utterance_audio, sr, 
                                                                   FRAME_LENGTH_MS, FRAME_HOP_MS, NUM_MFCC)
            if algo_features is not None:
                all_algorithmic_features.append(algo_features)
                all_labels.append(label) # Store corresponding label and speaker
                all_speaker_ids.append(speaker_id)
                processed_indices.append(index) # Store index of successful row
            else:
                 print(f"Skipping utterance due to feature extraction error/empty audio: {audio_filename} ({start_time}-{stop_time})")

        except Exception as e:
            print(f"Error extracting algorithmic features for {audio_filename} ({start_time}-{stop_time}): {e}")
            # Decide on error handling: skip or append placeholder? Skipping for now.

    # 4. Combine and Save Algorithmic Features
    if all_algorithmic_features:
        all_algorithmic_features_np = np.array(all_algorithmic_features)
        all_labels_np = np.array(all_labels)
        all_speaker_ids_np = np.array(all_speaker_ids)
        
        print(f"\nSuccessfully extracted algorithmic features for {len(all_algorithmic_features_np)} utterances.")
        print(f"Shape of features array: {all_algorithmic_features_np.shape}")

        # Save the extracted features, labels, and speaker IDs
        output_path = os.path.join(OUTPUT_DIR, 'algorithmic_features_data.npz')
        try:
            np.savez(output_path, 
                     features=all_algorithmic_features_np, 
                     labels=all_labels_np, 
                     speakers=all_speaker_ids_np,
                     indices=np.array(processed_indices)) # Save indices too, useful for joining later
            print(f"Saved algorithmic features data to {output_path}")
        except Exception as e:
             print(f"Error saving data to {output_path}: {e}")
             
        # --- TODO: Implement speaker-specific z-score normalization here ---
        # You would typically group the features by speaker ID and apply z-score
        # print("Normalization step needs to be implemented.")

    else:
        print("\nNo algorithmic features were successfully extracted.")
        
    # --- TODO: Implement Spectrogram Extraction ---
    # Similar loop, call a function like extract_spectrogram_utterance
    # Remember padding! Save spectrograms (e.g., as list of np arrays or stacked array)
    print("\nSpectrogram extraction needs to be implemented.")


    print("\nFeature extraction script finished.")

if __name__ == '__main__':
    main()