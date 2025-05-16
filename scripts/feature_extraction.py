import os
import sys
import pandas as pd
import numpy as np
import librosa 
from tqdm import tqdm 
from sklearn.preprocessing import StandardScaler # For z-score

# --- Adjust path if utils.py is in the parent directory ---
try:
    import utils 
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import utils

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
ANNOTATION_DIR = os.path.join(BASE_DIR, 'data/datasetAnnotation/')
AUDIO_DIR = os.path.join(BASE_DIR, 'data/extrAudio/')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Output File Names ---
ALGO_FEATURES_FILE = os.path.join(OUTPUT_DIR, 'algorithmic_features_normalized.npz')
SPECTROGRAM_FEATURES_FILE = os.path.join(OUTPUT_DIR, 'spectrogram_features.npz')
NORMALIZED_SPECTROGRAM_FILE = os.path.join(OUTPUT_DIR, 'spectrogram_features_normalized.npz')

# Feature extraction parameters
TARGET_SR = 16000
FRAME_LENGTH_MS = 25
FRAME_HOP_MS = 10 
NUM_MFCC = 13 
# Expected number of algorithmic features: (NUM_MFCC * 2 for mean/std) + (F0_mean + F0_std)
NUM_ALGO_FEATURES = NUM_MFCC * 2 + 2
N_FFT_SPECTRO = 512
NUM_FREQ_BINS_SPECTRO = 257


# --- Function to load all raw annotations and standardize Speaker ID ---
def load_all_raw_annotations(annotation_dir):
    # (Same as previously defined - includes .astype(str) for 'Speaker' column)
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
                if 'Speaker' in df.columns:
                    df['Speaker'] = df['Speaker'].astype(str) 
                else:
                    print(f"Warning: 'Speaker' column not found in {csv_file}")
                    continue
                df['audio_filename'] = csv_file.replace('.csv', '.wav')
                if 'trial_lie' in csv_file: df['label'] = 'deceptive'
                elif 'trial_truth' in csv_file: df['label'] = 'sincere'
                else: df['label'] = 'unknown' 
                all_annotations_list.append(df)
            except Exception as e: print(f"Error reading or processing CSV file {file_path}: {e}")
    if not all_annotations_list:
        print("No annotation files successfully loaded.")
        return pd.DataFrame()
    return pd.concat(all_annotations_list, ignore_index=True)

# --- Helper function for algorithmic features ---
def extract_algorithmic_features_utterance(utterance_audio, sr, frame_length_ms, frame_hop_ms, num_mfcc):
    # (Same as previously defined - MFCCs, F0, stats, error handling)
    if len(utterance_audio) == 0: return None
    frame_length_samples = int(sr * frame_length_ms / 1000)
    hop_length_samples = int(sr * frame_hop_ms / 1000)
    if frame_length_samples == 0 or hop_length_samples == 0: return None
    n_fft = max(frame_length_samples, 2048)
    try:
        mfccs = librosa.feature.mfcc(y=utterance_audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, 
                                     win_length=frame_length_samples, hop_length=hop_length_samples, window='hamming')
    except Exception: mfccs = np.full((num_mfcc, 1), np.nan)
    try:
        f0, voiced_flag, _ = librosa.pyin(utterance_audio, fmin=librosa.note_to_hz('C2'), 
                                          fmax=librosa.note_to_hz('C7'), sr=sr, frame_length=2048)
        f0_voiced = f0[voiced_flag]; f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
    except Exception: f0_voiced = np.array([])
    mfccs_mean = np.nanmean(mfccs, axis=1); mfccs_std = np.nanstd(mfccs, axis=1)
    f0_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0.0
    f0_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 0.0
    feature_vector = np.concatenate((mfccs_mean, mfccs_std, [f0_mean, f0_std]))
    return np.nan_to_num(feature_vector)

# --- Z-Score Normalization Function ---
# You can move this to utils.py if preferred
def normalize_features_per_speaker_script(features_array, speaker_ids_array):
    """
    Applies z-score normalization per speaker to the features array.
    """
    if features_array.size == 0:
        return features_array
        
    print("Normalizing algorithmic features per speaker using z-score...")
    df_for_norm = pd.DataFrame(features_array)
    df_for_norm['Speaker'] = speaker_ids_array
    
    feature_cols = df_for_norm.columns[:-1].tolist() # All columns except 'Speaker'
    
    scaler = StandardScaler()
    
    def scale_group(group):
        # Check if group is empty or has insufficient data for scaling
        if group[feature_cols].empty or len(group[feature_cols]) < 2: 
             # For single sample or empty, std dev is 0 or undefined. Return original or zeroed.
             # StandardScaler handles variance=0 by returning 0s. Or return group as is.
             # For simplicity, let's try to scale; StandardScaler will output 0 for zero variance.
             try:
                group[feature_cols] = scaler.fit_transform(group[feature_cols])
             except ValueError: # if scaler.fit_transform fails on very small/problematic groups
                 pass # Keep original values for this group if scaling fails
             return group
        
        group[feature_cols] = scaler.fit_transform(group[feature_cols])
        return group

    normalized_df = df_for_norm.groupby('Speaker', group_keys=False).apply(scale_group)
    print("Normalization complete.")
    return normalized_df[feature_cols].values


# --- Main Execution Logic ---
def main():
    print("Starting feature extraction script...")
    
    # --- Load and Filter Annotations (common step) ---
    raw_annotations_df = load_all_raw_annotations(ANNOTATION_DIR)
    if raw_annotations_df.empty:
        print("Failed to load raw annotations. Exiting.")
        return
    print(f"Loaded {len(raw_annotations_df)} total raw utterances.")
    required_cols = ['Start time', 'Stop time', 'Speaker', 'Gender', 'audio_filename', 'label']
    if not all(col in raw_annotations_df.columns for col in required_cols):
        print(f"ERROR: Raw annotations DF missing columns. Found: {raw_annotations_df.columns.tolist()}")
        return

    annotations_df = raw_annotations_df[raw_annotations_df['Speaker'] != 'TM'].copy()
    if annotations_df.empty:
        print("Warning: Filtering 'TM' resulted in an empty DataFrame.")
        return
    print(f"Using {len(annotations_df)} utterances after filtering 'TM' speakers.")
    num_unique_subjects = annotations_df['Speaker'].nunique()
    print(f"Found {num_unique_subjects} unique subjects in the filtered data.")
    
    # Add utterance duration if not already present (needed for padding length)
    if 'utterance_duration_sec' not in annotations_df.columns:
        if 'Start time' in annotations_df.columns and 'Stop time' in annotations_df.columns:
            annotations_df['utterance_duration_sec'] = annotations_df['Stop time'] - annotations_df['Start time']
        else:
            print("Error: Cannot calculate utterance durations, 'Start time' or 'Stop time' missing.")
            return
            
    # --- Algorithmic Features Processing ---
    print("\n--- Processing Algorithmic Features ---")
    if os.path.exists(ALGO_FEATURES_FILE):
        print(f"Found existing algorithmic features file: {ALGO_FEATURES_FILE}. Skipping extraction.")
    else:
        print(f"No existing algorithmic features file. Starting extraction...")
        # ... (Existing algorithmic feature extraction and normalization logic) ...
        # (This part remains the same as the previous version, ensuring it saves the .npz)
        all_algorithmic_features = []
        all_labels_algo = [] # Use different list names to avoid confusion
        all_speaker_ids_algo = []
        processed_indices_algo = []

        print("Extracting algorithmic features...")
        for index, row in tqdm(annotations_df.iterrows(), total=annotations_df.shape[0], desc="Algo Features"):
            # ... (load utterance audio) ...
            utterance_audio, sr = utils.load_utterance(os.path.join(AUDIO_DIR, row['audio_filename']), 
                                                       row['Start time'], row['Stop time'], TARGET_SR)
            if utterance_audio is None or sr is None or len(utterance_audio) == 0: continue
            
            algo_features = extract_algorithmic_features_utterance(utterance_audio, sr, 
                                                                   FRAME_LENGTH_MS, FRAME_HOP_MS, NUM_MFCC)
            if algo_features is not None and algo_features.shape[0] == NUM_ALGO_FEATURES:
                all_algorithmic_features.append(algo_features)
                all_labels_algo.append(row['label'])
                all_speaker_ids_algo.append(row['Speaker'])
                processed_indices_algo.append(index)
            # ... (error handling) ...

        if all_algorithmic_features:
            all_algorithmic_features_np = np.array(all_algorithmic_features)
            all_labels_np_algo = np.array(all_labels_algo)
            all_speaker_ids_np_algo = np.array(all_speaker_ids_algo)
            
            normalized_features_np = normalize_features_per_speaker_script(
                all_algorithmic_features_np, all_speaker_ids_np_algo)
            
            if normalized_features_np.size > 0:
                np.savez(ALGO_FEATURES_FILE, features=normalized_features_np, labels=all_labels_np_algo, 
                         speakers=all_speaker_ids_np_algo, indices=np.array(processed_indices_algo))
                print(f"Saved NORMALIZED algorithmic features data to {ALGO_FEATURES_FILE}")
        else:
            print("No algorithmic features extracted.")

    # --- Spectrogram Extraction ---
    print("\n--- Processing Spectrogram Features ---")
    # Define output file for normalized spectrograms
    NORMALIZED_SPECTROGRAM_FILE = os.path.join(OUTPUT_DIR, 'spectrogram_features_normalized.npz')
    
    if os.path.exists(NORMALIZED_SPECTROGRAM_FILE):
        print(f"Found existing normalized spectrogram features file: {NORMALIZED_SPECTROGRAM_FILE}. Skipping extraction.")
    elif os.path.exists(SPECTROGRAM_FEATURES_FILE):
        print(f"Found existing raw spectrogram file: {SPECTROGRAM_FEATURES_FILE}. Loading for normalization...")
        
        # Load existing spectrogram data
        loaded_data = np.load(SPECTROGRAM_FEATURES_FILE, allow_pickle=True)
        
        # Check if normal features array or list-based storage was used
        if 'features' in loaded_data:
            all_spectrograms_np = loaded_data['features']
            print(f"Loaded {all_spectrograms_np.shape[0]} spectrograms with shape {all_spectrograms_np.shape}")
            all_labels_np_spectro = loaded_data['labels']
            all_speaker_ids_np_spectro = loaded_data['speakers']
            processed_indices_spectro = loaded_data['indices']
            
            # Normalize spectrograms
            print("Normalizing spectrograms...")
            # You can choose normalization type: 'per-sample' (default), 'global', or 'min-max'
            # And whether to normalize per speaker (True) or across all samples (False)
            normalized_spectrograms = utils.normalize_spectrograms(
                all_spectrograms_np, 
                normalization_type='per-sample',  # Change as needed
                per_speaker=True,                # Change to False for global normalization
                speaker_ids=all_speaker_ids_np_spectro
            )
            
            # Save normalized spectrograms
            np.savez(NORMALIZED_SPECTROGRAM_FILE,
                     features=normalized_spectrograms,
                     labels=all_labels_np_spectro,
                     speakers=all_speaker_ids_np_spectro,
                     indices=processed_indices_spectro)
            print(f"Saved NORMALIZED spectrogram features to {NORMALIZED_SPECTROGRAM_FILE}")
        else:
            print("Found spectrograms saved as list. This format is not supported for normalization.")
            print("Re-extract spectrograms by deleting the existing file first.")
    else:
        print(f"No existing spectrogram features file. Starting extraction...")
        
        # 1. Determine max utterance length for padding
        if 'utterance_duration_sec' not in annotations_df.columns or annotations_df['utterance_duration_sec'].empty:
            print("Error: Utterance durations needed for padding are missing or empty.")
            return
            
        max_duration_sec = annotations_df['utterance_duration_sec'].max()
        target_samples_spectro = int(np.ceil(max_duration_sec * TARGET_SR)) # Ceil to be safe
        print(f"Max utterance duration: {max_duration_sec:.2f}s. Padding spectrogram inputs to {target_samples_spectro} samples.")

        all_spectrograms = []
        all_labels_spectro = []
        all_speaker_ids_spectro = []
        processed_indices_spectro = []

        # 2. Iterate and Extract Spectrograms
        print("Extracting spectrograms...")
        for index, row in tqdm(annotations_df.iterrows(), total=annotations_df.shape[0], desc="Spectrograms"):
            full_audio_path = os.path.join(AUDIO_DIR, row['audio_filename'])
            if not os.path.exists(full_audio_path): continue # Skip if file missing

            utterance_audio, sr = utils.load_utterance(full_audio_path, row['Start time'], row['Stop time'], TARGET_SR)
            if utterance_audio is None or sr is None or len(utterance_audio) == 0:
                continue
            
            try:
                # Call extract_spectrogram_utterance (ensure it's defined or imported)
                spectrogram = utils.extract_spectrogram_utterance(utterance_audio, sr, 
                                                            target_samples_spectro,
                                                            FRAME_LENGTH_MS, FRAME_HOP_MS,
                                                            N_FFT_SPECTRO, NUM_FREQ_BINS_SPECTRO)
                if spectrogram is not None:
                    all_spectrograms.append(spectrogram)
                    all_labels_spectro.append(row['label'])
                    all_speaker_ids_spectro.append(row['Speaker'])
                    processed_indices_spectro.append(index)
                else:
                    print(f"Warning: Spectrogram extraction failed for {row['audio_filename']}")
            except Exception as e:
                print(f"Error extracting spectrogram for {row['audio_filename']}: {e}")
        
        # 3. Save Spectrograms
        if all_spectrograms:
            # Stack into a 3D NumPy array if all spectrograms have the same shape (they should due to padding)
            try:
                all_spectrograms_np = np.stack(all_spectrograms, axis=0)
                all_labels_np_spectro = np.array(all_labels_spectro)
                all_speaker_ids_np_spectro = np.array(all_speaker_ids_spectro)

                print(f"\nSuccessfully extracted {len(all_spectrograms_np)} spectrograms.")
                print(f"Shape of spectrograms array: {all_spectrograms_np.shape}") # (num_samples, freq_bins, time_frames)

                # Save raw spectrograms first
                np.savez(SPECTROGRAM_FEATURES_FILE, 
                         features=all_spectrograms_np, 
                         labels=all_labels_np_spectro, 
                         speakers=all_speaker_ids_np_spectro,
                         indices=np.array(processed_indices_spectro))
                print(f"Saved raw spectrogram features data to {SPECTROGRAM_FEATURES_FILE}")
                
                # Now normalize and save normalized spectrograms
                print("Normalizing spectrograms...")
                normalized_spectrograms = utils.normalize_spectrograms(
                    all_spectrograms_np,
                    normalization_type='per-sample',  # Change as needed
                    per_speaker=True,                # Change to False for global normalization
                    speaker_ids=all_speaker_ids_np_spectro
                )
                
                np.savez(NORMALIZED_SPECTROGRAM_FILE,
                         features=normalized_spectrograms,
                         labels=all_labels_np_spectro,
                         speakers=all_speaker_ids_np_spectro,
                         indices=np.array(processed_indices_spectro))
                print(f"Saved NORMALIZED spectrogram features to {NORMALIZED_SPECTROGRAM_FILE}")
                
            except ValueError as ve:
                print(f"Error stacking spectrograms. Are they all the same shape? {ve}")
                print("Saving as a list of arrays instead (might require different loading logic).")
                np.savez(SPECTROGRAM_FEATURES_FILE, 
                         features_list=all_spectrograms, # Save as object array (list)
                         labels=np.array(all_labels_spectro), 
                         speakers=np.array(all_speaker_ids_spectro),
                         indices=np.array(processed_indices_spectro))
                print(f"Saved spectrograms (as list) data to {SPECTROGRAM_FEATURES_FILE}")
                print("Note: Normalized version not created because list format doesn't support normalization")
        else:
            print("\nNo spectrograms were successfully extracted.")

    print("\nFeature extraction script finished.")

if __name__ == '__main__':
    main()