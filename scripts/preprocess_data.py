# In scripts/preprocess_data.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# --- Adjust path if utils.py or other shared resources are in the parent directory ---
# This assumes utils.py might be in the same directory or the parent directory if needed later
# For now, this script primarily loads data and scikit-learn
try:
    # If you have common loading functions in utils.py you want to reuse
    import utils 
    pass
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import utils
    pass

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Project root
DATA_DIR = os.path.join(BASE_DIR, 'data')
ANNOTATION_DIR_RAW = os.path.join(BASE_DIR, 'data/datasetAnnotation/') # For gender info

# Input file (from extragere_trasaturi.py)
ALGO_FEATURES_FILE = os.path.join(DATA_DIR, 'algorithmic_features_normalized.npz')

# Output file for splits
CV_SPLITS_FILE = os.path.join(DATA_DIR, 'cv_speaker_splits.npz')

N_FOLDS = 5

# --- Function to load raw annotations for gender info ---
# (Similar to the one in extragere_trasaturi.py, but only for speaker/gender)
def get_speaker_gender_mapping(annotation_dir_raw):
    """
    Loads raw annotations to create a mapping from speaker ID to gender.
    Assumes 'Speaker' IDs are already standardized to strings if necessary.
    """
    all_annotations_list = []
    if not os.path.isdir(annotation_dir_raw):
        print(f"ERROR: Raw annotation directory not found: {annotation_dir_raw}")
        return {}
        
    for csv_file in os.listdir(annotation_dir_raw):
        if csv_file.endswith('.csv'):
            file_path = os.path.join(annotation_dir_raw, csv_file)
            try:
                df_temp = pd.read_csv(file_path, usecols=['Speaker', 'Gender'])
                # --- Standardize 'Speaker' column to string type ---
                if 'Speaker' in df_temp.columns:
                    df_temp['Speaker'] = df_temp['Speaker'].astype(str)
                else:
                    continue
                all_annotations_list.append(df_temp)
            except Exception as e:
                print(f"Error reading CSV {file_path} for gender mapping: {e}")
    
    if not all_annotations_list:
        print("No raw annotation files loaded for gender mapping.")
        return {}
        
    combined_df = pd.concat(all_annotations_list, ignore_index=True)
    # Filter out 'TM' and get unique speaker-gender pairs
    speaker_gender_map_df = combined_df[combined_df['Speaker'] != 'TM'].drop_duplicates(subset=['Speaker'])
    return pd.Series(speaker_gender_map_df.Gender.values, index=speaker_gender_map_df.Speaker).to_dict()


def create_speaker_independent_splits():
    """
    Creates and saves 5-fold speaker-independent cross-validation splits.
    Stratifies speakers by gender AND their predominant utterance class.
    """
    print(f"Loading processed data from: {ALGO_FEATURES_FILE}")
    if not os.path.exists(ALGO_FEATURES_FILE):
        print(f"ERROR: Processed features file not found: {ALGO_FEATURES_FILE}")
        return

    data = np.load(ALGO_FEATURES_FILE, allow_pickle=True)
    utterance_speakers = data['speakers'] # Speaker ID for each utterance (ensure these are strings)
    utterance_labels_str = data['labels'] # Label for each utterance (deceptive/sincere)
    
    if len(utterance_speakers) == 0:
        print("No speaker data found in the features file.")
        return

    print(f"Total utterances loaded: {len(utterance_speakers)}")

    # Create a DataFrame from utterance-level data to easily work with speakers and labels
    df_utterances = pd.DataFrame({
        'UtteranceIndex': np.arange(len(utterance_speakers)),
        'SpeakerID': utterance_speakers,
        'Label': utterance_labels_str
    })

    # 1. Get unique speakers and their properties for stratification
    speaker_gender_map = get_speaker_gender_mapping(ANNOTATION_DIR_RAW)
    if not speaker_gender_map:
        print("Could not create speaker-gender mapping. Cannot stratify by gender.")
        return

    unique_speakers_list = sorted(list(df_utterances['SpeakerID'].unique()))
    print(f"Found {len(unique_speakers_list)} unique speakers for splitting.")

    # For each speaker, determine gender and predominant utterance class
    speaker_properties = []
    for speaker_id in unique_speakers_list:
        gender = speaker_gender_map.get(str(speaker_id), 'Unknown') # Ensure speaker_id is string for map lookup
        
        speaker_utterances = df_utterances[df_utterances['SpeakerID'] == speaker_id]
        deceptive_count = (speaker_utterances['Label'] == 'deceptive').sum()
        sincere_count = (speaker_utterances['Label'] == 'sincere').sum()
        
        predominant_class = 'Balanced'
        if deceptive_count > sincere_count:
            predominant_class = 'PredDeceptive'
        elif sincere_count > deceptive_count:
            predominant_class = 'PredSincere'
            
        speaker_properties.append({
            'SpeakerID': speaker_id,
            'Gender': gender,
            'PredominantClass': predominant_class,
            'StratifyLabel': f"{gender}-{predominant_class}" # Combined label for stratification
        })

    df_speaker_meta = pd.DataFrame(speaker_properties)

    # Filter out speakers with 'Unknown' gender if any, as StratifiedGroupKFold needs valid labels
    df_speaker_meta_known_gender = df_speaker_meta[df_speaker_meta['Gender'] != 'Unknown']
    if len(df_speaker_meta_known_gender) < len(df_speaker_meta):
        print(f"Excluding {len(df_speaker_meta) - len(df_speaker_meta_known_gender)} speakers with 'Unknown' gender from CV splitting stratification.")
    
    if df_speaker_meta_known_gender.empty:
        print("No speakers with known gender to perform stratified splitting.")
        return
        
    # Prepare data for StratifiedGroupKFold
    # We want to split utterances (X_utterance_indices)
    # grouped by their speaker (utterance_to_speaker_group_ids)
    # and stratified by the speaker's combined label (utterance_to_speaker_stratify_label)

    # Create a map from SpeakerID to its StratifyLabel
    speaker_to_stratify_label_map = pd.Series(
        df_speaker_meta_known_gender.StratifyLabel.values, 
        index=df_speaker_meta_known_gender.SpeakerID
    ).to_dict()

    # Get the list of speakers we are actually using for splitting
    speakers_for_splitting = df_speaker_meta_known_gender['SpeakerID'].tolist()

    # Filter df_utterances to only include those from speakers_for_splitting
    df_utterances_for_splitting = df_utterances[df_utterances['SpeakerID'].isin(speakers_for_splitting)].copy()
    
    if df_utterances_for_splitting.empty:
        print("No utterances remaining after filtering for speakers with known gender/properties.")
        return

    # Assign the speaker's stratify label to each of their utterances
    df_utterances_for_splitting['SpeakerStratifyLabel'] = df_utterances_for_splitting['SpeakerID'].map(speaker_to_stratify_label_map)

    X_cv = df_utterances_for_splitting['UtteranceIndex'].values # These are the actual indices we want to split
    y_cv_stratify = df_utterances_for_splitting['SpeakerStratifyLabel'].values # Stratify by this speaker property
    groups_cv = df_utterances_for_splitting['SpeakerID'].values # Group by speaker

    # Check if any stratification class has fewer members than N_FOLDS
    stratify_counts = df_utterances_for_splitting.groupby('SpeakerID')['SpeakerStratifyLabel'].first().value_counts()
    if (stratify_counts < N_FOLDS).any():
        print(f"Warning: Some stratification classes for speakers have fewer than {N_FOLDS} members.")
        print(stratify_counts)
        print("StratifiedGroupKFold might raise an error or produce imbalanced folds. Consider alternative stratification or reducing N_FOLDS.")
        # Fallback to GroupKFold if stratification is problematic
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=N_FOLDS)
        print("Falling back to GroupKFold (speaker independence only).")
        # For GroupKFold, 'y' is not used for splitting
        splitter = gkf.split(X_cv, None, groups_cv) 
    else:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        splitter = sgkf.split(X_cv, y_cv_stratify, groups_cv)
        print(f"\nGenerating {N_FOLDS}-fold speaker-independent splits, stratified by speaker gender & predominant class...")


    fold_splits = []
    fold_num = 0
    for train_utterance_indices_in_split, val_utterance_indices_in_split in splitter:
        fold_num += 1
        
        # These indices are relative to X_cv, so they are already original utterance indices
        original_train_indices = X_cv[train_utterance_indices_in_split]
        original_val_indices = X_cv[val_utterance_indices_in_split]

        train_speakers = set(utterance_speakers[original_train_indices])
        val_speakers = set(utterance_speakers[original_val_indices])
        
        print(f"\nFold {fold_num}:")
        print(f"  Train speakers: {len(train_speakers)}, Validation speakers: {len(val_speakers)}")
        print(f"  Train utterances: {len(original_train_indices)}, Validation utterances: {len(original_val_indices)}")
        
        if not train_speakers.isdisjoint(val_speakers):
            print(f"  ERROR: Speaker overlap found in Fold {fold_num}!")
        else:
            print("  Speaker independence: OK")

        train_labels_fold = utterance_labels_str[original_train_indices]
        val_labels_fold = utterance_labels_str[original_val_indices]
        print(f"  Train labels (utterances): D={(train_labels_fold == 'deceptive').sum()}, S={(train_labels_fold == 'sincere').sum()}")
        print(f"  Val labels   (utterances): D={(val_labels_fold == 'deceptive').sum()}, S={(val_labels_fold == 'sincere').sum()}")

        train_speaker_genders = [speaker_gender_map.get(spk, 'Unknown') for spk in train_speakers]
        val_speaker_genders = [speaker_gender_map.get(spk, 'Unknown') for spk in val_speakers]
        print(f"  Train speaker genders: M={train_speaker_genders.count('M')}, F={train_speaker_genders.count('F')}")
        print(f"  Val speaker genders  : M={val_speaker_genders.count('M')}, F={val_speaker_genders.count('F')}")
        
        fold_splits.append({'train_indices': original_train_indices, 'val_indices': original_val_indices})

    try:
        np.savez(CV_SPLITS_FILE, folds=fold_splits)
        print(f"\nSaved {N_FOLDS}-fold cross-validation splits to {CV_SPLITS_FILE}")
    except Exception as e:
        print(f"Error saving CV splits: {e}")


if __name__ == '__main__':
    create_speaker_independent_splits()
    # Example of how to load the splits later:
    # loaded_splits_data = np.load(CV_SPLITS_FILE, allow_pickle=True)
    # loaded_folds = loaded_splits_data['folds']
    # for i, fold_data in enumerate(loaded_folds):
    #     train_indices = fold_data['train_indices']
    #     val_indices = fold_data['val_indices']
    #     print(f"Fold {i+1}: Train size {len(train_indices)}, Val size {len(val_indices)}")