# In scripts/train_models.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # For later
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
import itertools
# Ensure TensorFlow logging is not too verbose (optional)
# tf.get_logger().setLevel('ERROR')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow C++ level warnings
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# --- Adjust path if utils.py or other shared resources are in the parent directory ---
# (Not strictly needed for this script if data is loaded directly)

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results') # Directory to save training results
os.makedirs(RESULTS_DIR, exist_ok=True)

# Input files
ALGO_FEATURES_FILE = os.path.join(DATA_DIR, 'algorithmic_features_normalized.npz')
SPECTROGRAM_FEATURES_FILE = os.path.join(DATA_DIR, 'spectrogram_features.npz') # For CNN later
CV_SPLITS_FILE = os.path.join(DATA_DIR, 'cv_speaker_splits.npz')

# Output file for results
TRAINING_RESULTS_FILE = os.path.join(RESULTS_DIR, 'training_results.csv')

N_FOLDS = 5

def load_data():
    """Loads features, labels, and CV splits."""
    print("Loading data...")
    try:
        algo_data = np.load(ALGO_FEATURES_FILE, allow_pickle=True)
        X_algorithmic = algo_data['features']
        y_labels_str = algo_data['labels'] # Original string labels
        # speaker_ids = algo_data['speakers'] # Not directly needed here if using indices
        # utterance_indices = algo_data['indices'] # Original indices from annotations_df
        
        # Encode labels to numeric (0 and 1)
        label_encoder = LabelEncoder()
        y_numeric = label_encoder.fit_transform(y_labels_str)
        print(f"Labels encoded: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")

        # Spectrograms - load later when implementing CNNs
        X_spectrograms = None
        # if os.path.exists(SPECTROGRAM_FEATURES_FILE):
        #     spectro_data = np.load(SPECTROGRAM_FEATURES_FILE, allow_pickle=True)
        #     X_spectrograms = spectro_data['features'] # or 'features_list'
        #     # Ensure labels align if loaded separately
        
        cv_splits_data = np.load(CV_SPLITS_FILE, allow_pickle=True)
        folds_indices = cv_splits_data['folds']
        
        print("Data loaded successfully.")
        return X_algorithmic, y_numeric, X_spectrograms, folds_indices, label_encoder.classes_
        
    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        print("Please ensure extragere_trasaturi.py and preprocess_data.py have been run successfully.")
        return None, None, None, None, None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None, None, None, None, None

def train_and_evaluate_svm(X_train, y_train, X_val, y_val, config):
    """Trains and evaluates an SVM model for a given configuration."""
    model = SVC(C=config['C'], kernel=config['kernel'], gamma=config.get('gamma', 'scale'), probability=True, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    # SVM predict directly gives class labels, no rounding needed for accuracy
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def train_and_evaluate_rf(X_train, y_train, X_val, y_val, config):
    """Trains and evaluates a Random Forest model for a given configuration."""
    model = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config.get('max_depth', None), # None means nodes are expanded until all leaves are pure or contain less than min_samples_split
        criterion=config.get('criterion', 'gini'),
        min_samples_split=config.get('min_samples_split', 2),
        min_samples_leaf=config.get('min_samples_leaf', 1),
        max_features=config.get('max_features', 'sqrt'), # Changed from 'auto' as 'auto' is deprecated and equivalent to 'sqrt'
        random_state=42, # For reproducibility
        n_jobs=-1 # Use all available cores
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def build_and_train_fcnn(X_train, y_train, X_val, y_val, config, input_shape):
    """Builds, trains, and evaluates an FCNN model for a given configuration."""
    
    model = Sequential()
    # Input layer is implicitly defined by the first Dense layer's input_shape
    # Or, explicitly: model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    
    first_layer = True
    for units in config.get('hidden_units', [64]): # e.g., [64, 32]
        if first_layer:
            model.add(Dense(units, 
                            input_shape=input_shape,
                            activation=config.get('activation', 'relu'),
                            kernel_regularizer=l1_l2(l1=config.get('l1_reg', 0.0), l2=config.get('l2_reg', 0.0))))
            first_layer = False
        else:
            model.add(Dense(units, 
                            activation=config.get('activation', 'relu'),
                            kernel_regularizer=l1_l2(l1=config.get('l1_reg', 0.0), l2=config.get('l2_reg', 0.0))))
        
        if config.get('use_batch_norm', False):
            model.add(BatchNormalization())
        
        if config.get('dropout_rate', 0.0) > 0:
            model.add(Dropout(config['dropout_rate']))
            
    # Output layer: Single neuron, sigmoid activation (as per PDF Source: 100)
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model (PDF Source: 101)
    optimizer = Adam(learning_rate=config.get('learning_rate', 0.001))
    model.compile(optimizer=optimizer, 
                  loss=BinaryCrossentropy(), 
                  metrics=['accuracy'])
    
    # Callbacks (PDF Source: 102)
    callbacks = []
    if config.get('use_early_stopping', True):
        # Adjust patience as needed, could be a hyperparameter
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=config.get('patience', 10), 
                                       restore_best_weights=True, mode='max', verbose=0) 
        callbacks.append(early_stopping)
        
    # Training
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=config.get('epochs', 100), 
                        batch_size=config.get('batch_size', 32),
                        callbacks=callbacks,
                        verbose=0) # 0 for silent, 1 for progress bar, 2 for one line per epoch
    
    # Evaluate using the best weights if early stopping restored them
    # If early stopping didn't trigger or restore_best_weights=False, this uses final weights.
    loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    # Alternative for accuracy if rounding is preferred before metric (PDF Source: 107)
    # y_pred_proba = model.predict(X_val)
    # y_pred = (y_pred_proba > 0.5).astype(int)
    # accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy


def generate_fcnn_configs(limit=100):
    """
    Generates a list of FCNN hyperparameter configurations.
    Aims for >= 100 distinct configurations.
    """
    
    # Define options for each hyperparameter
    hidden_units_options = [
        [64],             # 1 layer, 64 units
        [128],            # 1 layer, 128 units
        [32, 32],         # 2 layers, 32 units each
        [64, 32],         # 2 layers, 64 then 32 units
        [128, 64],        # 2 layers, 128 then 64 units
        [64, 64, 32]      # 3 layers
    ] # 6 options
    
    activation_options = ['relu', 'tanh'] # 2 options
    
    dropout_rate_options = [0.0, 0.2, 0.5] # 3 options (0.0 means no dropout)
    
    l2_reg_options = [0.0, 0.001, 0.0001] # 3 options (0.0 means no L2 regularization)
    
    learning_rate_options = [0.001, 0.0005] # 2 options

    # Fixed parameters for this generation run
    fixed_params = {
        'epochs': 100,  # Max epochs, early stopping will likely stop it sooner
        'batch_size': 32,
        'use_batch_norm': True, # As it's encouraged
        'use_early_stopping': True,
        'patience': 10 # Patience for early stopping
    }

    fcnn_configs_list = []
    
    # Generate combinations using itertools.product
    for hu, act, dr, l2r, lr in itertools.product(
        hidden_units_options,
        activation_options,
        dropout_rate_options,
        l2_reg_options,
        learning_rate_options
    ):
        config = {
            'hidden_units': hu,
            'activation': act,
            'dropout_rate': dr,
            'l2_reg': l2r,
            'learning_rate': lr,
            **fixed_params # Add the fixed parameters
        }
        fcnn_configs_list.append(config)
        
    print(f"Generated {len(fcnn_configs_list)} FCNN configurations.") # Should be 6*2*3*3*2 = 216 configurations
    
    # If you need to strictly limit to just over 100, you could sample:
    import random
    if len(fcnn_configs_list) > limit: # Example limit
        fcnn_configs_list = random.sample(fcnn_configs_list, limit)
        print(f"Sampled down to {len(fcnn_configs_list)} FCNN configurations.")
        
    return fcnn_configs_list


def main():
    X_algorithmic, y_numeric, X_spectrograms, folds_indices, class_names = load_data()

    if X_algorithmic is None: # Check if data loading failed
        print("Algorithmic features could not be loaded. Exiting training.")
        return
    
    input_shape_algo = (X_algorithmic.shape[1],) 


    print(f"\nStarting model training with {N_FOLDS}-fold cross-validation...")
    print(f"Algorithmic features shape: {X_algorithmic.shape}")
    print(f"Labels shape: {y_numeric.shape}")
    if X_spectrograms is not None:
        print(f"Spectrogram features shape: {X_spectrograms.shape}")
    
    all_results = [] # To store dicts of results

    # --- 1. SVM with Algorithmic Features ---
    print("\n--- Training SVM Models ---")
    svm_configs = [
        {'C': 0.1, 'kernel': 'linear'},
        {'C': 1, 'kernel': 'linear'},
        {'C': 10, 'kernel': 'linear'},
        {'C': 1, 'kernel': 'rbf', 'gamma': 0.01},
        {'C': 1, 'kernel': 'rbf', 'gamma': 0.1},
        {'C': 10, 'kernel': 'rbf', 'gamma': 0.01},
        {'C': 10, 'kernel': 'rbf', 'gamma': 0.1},
        {'C': 1, 'kernel': 'poly', 'degree': 2}, # Example, degree for poly
        {'C': 1, 'kernel': 'poly', 'degree': 3},
        {'C': 10, 'kernel': 'sigmoid'}
        # Add more to reach >= 10 configurations as per PDF (Source: 96)
    ]

    for config_idx, svm_config in enumerate(svm_configs):
        print(f"  Training SVM with config {config_idx + 1}/{len(svm_configs)}: {svm_config}")
        fold_accuracies_svm = []
        for fold_num, split_indices in enumerate(folds_indices):
            train_indices = split_indices['train_indices']
            val_indices = split_indices['val_indices']

            X_train_fold_algo = X_algorithmic[train_indices]
            y_train_fold = y_numeric[train_indices]
            X_val_fold_algo = X_algorithmic[val_indices]
            y_val_fold = y_numeric[val_indices]
            
            # Handle cases where a fold might be too small or have only one class after splitting
            if len(np.unique(y_train_fold)) < 2:
                print(f"    Fold {fold_num+1}: Training data has only one class. Skipping this fold for this config.")
                # Or assign a default low accuracy, e.g., 0 or accuracy of majority class if applicable
                fold_accuracies_svm.append(0.0) # Or handle as NaN / skip
                continue

            accuracy = train_and_evaluate_svm(X_train_fold_algo, y_train_fold, X_val_fold_algo, y_val_fold, svm_config)
            print(f"    Fold {fold_num+1}/5 - Val Accuracy: {accuracy:.4f}")
            fold_accuracies_svm.append(accuracy)
            
            # Store detailed result
            all_results.append({
                'model_type': 'SVM',
                'config_id': config_idx,
                'config_params': str(svm_config), # Store as string for CSV
                'fold': fold_num + 1,
                'accuracy': accuracy
            })
        
        if fold_accuracies_svm: # If any folds were processed
            avg_accuracy_svm = np.mean(fold_accuracies_svm)
            print(f"  SVM Config {config_idx+1} Avg CV Accuracy: {avg_accuracy_svm:.4f}\n")
        else:
            print(f"  SVM Config {config_idx+1} had no folds processed successfully.\n")


    # --- 2. Random Forest with Algorithmic Features ---
    print("\n--- Training Random Forest Models ---")
    rf_configs = [
        {'n_estimators': 50, 'max_depth': 5, 'criterion': 'gini'},
        {'n_estimators': 100, 'max_depth': 5, 'criterion': 'gini'},
        {'n_estimators': 100, 'max_depth': 10, 'criterion': 'gini'},
        {'n_estimators': 200, 'max_depth': 10, 'criterion': 'gini', 'min_samples_split': 5},
        {'n_estimators': 100, 'max_depth': None, 'criterion': 'gini', 'min_samples_leaf': 5}, # No max_depth
        {'n_estimators': 50, 'max_depth': 5, 'criterion': 'entropy'},
        {'n_estimators': 100, 'max_depth': 5, 'criterion': 'entropy'},
        {'n_estimators': 100, 'max_depth': 10, 'criterion': 'entropy', 'max_features': 'log2'},
        {'n_estimators': 200, 'max_depth': None, 'criterion': 'entropy', 'min_samples_split': 10},
        {'n_estimators': 300, 'max_depth': 15, 'criterion': 'gini', 'min_samples_leaf': 3}

    ]

    for config_idx, rf_config in enumerate(rf_configs):
        print(f"  Training RF with config {config_idx + 1}/{len(rf_configs)}: {rf_config}")
        fold_accuracies_rf = []
        for fold_num, split_indices in enumerate(folds_indices):
            train_indices = split_indices['train_indices']
            val_indices = split_indices['val_indices']

            X_train_fold_algo = X_algorithmic[train_indices]
            y_train_fold = y_numeric[train_indices]
            X_val_fold_algo = X_algorithmic[val_indices]
            y_val_fold = y_numeric[val_indices]

            if len(np.unique(y_train_fold)) < 2:
                print(f"    Fold {fold_num+1}: RF Training data has only one class. Assigning 0 accuracy.")
                fold_accuracies_rf.append(0.0)
                all_results.append({
                    'model_type': 'RF', 'config_id': config_idx, 'config_params': str(rf_config),
                    'fold': fold_num + 1, 'accuracy': 0.0, 'notes': 'Skipped - single class in train'
                })
                continue

            accuracy = train_and_evaluate_rf(X_train_fold_algo, y_train_fold, X_val_fold_algo, y_val_fold, rf_config)
            print(f"    Fold {fold_num+1}/5 - Val Accuracy: {accuracy:.4f}")
            fold_accuracies_rf.append(accuracy)
            all_results.append({
                'model_type': 'RF',
                'config_id': config_idx,
                'config_params': str(rf_config),
                'fold': fold_num + 1,
                'accuracy': accuracy,
                'notes': ''
            })
            
        if fold_accuracies_rf:
            avg_accuracy_rf = np.mean([acc for acc in fold_accuracies_rf if acc is not None])
            print(f"  RF Config {config_idx+1} Avg CV Accuracy: {avg_accuracy_rf:.4f}\n")

    # --- 3. FCNN with Algorithmic Features ---
    print("\n--- Training FCNN Models ---")
    fcnn_configs = generate_fcnn_configs() # Call the generation function

    for config_idx, fcnn_config in enumerate(fcnn_configs):
        print(f"  Training FCNN with config {config_idx + 1}/{len(fcnn_configs)}: {fcnn_config}")
        fold_accuracies_fcnn = []
        for fold_num, split_indices in enumerate(folds_indices):
            train_indices = split_indices['train_indices']
            val_indices = split_indices['val_indices']

            X_train_fold_algo = X_algorithmic[train_indices]
            y_train_fold = y_numeric[train_indices]
            X_val_fold_algo = X_algorithmic[val_indices]
            y_val_fold = y_numeric[val_indices]

            if len(np.unique(y_train_fold)) < 2:
                # ... (handle single class as before) ...
                fold_accuracies_fcnn.append(0.0)
                all_results.append({
                    'model_type': 'FCNN', 'config_id': config_idx, 'config_params': str(fcnn_config),
                    'fold': fold_num + 1, 'accuracy': 0.0, 'notes': 'Skipped - single class in train'
                })
                continue
            
            y_train_fold_keras = y_train_fold.astype(np.float32)
            y_val_fold_keras = y_val_fold.astype(np.float32)

            # Pass the correctly defined input_shape_algo
            accuracy = build_and_train_fcnn(X_train_fold_algo, y_train_fold_keras, 
                                            X_val_fold_algo, y_val_fold_keras, 
                                            fcnn_config, input_shape_algo) 
            print(f"    Fold {fold_num+1}/5 - Val Accuracy: {accuracy:.4f}")
            fold_accuracies_fcnn.append(accuracy)
            all_results.append({
                'model_type': 'FCNN',
                'config_id': config_idx,
                'config_params': str(fcnn_config),
                'fold': fold_num + 1,
                'accuracy': accuracy,
                'notes': ''
            })
            
        if fold_accuracies_fcnn:
            valid_accuracies = [acc for acc in fold_accuracies_fcnn if acc is not None]
            if valid_accuracies:
                 avg_accuracy_fcnn = np.mean(valid_accuracies)
                 print(f"  FCNN Config {config_idx+1} Avg CV Accuracy: {avg_accuracy_fcnn:.4f}\n")


    # --- 4. CNN with Spectrograms (TODO) ---
    print("\n--- Training CNN Models (TODO) ---")
    # cnn_configs = [ {'filters': [32, 64], 'kernel_size': (3,3)}, ... ] # >= 20 configs
    # Similar to FCNN, but with Conv2D, MaxPooling2D, Flatten layers. Input will be X_spectrograms.

    # --- Save all results ---
    if all_results:
        results_df = pd.DataFrame(all_results)
        try:
            results_df.to_csv(TRAINING_RESULTS_FILE, index=False)
            print(f"\nSaved all training results to: {TRAINING_RESULTS_FILE}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
    else:
        print("\nNo results were generated to save.")
        
    print("\nModel training script finished.")

if __name__ == '__main__':
    main()