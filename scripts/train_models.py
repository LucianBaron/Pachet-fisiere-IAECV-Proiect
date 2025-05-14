# In scripts/train_models.py

import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier # For later
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
import itertools
import argparse
# Ensure TensorFlow logging is not too verbose (optional)
# tf.get_logger().setLevel('ERROR')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TensorFlow C++ level warnings
import random
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
print("--- TensorFlow Device Check ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU if multiple are found
        # and to prevent it from allocating all memory upfront (optional, but can be helpful)
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"Found GPU(s): {gpus}")
        print("TensorFlow should be able to use the GPU.")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(f"Error during GPU setup: {e}")
else:
    print("No GPU found by TensorFlow. Training will use CPU.")
print("-----------------------------")
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
    X_algorithmic, y_numeric, X_spectrograms, folds_indices, class_names_out = None, None, None, None, None
    try:
        algo_data = np.load(ALGO_FEATURES_FILE, allow_pickle=True)
        X_algorithmic = algo_data['features']
        y_labels_str = algo_data['labels'] 
        
        label_encoder = LabelEncoder()
        y_numeric = label_encoder.fit_transform(y_labels_str)
        class_names_out = label_encoder.classes_
        print(f"Labels encoded: {class_names_out} -> {label_encoder.transform(class_names_out)}")

        if os.path.exists(SPECTROGRAM_FEATURES_FILE):
            print(f"Loading spectrograms from: {SPECTROGRAM_FEATURES_FILE}")
            spectro_data = np.load(SPECTROGRAM_FEATURES_FILE, allow_pickle=True)
            # Check if features were saved as a list or stacked array
            if 'features_list' in spectro_data:
                # This case means stacking failed during feature extraction, and they are saved as a list of arrays.
                # This would require different handling (e.g. padding/truncating here or ensuring all are same shape)
                # For now, let's assume 'features' (stacked array) is present.
                print("Warning: Spectrograms were saved as a list. Ensure consistent shapes for batching.")
                X_spectrograms = spectro_data['features_list'] # This might be an object array
            elif 'features' in spectro_data:
                X_spectrograms = spectro_data['features']
            else:
                print("Warning: 'features' or 'features_list' key not found in spectrogram file.")
                X_spectrograms = None
            
            if X_spectrograms is not None and isinstance(X_spectrograms, np.ndarray):
                 # For Conv2D with 'channels_last', ensure there's a channel dimension
                if X_spectrograms.ndim == 3: # (samples, freq_bins, time_frames)
                    X_spectrograms = np.expand_dims(X_spectrograms, axis=-1) # (samples, freq_bins, time_frames, 1)
                print(f"Spectrograms loaded and reshaped to: {X_spectrograms.shape}")
        else:
            print(f"Spectrogram features file not found: {SPECTROGRAM_FEATURES_FILE}")
            X_spectrograms = None
        
        cv_splits_data = np.load(CV_SPLITS_FILE, allow_pickle=True)
        folds_indices = cv_splits_data['folds']
        
        print("Data loading attempt finished.")
        # Ensure all critical data is loaded
        if X_algorithmic is not None and y_numeric is not None and folds_indices is not None:
            print("Algorithmic features, labels, and CV splits loaded successfully.")
        else:
            raise FileNotFoundError("Missing critical data for algorithmic features or CV splits.")


    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}")
        return None, None, None, None, None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None, None, None, None, None
        
    return X_algorithmic, y_numeric, X_spectrograms, folds_indices, class_names_out

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



def build_and_train_cnn(X_train, y_train, X_val, y_val, config, input_shape_spectro):
    """Builds, trains, and evaluates a CNN model."""
    
    model = Sequential()
    model.add(Input(shape=input_shape_spectro)) # Explicit Input layer

    # Convolutional blocks
    for filters, kernel_size, pool_size in config.get('conv_blocks', [(32, (3,3), (2,2))]):
        model.add(Conv2D(filters, kernel_size, 
                         activation=config.get('activation_cnn', 'relu'),
                         padding='same', 
                         kernel_regularizer=l1_l2(l2=config.get('l2_reg_cnn', 0.0))))
        if config.get('use_batch_norm_cnn', True):
            model.add(BatchNormalization())
        if pool_size:
            model.add(MaxPooling2D(pool_size=pool_size))
        if config.get('dropout_rate_cnn', 0.0) > 0.2: 
            model.add(Dropout(config['dropout_rate_cnn'] / 2)) 

    model.add(Flatten())

    # Dense layers
    for units in config.get('dense_units_cnn', [64]):
        model.add(Dense(units, 
                        activation=config.get('activation_cnn', 'relu'),
                        kernel_regularizer=l1_l2(l2=config.get('l2_reg_cnn', 0.0))))
        if config.get('use_batch_norm_cnn', True):
            model.add(BatchNormalization())
        if config.get('dropout_rate_cnn', 0.0) > 0:
            model.add(Dropout(config['dropout_rate_cnn']))
            
    model.add(Dense(1, activation='sigmoid')) # Output layer
    
    # --- ADD THESE LINES FOR DEBUGGING ---
    print("\n--- CNN Model Summary ---")
    model.summary(line_length=120) # Print the model summary
    print("-------------------------\n")
    # --- END OF ADDED LINES ---
    
    optimizer = Adam(learning_rate=config.get('learning_rate_cnn', 0.001))
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])
    
    # ... (rest of the function: tf.data.Dataset creation, model.fit, model.evaluate) ...
    # ... (as previously discussed) ...
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(config.get('batch_size', 32)).prefetch(tf.data.AUTOTUNE)
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(config.get('batch_size', 32)).prefetch(tf.data.AUTOTUNE)
        
    callbacks = []
    if config.get('use_early_stopping', True):
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=config.get('patience', 10),
                                       restore_best_weights=True, mode='max', verbose=0)
        callbacks.append(early_stopping)
        
    history = model.fit(train_dataset, 
                        validation_data=val_dataset,
                        epochs=config.get('epochs', 100), 
                        callbacks=callbacks,
                        verbose=0) 
    
    loss, accuracy = model.evaluate(val_dataset, verbose=0) 
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

    if len(fcnn_configs_list) > limit: # Example limit
        fcnn_configs_list = random.sample(fcnn_configs_list, limit)
        print(f"Sampled down to {len(fcnn_configs_list)} FCNN configurations.")
        
    return fcnn_configs_list

def generate_cnn_configs(limit=20):
    """Generates a list of CNN hyperparameter configurations (>= 20)."""
    cnn_configs_list = []
    
    conv_block_options = [ # Each element is a list of (filters, kernel_size, pool_size) for conv blocks
        [ (32, (3,3), (2,2)) ],                                          # 1 Conv block
        [ (32, (3,3), (2,2)), (64, (3,3), (2,2)) ],                      # 2 Conv blocks
        [ (16, (5,5), (2,2)), (32, (3,3), (2,2)) ],                      # 2 Conv blocks, different kernel
        [ (32, (3,3), (2,2)), (64, (3,3), (2,2)), (128, (3,3), None) ], # 3 Conv blocks, no pooling in last
    ] # 4 options
    
    dense_units_options = [ [64], [128], [64, 32] ] # 3 options
    dropout_rate_options = [0.0, 0.3, 0.5] # 3 options
    learning_rate_options = [0.001, 0.0005] # 2 options
    # L2 can also be varied: l2_reg_options = [0.0, 0.001]

    fixed_cnn_params = {
        'activation_cnn': 'relu',
        'use_batch_norm_cnn': True,
        'l2_reg_cnn': 0.001, # Example fixed L2
        'epochs': 30,
        'batch_size': 32,
        'use_early_stopping': True,
        'patience': 10
    }

    for conv_blocks, dense_units, dr, lr in itertools.product(
        conv_block_options,
        dense_units_options,
        dropout_rate_options,
        learning_rate_options
    ):
        config = {
            'conv_blocks': conv_blocks,
            'dense_units_cnn': dense_units,
            'dropout_rate_cnn': dr,
            'learning_rate_cnn': lr,
            **fixed_cnn_params
        }
        cnn_configs_list.append(config)
        
    # This generates 4 * 3 * 3 * 2 = 72 configurations, which is > 20.
    # You can trim this with random.sample if needed, or add more variations.
    print(f"Generated {len(cnn_configs_list)} CNN configurations.")
    
    
    if len(cnn_configs_list) > limit: # Example limit
        cnn_configs_list = random.sample(cnn_configs_list, limit)
        print(f"Sampled down to {len(cnn_configs_list)} FCNN configurations.")
    return cnn_configs_list




def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Train specified machine learning models.")
    parser.add_argument(
        '--models', 
        nargs='+', 
        choices=['svm', 'rf', 'fcnn', 'cnn', 'all'], 
        default=['all'],
        help="Specify which model(s) to train: svm, rf, fcnn, cnn, or all. (default: all)"
    )
    parser.add_argument(
        '--append_results',
        action='store_true', # This makes it a flag; if present, it's True
        help="Append results to existing CSV file instead of overwriting."
    )
    args = parser.parse_args()

    models_to_train = args.models
    if 'all' in models_to_train:
        models_to_train = ['svm', 'rf', 'fcnn', 'cnn']
    
    print(f"Models to train: {models_to_train}")
    print(f"Append results: {args.append_results}")

    # --- Load Data (common for all models that need it) ---
    X_algorithmic, y_numeric, X_spectrograms, folds_indices, class_names = load_data()

    

    
    if X_algorithmic is None and ('svm' in models_to_train or 'rf' in models_to_train or 'fcnn' in models_to_train):
        print("Algorithmic features could not be loaded. Cannot train SVM, RF, or FCNN. Exiting.")
        return
    if X_spectrograms is None and 'cnn' in models_to_train:
        print("Spectrogram features could not be loaded. Cannot train CNN. Exiting.")
        return


    # print(f"\nStarting model training with {N_FOLDS}-fold cross-validation...")
    print(f"Algorithmic features shape: {X_algorithmic.shape}")
    print(f"Labels shape: {y_numeric.shape}")
    if X_spectrograms is not None:
        print(f"Spectrogram features shape: {X_spectrograms.shape}")
    
    # --- Initialize or Load Results ---
    all_results = []
    if args.append_results and os.path.exists(TRAINING_RESULTS_FILE):
        try:
            existing_results_df = pd.read_csv(TRAINING_RESULTS_FILE)
            all_results = existing_results_df.to_dict('records') # Convert DataFrame rows to list of dicts
            print(f"Loaded {len(all_results)} existing results from {TRAINING_RESULTS_FILE}")
        except Exception as e:
            print(f"Could not load existing results file: {e}. Starting with empty results.")
            all_results = []
    else:
        print("Starting with a fresh results list (or file does not exist/overwrite).")

    # --- 1. SVM with Algorithmic Features ---
    if 'svm' in models_to_train and X_algorithmic is not None:
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
    if 'rf' in models_to_train and X_algorithmic is not None:
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
    if 'fcnn' in models_to_train and X_algorithmic is not None:
        input_shape_algo = (X_algorithmic.shape[1],) 
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


    # --- 4. CNN with Spectrograms ---
    if 'cnn' in models_to_train and X_spectrograms is not None:
        if X_spectrograms is not None and isinstance(X_spectrograms, np.ndarray):
            X_spectrograms = X_spectrograms.astype(np.float32) # Ensure float32
            if X_spectrograms.ndim == 3:
                X_spectrograms = np.expand_dims(X_spectrograms, axis=-1)
            print(f"Spectrograms loaded, ensured float32, and reshaped to: {X_spectrograms.shape}")
            
            input_shape_spectro = X_spectrograms.shape[1:] # (freq_bins, time_frames, channels)
            print(f"Input shape for CNN: {input_shape_spectro}")
            
            cnn_configs = generate_cnn_configs(20) # Generate >= 20 configs

            for config_idx, cnn_config in enumerate(cnn_configs):
                print(f"  Training CNN with config {config_idx + 1}/{len(cnn_configs)}: {cnn_config}")
                fold_accuracies_cnn = []
                for fold_num, split_indices in enumerate(folds_indices):
                    train_indices = split_indices['train_indices']
                    val_indices = split_indices['val_indices']

                    X_train_fold_spectro = X_spectrograms[train_indices]
                    y_train_fold = y_numeric[train_indices] # Use the same numeric labels
                    X_val_fold_spectro = X_spectrograms[val_indices]
                    y_val_fold = y_numeric[val_indices]

                    if len(np.unique(y_train_fold)) < 2:
                        print(f"    Fold {fold_num+1}: CNN Training data has only one class. Assigning 0 accuracy.")
                        fold_accuracies_cnn.append(0.0)
                        all_results.append({
                            'model_type': 'CNN', 'config_id': config_idx, 'config_params': str(cnn_config),
                            'fold': fold_num + 1, 'accuracy': 0.0, 'notes': 'Skipped - single class in train'
                        })
                        continue
                    
                    y_train_fold_keras = y_train_fold.astype(np.float32)
                    y_val_fold_keras = y_val_fold.astype(np.float32)

                    accuracy = build_and_train_cnn(X_train_fold_spectro, y_train_fold_keras,
                                                X_val_fold_spectro, y_val_fold_keras,
                                                cnn_config, input_shape_spectro)
                    print(f"    Fold {fold_num+1}/5 - Val Accuracy: {accuracy:.4f}")
                    fold_accuracies_cnn.append(accuracy)
                    all_results.append({
                        'model_type': 'CNN',
                        'config_id': config_idx,
                        'config_params': str(cnn_config),
                        'fold': fold_num + 1,
                        'accuracy': accuracy,
                        'notes': ''
                    })
                
                if fold_accuracies_cnn:
                    valid_accuracies = [acc for acc in fold_accuracies_cnn if acc is not None]
                    if valid_accuracies:
                        avg_accuracy_cnn = np.mean(valid_accuracies)
                        print(f"  CNN Config {config_idx+1} Avg CV Accuracy: {avg_accuracy_cnn:.4f}\n")
                tf.keras.backend.clear_session() # Clears the Keras session
                del model # Explicitly delete the model
                import gc
                gc.collect() # Trigger garbage collection
        else:
            print("\nSpectrogram features not loaded. Skipping CNN training.")

    # --- Save all results ---
    if all_results: # Only save if there's something to save
        results_df = pd.DataFrame(all_results)
        try:
            # If appending, new results_df contains old + new. If not, it's just new.
            # Either way, save the current results_df, overwriting the file.
            results_df.to_csv(TRAINING_RESULTS_FILE, index=False)
            print(f"\nSaved/Updated all training results to: {TRAINING_RESULTS_FILE}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
    else:
        print("\nNo new results were generated to save.")
        
    print("\nModel training script finished.")

if __name__ == '__main__':
    main()