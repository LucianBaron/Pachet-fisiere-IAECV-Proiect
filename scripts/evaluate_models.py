# In scripts/evaluate_models.py

import os
import pandas as pd
import numpy as np

# --- Configuration ---
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TRAINING_RESULTS_FILE = os.path.join(RESULTS_DIR, 'training_results.csv')

def evaluate_model_performance():
    """
    Loads training results, calculates average performance, 
    and identifies the best models and configurations.
    """
    print(f"Loading training results from: {TRAINING_RESULTS_FILE}")
    if not os.path.exists(TRAINING_RESULTS_FILE):
        print(f"ERROR: Training results file not found: {TRAINING_RESULTS_FILE}")
        print("Please run train_models.py first to generate results.")
        return

    try:
        results_df = pd.read_csv(TRAINING_RESULTS_FILE)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    if results_df.empty:
        print("The results file is empty. No data to evaluate.")
        return

    print(f"Loaded {len(results_df)} individual training records.\n")

    # --- Calculate Average Performance Across Folds ---
    # Group by model_type and config_params (or config_id if params are too long/complex for grouping)
    # Using config_params directly might be problematic if they are very long strings or dicts.
    # It's often better to use config_id if it uniquely identifies the hyperparameter set for a model_type.
    # For this example, let's assume config_id + model_type is unique for a set of hyperparameters.
    
    # Ensure 'accuracy' is numeric and handle potential 'notes' or non-numeric accuracies if any
    results_df['accuracy'] = pd.to_numeric(results_df['accuracy'], errors='coerce')
    results_df.dropna(subset=['accuracy'], inplace=True) # Remove rows where accuracy couldn't be parsed

    if results_df.empty:
        print("No valid accuracy data found after cleaning. Cannot evaluate.")
        return

    # Group by model_type and config_id (assuming config_id is unique per model_type for a set of params)
    # If config_id is not globally unique across model_types, you might need to group by model_type first,
    # then by config_id within each model_type.
    # For simplicity, let's assume config_id is specific enough within a model_type from train_models.py.
    
    # We need to ensure that 'config_params' is treated as a stable key for grouping if 'config_id' might repeat across different model types
    # (though in our train_models.py, config_id restarts for each model type).
    # A robust way is to group by 'model_type' and 'config_params'.
    
    # Convert 'config_params' string back to dict for easier reading if needed, but for grouping, string is fine.
    # For this script, we'll primarily use 'config_id' and 'model_type' for aggregation.
    
    # Calculate mean and std deviation of accuracy for each configuration
    # Grouping by model_type and config_id. We also keep config_params for display.
    agg_functions = {'accuracy': ['mean', 'std'], 'config_params': 'first'}
    # Ensure 'config_params' is present before trying to aggregate it
    if 'config_params' not in results_df.columns:
        agg_functions.pop('config_params')
        print("Warning: 'config_params' column not found in results. It will not be displayed for best configs.")


    # Handle cases where a group might have only one fold (std would be NaN)
    # For std, min_count=2 ensures std is NaN if only 1 observation, then fillna(0)
    # For mean, min_count=1 is default
    
    # We need a unique identifier for each configuration run.
    # Let's assume 'model_type' and 'config_id' together identify a unique set of hyperparameters for that model.
    # If 'config_params' is used, ensure it's hashable for groupby.
    
    # Let's make a unique config identifier string for robust grouping if needed
    # results_df['unique_config_str'] = results_df['model_type'] + "_" + results_df['config_id'].astype(str) + "_" + results_df['config_params']
    # For now, let's stick to model_type and config_id as it was in train_models.py
    
    # Group by model_type and config_id
    # We need to ensure that config_id is truly unique for each hyperparameter set *within* a model_type.
    # If config_params is a string representation of a dict, it can be used for grouping.
    
    # Let's assume 'config_params' is the most reliable unique identifier for a configuration's settings.
    # If 'config_params' is not suitable for direct grouping (e.g., too complex),
    # 'config_id' along with 'model_type' should be used.
    # The train_models.py script saves config_params as a string, which is fine for grouping.

    grouped_results = results_df.groupby(['model_type', 'config_params'])

    # Calculate mean and std for accuracy
    avg_performance = grouped_results['accuracy'].agg(['mean', 'std']).reset_index()
    avg_performance.rename(columns={'mean': 'avg_accuracy', 'std': 'std_accuracy'}, inplace=True)
    
    # Fill NaN std (for groups with 1 fold result, which shouldn't happen with 5-fold CV) with 0
    avg_performance['std_accuracy'].fillna(0, inplace=True)
    avg_performance = avg_performance.sort_values(by=['model_type', 'avg_accuracy'], ascending=[True, False])

    print("\n--- Average Performance per Configuration (Top 3 per Model Type) ---")
    for model_type in avg_performance['model_type'].unique():
        print(f"\nModel Type: {model_type}")
        top_configs = avg_performance[avg_performance['model_type'] == model_type].head(3)
        for _, row in top_configs.iterrows():
            print(f"  Config Params: {row['config_params']}")
            print(f"    Avg Accuracy: {row['avg_accuracy']:.4f} (+/- {row['std_accuracy']:.4f})")

    # --- Identify Best Configuration for Each Model Type ---
    best_overall_model_info = {
        'model_type': None,
        'config_params': None,
        'avg_accuracy': -1.0, # Initialize with a very low accuracy
        'std_accuracy': 0.0
    }
    
    print("\n\n--- Best Configuration per Model Type ---")
    for model_type in avg_performance['model_type'].unique():
        best_for_type = avg_performance[avg_performance['model_type'] == model_type].iloc[0] # Already sorted
        print(f"\nBest {model_type}:")
        print(f"  Config Params: {best_for_type['config_params']}")
        print(f"  Avg 5-Fold Accuracy: {best_for_type['avg_accuracy']:.4f}")
        print(f"  Std Dev Accuracy: {best_for_type['std_accuracy']:.4f}")

        if best_for_type['avg_accuracy'] > best_overall_model_info['avg_accuracy']:
            best_overall_model_info['model_type'] = model_type
            best_overall_model_info['config_params'] = best_for_type['config_params']
            best_overall_model_info['avg_accuracy'] = best_for_type['avg_accuracy']
            best_overall_model_info['std_accuracy'] = best_for_type['std_accuracy']

    # --- Determine Overall Best Performing Model ---
    print("\n\n--- Overall Best Performing Model ---")
    if best_overall_model_info['model_type']:
        print(f"Model Type: {best_overall_model_info['model_type']}")
        print(f"Configuration: {best_overall_model_info['config_params']}")
        print(f"Average 5-Fold Accuracy: {best_overall_model_info['avg_accuracy']:.4f}")
        print(f"Standard Deviation: {best_overall_model_info['std_accuracy']:.4f}")
    else:
        print("Could not determine an overall best model from the results.")

    print("\nEvaluation finished.")

if __name__ == '__main__':
    evaluate_model_performance()
