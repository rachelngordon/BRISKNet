"""Run a batched grid search split across multiple jobs. Run: python3 grid_search_batch.py --help"""

import argparse
import itertools
import os
import subprocess
from datetime import datetime

import pandas as pd
import yaml

def set_nested_value(d, key_path, value):
    """Sets a value in a nested dictionary using a dot-separated path."""
    keys = key_path.split(".")
    for i, key in enumerate(keys[:-1]):
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value

def run_training(config_path, exp_name):
    """Runs the training script as a subprocess."""
    print("-" * 80)
    print(f"Starting training for experiment: {exp_name}")
    print(f"Using config: {config_path}")
    
    command = [
        "python3", "train.py", # <-- IMPORTANT: RENAME THIS
        "--config", config_path,
        "--exp_name", exp_name
    ]
    
    try:
        # We use check=True to raise an exception if the training script fails
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"Successfully completed experiment: {exp_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!!!!! ERROR in experiment: {exp_name} !!!!!!")
        print(f"Return code: {e.returncode}")
        print("--- STDOUT ---")
        print(e.stdout)
        print("--- STDERR ---")
        print(e.stderr)
        return False

def parse_results(exp_output_dir):
    """Parses the final evaluation metrics from the results CSV."""
    # Your script saves results to eval_dir/eval_metrics.csv
    results_file = os.path.join(exp_output_dir, "eval_results", "eval_metrics.csv")
    
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found at {results_file}")
        return {"ssim": float("nan"), "psnr": float("nan")}  # Return NaN for failed runs
        
    try:
        df = pd.read_csv(results_file)
        # Assuming the DL model's metrics are in the second row (index 1)
        dl_metrics = df.iloc[0]
        # Assuming the format is like '0.9123 ± 0.0123', we take the mean value
        ssim = float(dl_metrics["SSIM"].split("±")[0].strip())
        psnr = float(dl_metrics["PSNR"].split("±")[0].strip())
        return {"ssim": ssim, "psnr": psnr}
    except Exception as e:
        print(f"Error parsing results file {results_file}: {e}")
        return {"ssim": float("nan"), "psnr": float("nan")}




def main():

    # --- NEW: Add argument parsing ---
    parser = argparse.ArgumentParser(description="Run a batch of a grid search.")
    parser.add_argument(
        "--total-batches",
        type=int,
        required=True,
        help="The total number of batches to split the grid into.",
    )
    parser.add_argument(
        "--current-batch",
        type=int,
        required=True,
        help="The index of the batch to run (0-indexed).",
    )
    args = parser.parse_args()

    # 1. DEFINE THE HYPERPARAMETER GRID (as before)
    # param_grid = {
    #     'model.optimizer.lr': [1e-2, 5e-3, 1e-3, 5e-4, 1e-4],
    #     'model.num_layers': [1, 2, 3],
    # }

    # param_grid = {
    #     'model.lambda_L': [0.0005, 0.0025, 0.05],
    #     'model.lambda_S': [0.005, 0.05, 0.1],
    #     'model.lambda_spatial_L': [0.005, 0.05, 0.1],
    #     'model.lambda_spatial_S': [0.005, 0.05, 0.1],
    # }


    param_grid = {
        'model.losses.adj_loss.weight': [0.0001, 0.001, 0.01, 0.1, 1],
        'model.losses.ei_loss.weight': [0, 1, 10, 100]
    }
    
    # Base configuration file
    base_config_file = 'configs/config_mc_lsfp.yaml'
    
    # Create a root directory for this grid search run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    grid_search_root = os.path.join("grid_search_results", f"run_{timestamp}")
    os.makedirs(grid_search_root, exist_ok=True)

    # 2. GENERATE ALL COMBINATIONS
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_runs = len(all_combinations)
    print(f"Total combinations to test: {total_runs}")

    # --- NEW: Select the subset for this batch ---
    runs_per_batch = (total_runs + args.total_batches - 1) // args.total_batches # Ceiling division
    start_index = args.current_batch * runs_per_batch
    end_index = min(start_index + runs_per_batch, total_runs)
    
    param_combinations_for_this_batch = all_combinations[start_index:end_index]

    if not param_combinations_for_this_batch:
        print(f"Batch {args.current_batch} has no jobs. Exiting.")
        return

    print(f"Running batch {args.current_batch}/{args.total_batches-1}, with {len(param_combinations_for_this_batch)} combinations (from index {start_index} to {end_index-1}).")
    
    all_results = []
    
    # 3. LOOP THROUGH COMBINATIONS AND RUN EXPERIMENTS
    for i, params in enumerate(param_combinations_for_this_batch):
        # Create a unique experiment name for this combination
        exp_name_parts = [f"{key.split('.')[-1]}_{val}" for key, val in params.items()]
        exp_name = f"grid_run_{i+1:03d}_" + "_".join(exp_name_parts)
        
        # Load the base config
        with open(base_config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Modify the config with the current set of parameters
        for key, value in params.items():
            set_nested_value(config, key, value)
            
        # Create a temporary directory and config file for this specific run
        run_config_dir = os.path.join(grid_search_root, "configs")
        os.makedirs(run_config_dir, exist_ok=True)
        temp_config_path = os.path.join(run_config_dir, f"{exp_name}_config.yaml")
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
            
        # Define the output directory based on your training script's logic
        # Your script saves to output/exp_name
        exp_output_dir = os.path.join("output", exp_name)

        # Run the training
        success = run_training(temp_config_path, exp_name)
        
        # 4. PARSE AND STORE RESULTS
        if success:
            metrics = parse_results(exp_output_dir)
        else:
            # If training failed, log NaN for metrics
            metrics = {'ssim': float('nan'), 'psnr': float('nan')}
        
        current_run_results = params.copy()
        current_run_results.update(metrics)
        current_run_results['exp_name'] = exp_name
        all_results.append(current_run_results)
        
        # Save results incrementally
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(grid_search_root, "grid_search_summary.csv"), index=False)
        print("\nUpdated summary:")
        print(results_df)

    # 5. SUMMARIZE AND DISPLAY FINAL RESULTS
    print("\n" + "="*80)
    print("GRID SEARCH COMPLETE")
    print("="*80)
    
    final_df = pd.DataFrame(all_results)
    
    # Sort by the metric you care about most (e.g., PSNR, descending)
    final_df = final_df.sort_values(by='ssim', ascending=False)
    
    print("Final Results Summary (sorted by SSIM):")
    print(final_df)
    
    final_summary_path = os.path.join(grid_search_root, "grid_search_summary_final.csv")
    final_df.to_csv(final_summary_path, index=False)
    print(f"\nFull summary saved to: {final_summary_path}")


if __name__ == '__main__':
    main()
