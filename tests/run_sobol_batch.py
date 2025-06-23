# run_sobol_batch.py

# Examples of how to run it: CLI
# python run_sobol_batch.py --input ../data/raw/nathaly_samples_sobol_final.pkl --n_samples 10 --test_n 10
# python run_sobol_batch.py --input ../data/raw/nathaly_samples_sobol_final.pkl --n_samples 100

import os
import sys
import pickle
import glob
import time
import argparse
import multiprocessing as mp
from datetime import datetime
import pandas as pd

# Add AlphaPEM module paths
sys.path.append(os.path.abspath("../external/AlphaPEM/"))
from configuration.settings import current_density_parameters, physical_parameters, computing_parameters, operating_inputs
from modules.display_modules import plot_lambda
from model.AlphaPEM import AlphaPEM

# Add project root for custom code
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.sampling.sampler import get_polarisation_curve_samples, build_fixed_parameters


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def run_worker_batch(worker_id, param_batch, fixed_params, base_name, save_every=10):
    temp_dir = "../data/raw/correct_sobol_sampling/temp"
    os.makedirs(temp_dir, exist_ok=True)
    save_path = os.path.join(temp_dir, f"worker_temp_{base_name}_core{worker_id}.pkl")

    results = []
    for i, param_dict in enumerate(param_batch, 1):
        try:
            df = get_polarisation_curve_samples(
                sampled_parameters=[param_dict],
                fixed_parameters=fixed_params,
                save_path=None
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                results.append(df.iloc[0].to_dict())
        except Exception as e:
            print(f"Worker {worker_id}, Sample {i} failed: {e}")

        if i % save_every == 0:
            pd.DataFrame(results).to_pickle(save_path)
            print(f"Worker {worker_id}: intermediate save with {len(results)} samples to {save_path}")

    if results:
        pd.DataFrame(results).to_pickle(save_path)
        print(f"Worker {worker_id}: final save with {len(results)} samples to {save_path}")


def run_parallel_simulations(df, fixed_params, n_cores, base_name):
    param_list = df.to_dict(orient='records')
    chunk_size = int(len(param_list) / n_cores) + 1
    chunks = [param_list[i:i + chunk_size] for i in range(0, len(param_list), chunk_size)]

    args_list = [
        (worker_id, chunk, fixed_params, base_name)
        for worker_id, chunk in enumerate(chunks)
    ]

    with mp.Pool(processes=n_cores) as pool:
        pool.starmap(run_worker_batch, args_list)

    return None


def main():
    parser = argparse.ArgumentParser(description="Run AlphaPEM simulations on a Sobol subsample.")
    parser.add_argument('--input', type=str, required=True, help="Path to source .pkl file with configs to sample from")
    parser.add_argument('--n_samples', type=str, required=True, help="Number of configs to sample and simulate, or 'all'")
    parser.add_argument('--test_n', type=int, default=None, help="(Optional) Number of configs to test, for quick benchmarking")
    args = parser.parse_args()

    input_path = args.input
    today_str = datetime.today().strftime("%d.%m.%Y")
    save_dir = '../data/raw/correct_sobol_sampling'
    save_dir_results = '../data/raw/correct_sobol_sampling/results'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_results, exist_ok=True)

    df = load_pickle(input_path)
    total_available = len(df)
    print(f"Loaded {total_available} configurations from {input_path}")

    # Parse --n_samples
    if args.n_samples.lower() == "all":
        n_samples = total_available
        use_all = True
    else:
        n_samples = int(args.n_samples)
        use_all = (n_samples == total_available)

    if n_samples > total_available:
        raise ValueError(f"Requested {n_samples} samples, but only {total_available} available.")

    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Test mode
    if args.test_n:
        test_df = df.sample(n=args.test_n, random_state=123).reset_index(drop=True)
        fixed_params = build_fixed_parameters()
        n_cores = max(mp.cpu_count() - 1, 1)

        print(f"\nRunning TEST MODE with {args.test_n} configs using {n_cores} cores...")

        start_test = time.time()
        run_parallel_simulations(test_df, fixed_params, n_cores, base_name)
        end_test = time.time()

        avg_time = (end_test - start_test) / args.test_n
        print(f"\nTest run complete with {args.test_n} samples.")
        print(f"Average time per simulation: {avg_time:.2f} seconds")
        return

    # Full run (not test)
    sampled_df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
    sampled_filename = os.path.join(save_dir, f"{base_name}_sobol_sampled_n{n_samples}_on_{today_str}.pkl")
    save_pickle(sampled_df, sampled_filename)
    print(f"Saved sampled configs to: {sampled_filename}")

    # Only save leftovers if not using all
    if not use_all:
        leftover_df = df.drop(index=sampled_df.index).reset_index(drop=True)
        leftover_filename = os.path.join(save_dir, f"{base_name}_sobol_leftover_n{len(leftover_df)}_on_{today_str}.pkl")
        save_pickle(leftover_df, leftover_filename)
        print(f"Saved leftover configs to: {leftover_filename}")

    fixed_params = build_fixed_parameters()
    n_cores = max(mp.cpu_count() - 1, 1)

    print(f"\nRunning FULL SIMULATION with {n_samples} configs using {n_cores} cores...")

    start = time.time()
    run_parallel_simulations(sampled_df, fixed_params, n_cores, base_name)
    end = time.time()

    runtime_min = (end - start) / 60

    print(f"\nFull run complete. You can now merge worker files in temp to create a final dataset.")
    
    results_path = os.path.join(save_dir_results, f"results_{base_name}_sobol_n{n_samples}_on_{today_str}.pkl")

    temp_dir = os.path.join(save_dir, "temp")
    temp_files = glob.glob(os.path.join(temp_dir, f"worker_temp_{base_name}_core*.pkl"))

    if not temp_files:
        print("No temp files found to merge. Check if workers completed successfully.")
        return

    result_dfs = [pd.read_pickle(f) for f in temp_files]
    merged_results = pd.concat(result_dfs, ignore_index=True)

    merged_results.to_pickle(results_path)

    print(f"\nMerged {len(temp_files)} worker files into:")
    print(f"Saved to: {results_path}")
    print(f"Total simulated samples: {len(merged_results)}")
    print(f"Total time: {runtime_min:.2f} minutes")


if __name__ == '__main__':
    main()


