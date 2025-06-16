# run_sobol_batch.py

# Run python run_sobol_batch.py --input ../data/raw/sobol_sampling_design/sobol_third_middle.pkl --n_samples 1000
# python run_sobol_batch.py --input ../data/raw/sobol_sampling_configurations_hyper_box_undertermined_restricted_trimmed.pkl --n_samples 237
# or python run_sobol_batch.py --input ../data/raw/sobol_sampling_design/middle_not_sampled_yet_until_08.06.2025.pkl --n_samples 1000 --test
# 
import os
import sys
import pickle
import time
import argparse
import multiprocessing as mp
from functools import partial
from datetime import datetime
import pandas as pd

# Add AlphaPEM module paths
sys.path.append(os.path.abspath("../external/AlphaPEM_v1.0/"))

from configuration.settings import current_density_parameters, physical_parameters, computing_parameters, operating_inputs
from modules.display_modules import plot_lambda
from model.AlphaPEM import AlphaPEM

# Add project root for custom code
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.sampling.sampler import get_polarisation_curve_samples, build_fixed_parameters, sample_parameters, PARAMETER_RANGES


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def run_single_sample(param_dict, fixed_params):
    results_df = get_polarisation_curve_samples(
        sampled_parameters=[param_dict],
        fixed_parameters=fixed_params,
        save_path=None
    )

    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        return results_df.iloc[0].to_dict()
    else:
        return None


def run_parallel_simulations(df, fixed_params, n_cores):
    param_list = df.to_dict(orient='records')
    results = []

    with mp.Pool(processes=n_cores) as pool:
        worker = partial(run_single_sample, fixed_params=fixed_params)
        for i, res in enumerate(pool.imap_unordered(worker, param_list), 1):
            results.append(res)
            if i % 10 == 0 or i == len(param_list):
                print(f"Processed {i}/{len(param_list)} samples")

    # Filter out any failed/None results
    results = [r for r in results if r is not None]
    return results


def main():
    parser = argparse.ArgumentParser(description="Run AlphaPEM simulations on a Sobol subsample.")
    parser.add_argument('--input', type=str, required=True, help="Path to source .pkl file with configs to sample from")
    parser.add_argument('--n_samples', type=int, required=True, help="Number of configs to sample and simulate")
    parser.add_argument('--test', action='store_true', help="Run test mode (5% of n_samples)")
    args = parser.parse_args()

    # Setup
    input_path = args.input
    n_samples = args.n_samples
    test_mode = args.test
    today_str = datetime.today().strftime("%d.%m.%Y")
    save_dir = '../data/raw/sobol_sampling_design'
    save_dir_results = '../data/raw/sobol_sampling_design/results'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_results, exist_ok=True)

    # Load source configs
    df = load_pickle(input_path)
    print(f"Loaded {len(df)} configurations from {input_path}")

    if n_samples > len(df):
        raise ValueError(f"Requested {n_samples} samples, but only {len(df)} available.")

    # Sample and split
    sampled_df = df.sample(n=n_samples, random_state=42)
    remaining_df = df.drop(index=sampled_df.index)

    # Save sampled and remaining configs
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    sampled_path = os.path.join(save_dir, f"{base_name}_sampled_{n_samples}.pkl")
    remaining_path = os.path.join(save_dir, f"middle_not_sampled_yet_until_{today_str}.pkl")

    save_pickle(sampled_df, sampled_path)
    save_pickle(remaining_df, remaining_path)
    print(f"Saved sampled configs to: {sampled_path}")
    print(f"Saved remaining pool to: {remaining_path}")

    # Set up fixed parameters
    fixed_params = build_fixed_parameters()
    n_cores = max(mp.cpu_count() - 1, 1)

    # If test mode: use 0.01% of sample
    if test_mode:
        test_n = int(0.001 * len(sampled_df))
        test_df = sampled_df.sample(n=test_n, random_state=123).reset_index(drop=True)
        print(f"\nRunning TEST MODE with {test_n} configs using {n_cores} cores...")

        start_test = time.time()
        test_results = run_parallel_simulations(test_df, fixed_params, n_cores)
        end_test = time.time()

        avg_time = (end_test - start_test) / test_n
        test_output_path = os.path.join(save_dir_results, f"test_results_{base_name}_{test_n}.pkl")
        save_pickle(test_results, test_output_path)

        print(f"\nTest run complete. Saved to: {test_output_path}")
        print(f"Average time per simulation: {avg_time:.2f} seconds")
    else:
        # Full run
        print(f"\nRunning FULL SIMULATION with {n_samples} configs using {n_cores} cores...")

        start = time.time()
        results = run_parallel_simulations(sampled_df.reset_index(drop=True), fixed_params, n_cores)
        end = time.time()

        runtime_min = (end - start) / 60
        results_path = os.path.join(save_dir_results, f"results_{base_name}_n{n_samples}_{today_str}.pkl")
        save_pickle(results, results_path)

        print(f"\nFull run complete. Saved to: {results_path}")
        print(f"Total time: {runtime_min:.2f} minutes")


if __name__ == '__main__':
    main()
