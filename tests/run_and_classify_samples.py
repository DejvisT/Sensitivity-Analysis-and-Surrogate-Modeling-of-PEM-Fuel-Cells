# python run_and_classify_samples.py --input ../data/raw/hyperbox_sampling/valid_region_sobol_24.pkl --n_samples 500 --test

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

def classify_curve(ifc, ucell, voltage_bounds=(0, 1.23), min_voltage_allowed=-5, early_tolerance=3):
    if not isinstance(ifc, list) or not isinstance(ucell, list):
        return {
            "start_in_range": False,
            "early_values_in_range": False,
            "monotonic": False,
            "minimum_voltage": False,
            "valid": False
        }

    start_in_range = voltage_bounds[0] <= ucell[0] <= voltage_bounds[1]
    early_values_in_range = all(voltage_bounds[0] <= u <= voltage_bounds[1] for u in ucell[:early_tolerance])
    monotonic = all(x >= y for x, y in zip(ucell, ucell[1:]))
    minimum_voltage = ucell[-1] > min_voltage_allowed
    valid = start_in_range and early_values_in_range and monotonic and minimum_voltage

    return {
        "start_in_range": start_in_range,
        "early_values_in_range": early_values_in_range,
        "monotonic": monotonic,
        "minimum_voltage": minimum_voltage,
        "valid": valid
    }

def run_single_sample(param_dict, fixed_params):
    # Drop 'id' before simulation if it exists
    param_dict = {k: v for k, v in param_dict.items() if k != "id"}

    results_df = get_polarisation_curve_samples(
        sampled_parameters=[param_dict],
        fixed_parameters=fixed_params,
        save_path=None
    )

    if isinstance(results_df, pd.DataFrame) and not results_df.empty:
        result = results_df.iloc[0].to_dict()
        classification = classify_curve(result.get("ifc"), result.get("Ucell"))
        result.update(classification)

        print(f"Sample: {param_dict.get('id', '[no id]')}")
        for key, value in classification.items():
            print(f"  {key}: {value}")

        return result
    else:
        return None

def run_parallel_simulations(df, fixed_params, n_cores):
    df = df.copy()
    df["id"] = range(len(df))
    param_list = df.to_dict(orient='records')
    results = []

    with mp.Pool(processes=n_cores) as pool:
        worker = partial(run_single_sample, fixed_params=fixed_params)
        for i, res in enumerate(pool.imap_unordered(worker, param_list), 1):
            results.append(res)
            if i % 10 == 0 or i == len(param_list):
                print(f"Processed {i}/{len(param_list)} samples")

    results = [r for r in results if r is not None]
    return results

def main():
    parser = argparse.ArgumentParser(description="Run AlphaPEM simulations on a Sobol subsample.")
    parser.add_argument('--input', type=str, required=True, help="Path to source .pkl file with configs to sample from")
    parser.add_argument('--n_samples', type=int, required=True, help="Number of configs to sample and simulate")
    parser.add_argument('--test', action='store_true', help="Run test mode (0.1% of n_samples)")
    args = parser.parse_args()

    input_path = args.input
    n_samples = args.n_samples
    test_mode = args.test
    today_str = datetime.today().strftime("%d.%m.%Y")
    save_dir = '../data/raw/hyperbox_sampling'
    save_dir_results = '../data/raw/hyperbox_sampling/results'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir_results, exist_ok=True)

    df = load_pickle(input_path)
    print(f"Loaded {len(df)} configurations from {input_path}")

    if n_samples > len(df):
        raise ValueError(f"Requested {n_samples} samples, but only {len(df)} available.")

    sampled_df = df.sample(n=n_samples, random_state=42)
    remaining_df = df.drop(index=sampled_df.index)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    sampled_path = os.path.join(save_dir, f"{base_name}_sampled_{n_samples}.pkl")
    remaining_path = os.path.join(save_dir, f"middle_not_sampled_yet_until_{today_str}.pkl")

    save_pickle(sampled_df, sampled_path)
    save_pickle(remaining_df, remaining_path)
    print(f"Saved sampled configs to: {sampled_path}")
    print(f"Saved remaining pool to: {remaining_path}")

    fixed_params = build_fixed_parameters()
    n_cores = max(mp.cpu_count() - 1, 1)

    if test_mode:
        test_n = max(1, int(0.001 * len(sampled_df)))
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
