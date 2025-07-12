# run_sobol_batch_modified.py

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

def run_worker_batch(worker_id, param_batch, fixed_params, base_name, save_every=100):
    temp_dir = "../data/raw/correct_sobol_sampling/temp"
    os.makedirs(temp_dir, exist_ok=True)
    save_path = os.path.join(temp_dir, f"worker_temp_{base_name}_core{worker_id}.pkl")

    results = []
    for i, row in enumerate(param_batch, 1):
        try:
            param_dict = {k: row[k] for k in row if k not in ['config_id', 'index']}
            df = get_polarisation_curve_samples(
                sampled_parameters=[param_dict],
                fixed_parameters=fixed_params,
                save_path=None
            )
            if isinstance(df, pd.DataFrame) and not df.empty:
                result = df.iloc[0].to_dict()
                result['config_id'] = row.get('config_id')
                result['index'] = row.get('index')
                results.append(result)
        except Exception as e:
            print(f"Worker {worker_id}, Sample {i} failed: {e}")

        if i % save_every == 0:
            pd.DataFrame(results).to_pickle(save_path)
            print(f"Worker {worker_id}: intermediate save with {len(results)} samples to {save_path}")

    if results:
        pd.DataFrame(results).to_pickle(save_path)
        print(f"Worker {worker_id}: final save with {len(results)} samples to {save_path}")

def run_parallel_simulations(df, fixed_params, base_name):
    param_list = df.to_dict(orient='records')
    n_cores = mp.cpu_count()
    chunk_size = int(len(param_list) / n_cores) + 1
    chunks = [param_list[i:i + chunk_size] for i in range(0, len(param_list), chunk_size)]

    with mp.Pool(processes=n_cores) as pool:
        results_nested = pool.starmap(
            run_worker_batch,
            [(i, chunk, fixed_params, base_name) for i, chunk in enumerate(chunks)]
        )

    results = [item for sublist in results_nested for item in sublist]
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Run AlphaPEM simulations on a Sobol subsample.")
    parser.add_argument('--input', type=str, required=True, help="Path to source .pkl file with configs to sample from")
    parser.add_argument('--n_samples', type=str, required=True, help="Number of configs to sample and simulate, or 'all'")
    parser.add_argument('--offset', type=int, default=0, help="Starting index for sampling (default is 0)")
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

    if args.n_samples.lower() == "all":
        sampled_df = df.iloc[args.offset:].reset_index(drop=True)
        leftover_df = pd.DataFrame()
    else:
        n_samples = int(args.n_samples)
        end_index = args.offset + n_samples
        if end_index > total_available:
            raise ValueError(f"Requested samples from {args.offset} to {end_index}, but only {total_available} available.")
        sampled_df = df.iloc[args.offset:end_index].copy().reset_index(drop=True)
        leftover_df = pd.concat([df.iloc[:args.offset], df.iloc[end_index:]]).reset_index(drop=True)

    base_name = os.path.splitext(os.path.basename(input_path))[0] + f"_offset{args.offset}"
    sampled_filename = os.path.join(save_dir, f"{base_name}_sobol_sampled_n{len(sampled_df)}_on_{today_str}.pkl")
    leftover_filename = os.path.join(save_dir, f"{base_name}_sobol_leftover_n{len(leftover_df)}_on_{today_str}.pkl")

    # save_pickle(sampled_df, sampled_filename)
    # print(f"Saved sampled configs to: {sampled_filename}")

    if not leftover_df.empty:
        save_pickle(leftover_df, leftover_filename)
        print(f"Saved leftover configs to: {leftover_filename}")

    fixed_params = build_fixed_parameters()

    print(f"\nRunning SIMULATION with {len(sampled_df)} configs using all {mp.cpu_count()} cores...")
    start = time.time()
    results_df = run_parallel_simulations(sampled_df, fixed_params, base_name)
    end = time.time()

    runtime_min = (end - start) / 60
    results_path = os.path.join(save_dir_results, f"results_{base_name}_sobol_n{len(sampled_df)}_on_{today_str}.pkl")

    results_df.to_pickle(results_path)
    print(f"\nSaved final results to: {results_path}")
    print(f"Total simulated samples: {len(results_df)}")
    print(f"Total time: {runtime_min:.2f} minutes")

if __name__ == '__main__':
    main()