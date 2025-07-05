import os
import sys
import time
import multiprocessing
# Add project root to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)

from src.analysis.sensitivity import SensitivityAnalyzer
from omegaconf import OmegaConf
from datetime import datetime

sys.path.append(os.path.abspath("external/AlphaPEM/"))

from configuration.settings import current_density_parameters, physical_parameters, computing_parameters, operating_inputs
from modules.display_modules import plot_lambda
from model.AlphaPEM import AlphaPEM


# Add project root for custom code
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
from src.sampling.sampler import get_polarisation_curve_samples, build_fixed_parameters

import multiprocessing as mp
from functools import partial
import pandas as pd
# Function: simulate one sample (no fixed params passed)
def run_single_sample(param_dict):
    try:
        df = get_polarisation_curve_samples([param_dict], fixed_parameters="default", save_path=None, save_every=1)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df.iloc[0].to_dict()
    except Exception as e:
        print(f"❌ Sample failed: {e}")
    return None


# Function: run in parallel
def run_parallel(samples_df, n_cores=4):
    param_dicts = samples_df.to_dict(orient='records')
    total = len(param_dicts)
    print(f"🧠 Using {n_cores} cores for {total} samples.")
    start = time.time()

    results = []
    with mp.get_context("spawn").Pool(processes=n_cores) as pool:
        for i, res in enumerate(pool.imap(run_single_sample, param_dicts), start=1):
            if res is not None:
                results.append(res)
            if i % 100 == 0 or i == total:
                print(f"📈 Processed {i}/{total} samples")

    end = time.time()
    print(f"⏱️ Parallel processing completed in {end - start:.2f} seconds.")
    return results



def main():
    print(f"🕒 Starting simulations at {datetime.now().isoformat(sep=' ', timespec='seconds')}")

    parameter_ranges = OmegaConf.load('param_config.yaml')
    from src.analysis.sensitivity import SensitivityAnalyzer
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=5, seed=42)
    parameters = ['tau', 'epsilon_mc', 'i0_c_ref', 'Tfc', 'kappa_c']
    sample = sampler.random(1000)
    sample = pd.DataFrame(sample, columns=parameters)

    dependent_parameter_names = ['Pc_des']
    dependent_parameters = [{'parameter_name': 'Pc_des', 'function': lambda Pa_des : Pa_des - 20000, 'dependent_param': 'Pa_des'}]

    SA = SensitivityAnalyzer({k: parameter_ranges[k] for k in parameters if k in parameter_ranges}, dependent_parameter_names=None, method='sobol', seed=42, N=1024 , calculate_second_order=True)
    SA.samples_df = sample

    SA.rescale_samples()

    param_dict = {'Pa_des': 1.5e5, "Phi_c_des": 0.6, "Re": 5.70e-7, "e": 5, "epsilon_c": 0.271, "epsilon_gdl": 0.701, "Sc": 2.0, "kappa_co": 27.2, 'Pc_des': 1.3e5}
    fixed_df = pd.DataFrame([param_dict] * 1000)

    samples = pd.concat([SA.samples_df, fixed_df], axis=1)
    
    print(multiprocessing.cpu_count())
    n_cores = max(1, multiprocessing.cpu_count())-1


    # === Run simulations (no fixed_params passed) ===
    print(f"🚀 Running simulations in parallel with {n_cores} cores...")
    results = run_parallel(samples, n_cores=n_cores)

    # === Collect and save results ===
    results_df = pd.DataFrame(results)
    print(f"✅ {len(results_df)} simulations completed successfully.")

    output_path = os.path.join(project_root, 'sampling_test', f'LHS_1000_5param.pkl')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_pickle(output_path)
    print(f"📁 Results saved to: {output_path}")


if __name__ == '__main__':
    mp.set_start_method("forkserver", force=True)
    main()