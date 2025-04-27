import random
import numpy as np
import pandas as pd
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
import matplotlib.pyplot as plt

def generate_morris_samples(parameter_ranges, dependent_parameter_names, seed=42, N=10, num_levels=4):
    num_independent_param = len(parameter_ranges) -len(dependent_parameter_names)
    independent_param_names = [k for k in parameter_ranges if k not in dependent_parameter_names]
    problem = {
        "num_vars": num_independent_param,
        "names": independent_param_names,
        "bounds": [[0, 1] for _ in range(num_independent_param)]
    }

    np.random.seed(seed)
    random.seed(seed)
    samples = morris.sample(problem, N=N, num_levels=num_levels)

    # Add a new column for the trajectory number
    trajectory_numbers = np.tile(np.arange(1, N + 1), (problem["num_vars"], 1)).flatten()

    return pd.DataFrame(samples, columns=problem["names"]), trajectory_numbers, problem

def rescale_samples(samples_df, parameter_ranges):
    rescaled = samples_df.copy()
    for param in parameter_ranges:
        bounds = parameter_ranges[param]
        if bounds is None:
            continue

        # Categorical variable: a list of discrete options
        if isinstance(bounds, list) and all(isinstance(x, (int, float)) for x in bounds) and len(set(bounds)) > 2:
            options = sorted(bounds)
            n_options = len(options)
            rescaled[param] = rescaled[param].apply(
                lambda x: options[min(int(x * n_options), n_options - 1)]
            )

        # Continuous variable: (lower, upper)
        elif isinstance(bounds, tuple) and len(bounds) == 2:
            # Making bounds exclusive
            lower = bounds[0] + 1e-8
            upper = bounds[1] - 1e-8
            rescaled[param] = lower + (upper - lower) * rescaled[param]

    return rescaled

def apply_dependent_parameters(df, dependent_parameters):
    df = df.copy()
    for parameter in dependent_parameters:
        new_col = parameter['parameter_name']
        func = parameter['function']
        dependent_param = parameter['dependent_param']
        df[new_col] = df.apply(lambda row: np.random.uniform(*func(row[dependent_param])), axis=1)
    return df

def run_morris_analysis(df_samples, problem, output, num_levels=4):
    output = np.stack(output)
    n_outputs = output.shape[1]

    morris_all = []
    for i in range(n_outputs):
        analysis = morris_analyze.analyze(
            problem=problem,
            X=df_samples.to_numpy(),
            Y=output[:, i],
            conf_level=0.95,
            num_levels=num_levels,
            print_to_console=False
        )
        morris_all.append({
            'mu_star': analysis['mu_star'],
            'sigma': analysis['sigma'],
            'param': problem['names'],
            'output_index': i
        })
    return morris_all

def plot_morris_grid(morris_all, params, n_cols=3, same_axis=True):
    """
    Plot a grid of subplots showing sensitivity (mu_star ± sigma) for each parameter across outputs.

    Parameters:
    - morris_all: list of dicts with keys 'mu_star', 'sigma' (output of Morris analysis for each output index)
    - params: list of parameter names
    - n_cols: number of columns in subplot grid (default is 3)
    - same_axis: bool, if True all subplots share the same y-axis limits
    """
    n_params = len(params)
    n_outputs = len(morris_all)
    n_rows = int(np.ceil(n_params / n_cols))

    # Precompute mu_star ± sigma arrays
    all_mu_star = np.array([out['mu_star'] for out in morris_all])  # shape: (n_outputs, n_params)
    all_sigma = np.array([out['sigma'] for out in morris_all])      # shape: (n_outputs, n_params)

    # Determine global y-axis limits if needed
    if same_axis:
        y_min = np.min(all_mu_star - all_sigma)
        y_max = np.max(all_mu_star + all_sigma)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True)

    for i in range(n_params):
        ax = axes.flat[i]
        mu_star_vals = [out['mu_star'][i] for out in morris_all]
        sigma_vals = [out['sigma'][i] for out in morris_all]

        ax.errorbar(range(n_outputs), mu_star_vals, yerr=sigma_vals, fmt='-o', capsize=3)
        ax.set_title(params[i])
        ax.set_xlabel("Output Index")
        ax.set_ylabel(r"$\mu^*$ ± $\sigma$")
        ax.grid(True)

        if same_axis:
            ax.set_ylim(y_min, y_max)

    # Remove unused axes
    for j in range(n_params, n_rows * n_cols):
        fig.delaxes(axes.flat[j])

    fig.suptitle("Sensitivity for Each Parameter Across Outputs", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def summarize_morris_results(morris_all, problem):
    """
    Summarizes Morris sensitivity analysis across multiple outputs.

    Parameters:
    - morris_all: list of dicts with keys 'mu_star' (list of float) and 'param' (list of str)
    - problem: dict containing 'names' (list of parameter names)

    Returns:
    - df_summary: DataFrame with mean and std of mu_star across outputs, sorted by importance
    """
    params = problem['names']
    mu_star_matrix = np.array([output['mu_star'] for output in morris_all])

    mu_star_mean = mu_star_matrix.mean(axis=0)
    mu_star_std = mu_star_matrix.std(axis=0)

    df_summary = pd.DataFrame({
        'Parameter': params,
        'Mu*_mean': mu_star_mean,
        'Mu*_std': mu_star_std
    })

    df_summary = df_summary.sort_values(by='Mu*_mean', ascending=False).reset_index(drop=True)
    return df_summary