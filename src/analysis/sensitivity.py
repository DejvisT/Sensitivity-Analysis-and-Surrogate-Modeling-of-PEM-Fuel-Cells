import random
import numpy as np
import pandas as pd
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
import matplotlib.pyplot as plt
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from sklearn.decomposition import PCA

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
    samples = morris.sample(problem, N=N, num_levels=num_levels, seed=42)

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

def apply_dependent_parameters(df, dependent_parameters, seed=42):
    np.random.seed(seed)
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


class SensitivityAnalyzer:
    def __init__(self, parameter_ranges, dependent_parameter_names=None, method='morris',
                 seed=42, N=10, num_levels=4):
        self.parameter_ranges = parameter_ranges
        self.dependent_parameter_names = dependent_parameter_names or []
        self.method = method.lower()
        self.seed = seed
        self.N = N
        self.num_levels = num_levels
        self.problem = self._define_problem()
        self.samples_df = None
        self.trajectory_numbers = None

    def _define_problem(self):
        independent_param_names = [k for k in self.parameter_ranges if k not in self.dependent_parameter_names]
        return {
            "num_vars": len(independent_param_names),
            "names": independent_param_names,
            "bounds": [[0, 1] for _ in independent_param_names]
        }

    def generate_samples(self):
        np.random.seed(self.seed)
        random.seed(self.seed)
        if self.method == 'morris':
            samples = morris.sample(
                self.problem, N=self.N, num_levels=self.num_levels, seed=self.seed
            )
            self.trajectory_numbers = np.tile(np.arange(1, self.N + 1), (self.problem["num_vars"], 1)).flatten()
        elif self.method == 'sobol':
            samples = sobol.sample(
                self.problem, N=self.N, calc_second_order=False, seed=self.seed
            )
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'morris' or 'sobol'.")
        self.samples_df = pd.DataFrame(samples, columns=self.problem["names"])
        return self.samples_df

    def rescale_samples(self):
        df = self.samples_df.copy()
        for param, bounds in self.parameter_ranges.items():
            if bounds is None:
                continue
            if isinstance(bounds, list) and all(isinstance(x, (int, float)) for x in bounds):
                options = sorted(bounds)
                n_options = len(options)
                df[param] = df[param].apply(lambda x: options[min(int(x * n_options), n_options - 1)])
            elif isinstance(bounds, tuple) and len(bounds) == 2:
                lower, upper = bounds[0] + 1e-8, bounds[1] - 1e-8
                df[param] = lower + (upper - lower) * df[param]
        self.samples_df = df
        return df

    def apply_dependent_parameters(self, dependent_parameters):
        np.random.seed(self.seed)
        df = self.samples_df.copy()
        for param in dependent_parameters:
            new_col = param['parameter_name']
            func = param['function']
            dep = param['dependent_param']
            df[new_col] = df.apply(lambda row: np.random.uniform(*func(row[dep])), axis=1)
        self.samples_df = df
        return df
    
    def aggregate_output_function(self, data, aggregation_method):
        if aggregation_method == "sum":
            return data['Ucell'].apply(lambda x: [np.sum(x)])

        elif aggregation_method == "AUC":
            return data.apply(lambda row: np.trapezoid(row['Ucell'], row['ifc']), axis=1)

        elif aggregation_method == "fPCA":
            Ucell_matrix = np.stack(data['Ucell'].apply(np.array)) 
            n_components = 5
            pca = PCA(n_components=n_components)
            scores = pca.fit_transform(Ucell_matrix)
            weights = pca.explained_variance_ratio_[:n_components]
            scalar_outputs = (scores[:, :n_components] * weights).sum(axis=1)
            return scalar_outputs
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
    def run_analysis(self, data, aggregation_method):
        if aggregation_method is None:
            outputs = data['Ucell']
        else:
            outputs = self.aggregate_output_function(data, aggregation_method)

        outputs = np.stack(outputs)
        if outputs.ndim == 1:
            outputs = outputs[:, np.newaxis]
        n_outputs = outputs.shape[1]
        results = []

        if self.method == 'morris':
            for i in range(n_outputs):
                analysis = morris_analyze.analyze(
                    problem=self.problem,
                    X=self.samples_df[self.problem["names"]].to_numpy(),
                    Y=outputs[:, i],
                    conf_level=0.95,
                    num_levels=self.num_levels,
                    print_to_console=False
                )
                results.append({
                    'mu_star': analysis['mu_star'],
                    'sigma': analysis['sigma'],
                    'param': self.problem['names'],
                    'output_index': i
                })
        elif self.method == 'sobol':
            for i in range(n_outputs):
                analysis = sobol_analyze.analyze(
                    problem=self.problem,
                    Y=outputs[:, i],
                    print_to_console=False
                )
                results.append({
                    'S1': analysis['S1'],
                    'S1_conf': analysis['S1_conf'],
                    'ST': analysis['ST'],
                    'ST_conf': analysis['ST_conf'],
                    'param': self.problem['names'],
                    'output_index': i
                })
        else:
            raise ValueError(f"Unknown method '{self.method}'")

        return results
    
    def plot_grid(self, results, n_cols=3, same_axis=True):
        """
        Plot sensitivity (mu_star ± sigma) for each parameter across outputs.
        If there's only one output index, combine all into a single plot.
        """
        mu_all = np.array([d['mu_star'] for d in results])
        sig_all = np.array([d['sigma'] for d in results])
        n_outputs, n_params = mu_all.shape

        if n_outputs == 1:
            plt.figure(figsize=(10, max(2, n_params * 0.5)))
            plt.errorbar(mu_all[0], range(n_params), xerr=sig_all[0], fmt='o', capsize=3)
            plt.yticks(range(n_params), self.problem['names'])
            plt.xlabel(r"$\mu^*$ ± $\sigma$")
            plt.ylabel("Parameter")
            plt.title("Sensitivity (Single Output)")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            return

        n_rows = int(np.ceil(n_params / n_cols))
        xlim = (np.min(mu_all - sig_all), np.max(mu_all + sig_all)) if same_axis else None
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows), sharey=True)

        for i, param in enumerate(self.problem['names']):
            ax = axes.flat[i]
            ax.errorbar(mu_all[:, i], range(n_outputs), xerr=sig_all[:, i], fmt='-o', capsize=3)
            ax.set_title(param)
            ax.set_ylabel("Output Index")
            ax.set_xlabel(r"$\mu^*$ ± $\sigma$")
            ax.grid(True)
            if same_axis:
                ax.set_xlim(xlim)

        for j in range(n_params, len(axes.flat)):
            fig.delaxes(axes.flat[j])

        fig.suptitle("Sensitivity for Each Parameter Across Outputs", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()