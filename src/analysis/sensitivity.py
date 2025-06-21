import random
import numpy as np
import pandas as pd
from SALib.sample import morris
from SALib.analyze import morris as morris_analyze
import matplotlib.pyplot as plt
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import fast_sampler
from SALib.analyze import fast
from sklearn.decomposition import PCA

class SensitivityAnalyzer:
    def __init__(self, parameter_ranges, dependent_parameter_names=None, method='morris',
                 seed=42, N=10, num_levels=4, calculate_second_order=False):
        self.parameter_ranges = parameter_ranges
        self.dependent_parameter_names = dependent_parameter_names or []
        self.method = method.lower()
        self.seed = seed
        self.N = N
        self.num_levels = num_levels
        self.problem = self._define_problem()
        self.samples_df = None
        self.calculate_second_order = calculate_second_order

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
            samples = morris.sample(self.problem, N=self.N, num_levels=self.num_levels, seed=self.seed)

        elif self.method == 'sobol':
            samples = sobol.sample(self.problem, N=self.N, calc_second_order=self.calculate_second_order, seed=self.seed)

        elif self.method == 'fast':
            samples = fast_sampler.sample(self.problem, N=self.N, seed=self.seed)
            
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'morris', 'sobol', 'fast'.")
        self.samples_df = pd.DataFrame(samples, columns=self.problem["names"])
        return self.samples_df

    def rescale_samples(self):
        df = self.samples_df.copy()
        for param, bounds in self.parameter_ranges.items():
            if bounds is None:
                continue
            if len(bounds) > 2:
                options = sorted(bounds)
                n_options = len(options)
                df[param] = df[param].apply(lambda x: options[min(int(x * n_options), n_options - 1)])
            elif len(bounds) == 2:
                lower, upper = bounds[0], bounds[1]
                epsilon = 1e-4
                delta = (upper - lower) * epsilon
                lower += delta
                upper -= delta
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
            df[new_col] = df.apply(lambda row: func(row[dep]), axis=1)
        self.samples_df = df
        return df
    
    def aggregate_output_function(self, data, aggregation_method, by_regions=False,bins=None):
        if by_regions:
            # we validate the bins input
            if bins is None:
                bins = [0, 0.4, 1.6, np.inf]
            else:
                if not isinstance(bins, (list, tuple, np.ndarray)):
                    raise TypeError("`bins` must be a list, tuple, or numpy array.")

                if len(bins) != 4:
                    raise ValueError("`bins` must contain exactly 4 numeric values to define 3 regions.")

                # Ensure all elements are numeric
                if not all(isinstance(b, (int, float, np.integer, np.floating)) for b in bins):
                    raise TypeError("All elements in `bins` must be numeric.")

                # Ensure they are strictly increasing
                if not all(bins[i] < bins[i + 1] for i in range(len(bins) - 1)):
                    raise ValueError("`bins` must be strictly increasing.")
            
            # We define the regiosns based on the bins
            labels = ['activation', 'ohmic', 'mass'] # Regions labels
            grouped = pd.cut(data['ifc'][0], bins=bins, labels=labels, right=False)  # right=False means intervals like [0, 0.4)
            num_bins = grouped.value_counts()
            # Define regions based on the number of bins
            regions=[(np.min(bins),num_bins['activation']),
                    (num_bins['activation']+1, (num_bins['activation']+1) + num_bins['ohmic']),
                    ((num_bins['activation']+1)  + num_bins['ohmic'], -1)]
            # Convert regions to integer tuples
            regions = [(int(start), int(end)) for start, end in regions]


            if aggregation_method == "sum":
                def sum_ucell_regions(ucell, regions):
                    return [ np.sum(ucell[start:end+1]) if end != -1 else np.sum(ucell[start:]) for start, end in regions]
                return data['Ucell'].apply(lambda x: sum_ucell_regions(x, regions) if x is not None else [np.nan]*len(regions))
                

            elif aggregation_method == "AUC":
                def auc_ucell_regions(ucell, ifc, regions):
                    return [
                        np.trapezoid(ifc[start:end+1], x=ucell[start:end+1])
                        if end != -1 else
                        np.trapezoid(ifc[start:], x=ucell[start:])
                        for start, end in regions
                        ]
                def is_valid_array(arr):
                    return isinstance(arr, (list, np.ndarray)) and not pd.isna(arr).all()
                return data.apply(
                                    lambda row: auc_ucell_regions(row['Ucell'], row['ifc'], regions)
                                    if is_valid_array(row['Ucell']) and is_valid_array(row['ifc'])
                                    else [np.nan]*len(regions),
                                    axis=1
                                )
                                                

            elif aggregation_method == "fPCA":
                raise ValueError(f"fPCA method is not compatible with region-based aggregation. Please use 'sum' or 'AUC'.")

        else:
            if aggregation_method == "sum":
                return data['Ucell'].apply(lambda x: np.sum(x))

            elif aggregation_method == "AUC":
                return data.apply(lambda row: np.trapezoid(row['Ucell'], row['ifc']) if row['Ucell'] is not None and row['ifc'] is not None else np.nan, axis=1)

            elif aggregation_method == "fPCA":
                valid_ucell = data['Ucell'].apply(lambda x: x is not None and isinstance(x, (list, np.ndarray)))
                filtered_ucell = data.loc[valid_ucell, 'Ucell']
                Ucell_matrix = np.stack(filtered_ucell.apply(np.array))
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
        elif self.method == 'fast':
            for i in range(n_outputs):
                analysis = fast.analyze(
                    problem=self.problem,
                    Y=outputs[:, i],
                    print_to_console=False
                )
                results.append({
                    'S1': analysis['S1'],
                    'ST': analysis['ST'],
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