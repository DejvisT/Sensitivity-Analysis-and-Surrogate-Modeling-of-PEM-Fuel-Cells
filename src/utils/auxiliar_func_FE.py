import gc
gc.collect()
import numpy as np
import pandas as pd
import os
import sys
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import json
import joblib

# ----------------------------------------------------------------------
# Unique color palette for all input parameters
# ----------------------------------------------------------------------
# Load parameter names
param_config = OmegaConf.load('../param_config.yaml')
parameter_names = list(param_config.keys())  

# Add 'ifc' if it's ever included in plots
if 'ifc' not in parameter_names:
    parameter_names.append('ifc')

# Define a consistent color map for features
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#612b20", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#A6761D", "#ffbb78", "#98df8a", "#393b79"
]

FEATURE_COLOR_MAP = {feat: COLORS[i % len(COLORS)] for i, feat in enumerate(parameter_names)}

# ----------------------------------------------------------------------

def load_cv_results(save_dir='results', run_name='model_run'):
    """
    Load a previously saved model, best hyperparameters, and metrics.
    
    Parameters:
    - save_dir: folder containing the saved files
    - run_name: base name used when saving

    Returns:
    - model: trained sklearn or XGBoost model
    - best_params: dictionary of best hyperparameters
    - metrics: dictionary of evaluation metrics
    """

    model_path = os.path.join(save_dir, f"{run_name}_final_model.pkl")
    params_path = os.path.join(save_dir, f"{run_name}_best_params.json")
    metrics_path = os.path.join(save_dir, f"{run_name}_metrics.json")

    if not all(os.path.exists(p) for p in [model_path, params_path, metrics_path]):
        raise FileNotFoundError("[ERROR] One or more result files not found. Check save_dir and run_name.")

    model = joblib.load(model_path)

    with open(params_path, 'r') as f:
        best_params = json.load(f)

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    print(f"[INFO] Loaded model from {model_path}")
    print(f"[INFO] Loaded hyperparameters from {params_path}")
    print(f"[INFO] Loaded metrics from {metrics_path}")

    return model, best_params, metrics


def plot_shap_bar(shap_df, top_n=13):
    """
    Plot a horizontal bar chart of mean absolute SHAP values for the top features.

    Parameters
    ----------
    shap_df : pd.DataFrame
        DataFrame containing 'feature' and 'mean_abs_shap' columns.
    top_n : int
        Number of top features to display.
    """
    plt.figure(figsize=(8, 5))
    features = shap_df["feature"][:top_n][::-1]
    colors = [FEATURE_COLOR_MAP.get(f, "#cccccc") for f in features]
    plt.barh(features, shap_df["mean_abs_shap"][:top_n][::-1], color=colors)
    plt.xlabel("Mean(|SHAP value|)")
    plt.title("SHAP Feature Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_sobol_convergence_analysis(Y, problem, step=128, max_N=1024, index_type="S1"):
    """
    Run Sobol analysis for increasing sample sizes (powers of 2) to check convergence.

    Parameters
    ----------
    Y : np.ndarray
        Target values for sensitivity analysis.
    problem : dict
        SALib problem definition with parameter names and bounds.
    step : int
        Smallest N to consider (e.g. 128).
    max_N : int
        Largest N to consider (e.g. 1024).
    index_type : str
        "S1" or "ST"

    Returns
    -------
    sobol_convergence : dict
        Dictionary mapping N → {"sobol_df": DataFrame}
    """
    assert index_type in ["S1", "ST"], "index_type must be 'S1' or 'ST'"

    Y = Y.to_numpy()

    D = problem["num_vars"]
    samples_per_N = 2 * D + 2
    N_values = [2**i for i in range(int(np.log2(step)), int(np.log2(max_N)) + 1)]

    sobol_convergence = {}

    for N_i in N_values:
        end_idx = N_i * samples_per_N
        if end_idx > len(Y):
            print(f"[SKIP] Not enough samples for N={N_i}. Needed {end_idx}, but got {len(Y)}.")
            continue

        Y_subset = Y[:end_idx]

        try:
            Si = sobol.analyze(problem, Y_subset, calc_second_order=True, print_to_console=False)
            df_all = Si.to_df()
            df_si = df_all[0] if index_type == "ST" else df_all[1]
            df_si = df_si.copy()
            df_si["feature"] = problem["names"]
            sobol_convergence[N_i] = {"sobol_df": df_si}
        except Exception as e:
            print(f"[FAIL] Sobol failed at N={N_i}: {e}")

    return sobol_convergence

def plot_sobol_index_convergence(sobol_results, index_type="ST", top_k=8, title=None, log_x=False, region=None, step=128, max_N=1024):
    """
    Plot how each feature's Sobol index (S1 or ST) changes as sample size N increases.

    Parameters
    ----------
    sobol_results : dict
        Output from run_sobol_convergence_analysis(). Maps N → dict with key 'sobol_df'.
    index_type : str
        Which index to plot: "S1" or "ST".
    top_k : int
        Number of top features to show based on the highest N.
    title : str or None
        Custom plot title.
    log_x : bool
        Use log scale on the x-axis.
    region : str or None
        Region name to include in the title.
    step : int
        Step size used during convergence (used for ticks).
    max_N : int
        Max base N used during convergence (used for ticks).
    """
    assert index_type in ["S1", "ST"], "index_type must be 'S1' or 'ST'"

    max_N_avail = max(sobol_results)
    df_max = sobol_results[max_N_avail]["sobol_df"]
    top_features = df_max.sort_values(index_type, ascending=False)["feature"].head(top_k).tolist()

    N_vals = sorted(sobol_results.keys())
    data = {f: [] for f in top_features}

    for N in N_vals:
        df = sobol_results[N]["sobol_df"].set_index("feature")
        for f in top_features:
            value = df.loc[f, index_type] if f in df.index else np.nan
            data[f].append(value)

    plt.figure(figsize=(10, 6))
    for f in top_features:
        plt.plot(N_vals, data[f], marker="o", label=f, color=FEATURE_COLOR_MAP.get(f, "#cccccc"))

    plt.ylabel(f"{index_type} Sobol Index", fontsize = 16)
    plt.xlabel("Sample Size N", fontsize = 16)

    # Force x-tick alignment
    full_N_ticks = list(range(step, max_N + 1, step))
    plt.xticks(full_N_ticks, fontsize = 12)

    plt.ylim(0, 1.05)
    if title is None:
        title = f"{index_type} Index Convergence for {region.capitalize()} Region" if region else f"{index_type} Index Convergence"
    plt.title(title, fontsize=18)

    if log_x:
        plt.xscale("log")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0), fontsize=11)
    plt.tight_layout()
    plt.show()

def plot_sobol_ranking(sobol_results, top_n=10):
    """
    Plot top first-order Sobol indices as horizontal bar plots for each N.

    Parameters
    ----------
    sobol_results : dict
        Dictionary where each key is N (sample size) and value contains 'sobol_df'.
    top_n : int
        Number of top features to display.
    """

    for N, result in sobol_results.items():
        plt.figure(figsize=(8, 4))
        df = result['sobol_df']
        top = df.sort_values("S1", ascending=False).head(top_n)
        colors = [FEATURE_COLOR_MAP.get(f, "#cccccc") for f in top["feature"][::-1]]
        plt.barh(top["feature"][::-1], top["S1"][::-1], color=colors)
        plt.title(f"Top {top_n} First-Order Sobol Indices (N={N})")
        plt.xlabel("S1")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def save_FE_results(
    region_name,
    raw_shap,
    shap_df,
    sobol_results,
    save_dir="../results/xgboost",
    tag=None
):

    """
    Save SHAP and Sobol feature importance results to disk for a given region.

    Parameters
    ----------
    region_name : str
        Name of the region to use in the output filenames.
    raw_shap: shap._explanation.Explanation
        SHAP Explanation object to be saved for further inspection or plotting.
    shap_df : pd.DataFrame
        DataFrame containing SHAP values and feature rankings.
    sobol_results : dict or None
        Dictionary containing Sobol analysis results (can be None if not used).
    save_dir : str
        Directory to save the output files to. Will be created if it doesn't exist.
    tag : str or None
        Optional tag to distinguish different versions (appended to filename).
    
    Saves
    -----
    - SHAP CSV: <save_dir>/xgb_<region_name>[_<tag>]_shap.csv
    - Sobol pickle: <save_dir>/xgb_<region_name>[_<tag>]_sobol_results.pkl
    """
    
    os.makedirs(save_dir, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    base = os.path.join(save_dir, f"xgb_{region_name}{suffix}")

    # Save SHAP
    shap_df.to_csv(f"{base}_shap.csv", index=False)
    print(f"[INFO] Saved SHAP ranking to {base}_shap.csv")

    # Save raw SHAP Explanation object
    raw_shap_path = f"{base}_raw_shap.pkl"
    joblib.dump(raw_shap, raw_shap_path)
    print(f"[INFO] Saved raw SHAP Explanation to {raw_shap_path}")

    # Save all Sobol results (DFs + Si + diagnostics) as one .pkl
    if sobol_results:
        sobol_path = f"{base}_sobol_results.pkl"
        joblib.dump(sobol_results, sobol_path)
        print(f"[INFO] Saved all Sobol results to {sobol_path}")


def build_sobol_summary_table(
    region_to_df: dict,
    param_order: list,
    index_type: str = "S1"
):
    """
    Build a summary table of Sobol indices across regions with value ± CI_half and ranking.

    Parameters
    ----------
    region_to_df : dict
        Maps region name → corresponding Si DataFrame (first_Si, total_Si, or second_Si).
    param_order : list
        List of parameter names in the desired row order.
    index_type : str
        One of "S1", "ST", or "S2".

    Returns
    -------
    pd.DataFrame
        Table with one row per parameter, and two columns per region: value ± CI, and rank.
    """
    assert index_type in ["S1", "ST", "S2"]

    rows = []
    regions = list(region_to_df.keys())

    for param in param_order:
        row = {"Parameter": param}
        for region in regions:
            df = region_to_df[region].copy()

            # Ensure there's a 'feature' column
            if "feature" not in df.columns:
                df["feature"] = df.index

            df = df.set_index("feature")

            if param not in df.index:
                row[f"{region}_value"] = "NaN"
                row[f"{region}_rank"] = np.nan
                continue

            value = df.loc[param, index_type]
            conf = df.loc[param, f"{index_type}_conf"]
            formatted = f"{value:.3f} ± {conf:.2f}"
            row[f"{region}_value"] = formatted

        # Rankings
        for region in regions:
            df = region_to_df[region].copy()
            if "feature" not in df.columns:
                df["feature"] = df.index
            df = df.sort_values(by=index_type, ascending=False).reset_index(drop=True)
            rank_map = {name: i + 1 for i, name in enumerate(df["feature"])}
            row[f"{region}_rank"] = rank_map.get(param, np.nan)

        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("Parameter")

    # Add Sum and Avg Conf if applicable
    if index_type in ["S1", "ST"]:
        sum_row = {"Parameter": "Sum"}
        conf_row = {"Parameter": "Avg Conf"}

        for region in regions:
            df = region_to_df[region]
            sum_val = df[index_type].sum()
            avg_conf = df[f"{index_type}_conf"].mean()
            sum_row[f"{region}_value"] = f"{sum_val:.3f}"
            sum_row[f"{region}_rank"] = ""
            conf_row[f"{region}_value"] = f"{avg_conf:.4f}"
            conf_row[f"{region}_rank"] = ""

        summary_df.loc["Sum"] = sum_row
        summary_df.loc["Avg Conf"] = conf_row

    return summary_df


import os
import pandas as pd
import joblib

def load_FE_results(
    region_name,
    save_dir="../results/xgboost",
    tag=None
):
    """
    Load SHAP and Sobol results for a given region, including the raw SHAP Explanation object.

    Parameters
    ----------
    region_name : str
        Name of the region to load results for.
    save_dir : str
        Directory where results were saved.
    tag : str or None
        Optional tag to distinguish versions.

    Returns
    -------
    shap_df : pd.DataFrame
        DataFrame of SHAP values and rankings.
    raw_shap : shap.Explanation or None
        SHAP Explanation object (or None if not found).
    sobol_results : dict or None
        Dictionary of Sobol results (or None if not found).
    """
    suffix = f"_{tag}" if tag else ""
    base = os.path.join(save_dir, f"xgb_{region_name}{suffix}")

    shap_path = f"{base}_shap.csv"
    raw_shap_path = f"{base}_raw_shap.pkl"
    sobol_path = f"{base}_sobol_results.pkl"

    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"[ERROR] SHAP file not found: {shap_path}")
    shap_df = pd.read_csv(shap_path)
    print(f"[INFO] Loaded SHAP ranking from {shap_path}")

    raw_shap = None
    if os.path.exists(raw_shap_path):
        raw_shap = joblib.load(raw_shap_path)
        print(f"[INFO] Loaded raw SHAP Explanation from {raw_shap_path}")
    else:
        print(f"[WARN] Raw SHAP Explanation not found: {raw_shap_path}")

    sobol_results = None
    if os.path.exists(sobol_path):
        sobol_results = joblib.load(sobol_path)
        print(f"[INFO] Loaded Sobol results from {sobol_path}")
    else:
        print(f"[WARN] Sobol results not found: {sobol_path}")

    return shap_df, raw_shap, sobol_results


def plot_top_k_rankings_across_regions(rank_sources, source_type="shap", top_k=13, figsize=(10, 6)):
    """
    Plot parameter ranking changes across regions.

    Parameters
    ----------
    rank_sources : dict
        Maps region name → either SHAP df or Si dict.
        SHAP df must have 'feature' and 'mean_abs_shap'.
        Si dict must have 'S1' or 'ST' and 'names'.
    source_type : str
        Either "shap", "S1", or "ST"
    top_k : int
        How many ranks to show (y-axis = 1 to top_k)
    figsize : tuple
        Size of the figure
    """
    regions = list(rank_sources.keys())
    all_features = set()
    ranks_per_region = {}

    for region, data in rank_sources.items():
        if source_type == "shap":
            df = data.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            features = df["feature"].tolist()
        elif source_type in ["S1", "ST"]:
            scores = data[source_type]
            features = data["feature"]
            df = pd.DataFrame({"feature": features, "score": scores})
            df = df.sort_values("score", ascending=False).reset_index(drop=True)
            features = df["feature"].tolist()
        else:
            raise ValueError("Invalid source_type. Use 'shap', 'S1', or 'ST'.")

        all_features.update(features)
        ranks = {feat: i + 1 for i, feat in enumerate(features)}
        ranks_per_region[region] = ranks

    all_features = sorted(all_features)
    region_labels = list(rank_sources.keys())

    ranking_matrix = pd.DataFrame(index=all_features, columns=region_labels)
    for region in region_labels:
        for feat in all_features:
            ranking_matrix.loc[feat, region] = ranks_per_region[region].get(feat, np.nan)

    # Filter to top_k only
    filtered = ranking_matrix.apply(lambda row: any(r <= top_k for r in row if pd.notna(r)), axis=1)
    ranking_matrix = ranking_matrix[filtered]

    # --- Sort features by ranking in last region (e.g., mass transport) ---
    last_region = region_labels[-1]
    feature_order = ranking_matrix[last_region].sort_values().index.tolist()

    plt.figure(figsize=figsize)
    for feature in feature_order:
        y_vals = ranking_matrix.loc[feature].values.astype(float)
        color = FEATURE_COLOR_MAP.get(feature, "#cccccc")
        plt.plot(region_labels, y_vals, marker='o', label=feature, color=color)

    plt.gca().invert_yaxis()
    plt.xticks(rotation=0, fontsize = 13)
    plt.yticks(range(1, top_k + 1), fontsize=13)
    plt.xlabel("Current density region", fontsize=13)
    plt.ylabel("Ranking (1 = most important)", fontsize=13)
    plt.title(f"Top {top_k} Parameter Rankings across Regions ({source_type})", fontsize=16)

    # Sort legend handles to match line order
    handles, labels = plt.gca().get_legend_handles_labels()
    sorted_handles = [handles[labels.index(feat)] for feat in feature_order if feat in labels]
    plt.legend(sorted_handles, feature_order, bbox_to_anchor=(1.05, 1), loc='upper left',
               title="Parameter", fontsize=13)

    plt.tight_layout()
    plt.grid(True)
    plt.show()


def select_top_features(
    rank_sources,
    source_type="shap",          # "shap", "S1", or "ST"
    method="threshold",          # "threshold" or "topk"
    threshold=0.90,              # used if method="threshold"
    top_k=6                      # used if method="topk"
):
    """
    Select top features per region based on SHAP or Sobol importance.

    Parameters
    ----------
    rank_sources : dict
        Maps region name → SHAP DataFrame or Sobol Si dict.
    source_type : str
        "shap", "S1", or "ST"
    method : str
        "threshold" (cumulative importance) or "topk" (fixed count)
    threshold : float
        Minimum cumulative importance to reach (only for method="threshold")
    top_k : int
        Number of top features to select (only for method="topk")

    Returns
    -------
    selected_features_per_region : dict
        Region → list of selected features
    union_set : set
        Union of all selected features across regions
    """
    import pandas as pd

    if source_type in ["S1", "ST"] and method != "threshold":
        raise ValueError(f"For Sobol source_type '{source_type}', only method='threshold' is supported.")
    
    if source_type in ["shap"] and method != "topk":
        raise ValueError(f"For Shapley FI (source_type = '{source_type}'), only method='topk' is supported.")

    selected_features_per_region = {}

    for region, data in rank_sources.items():
        if source_type == "shap":
            df = data.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
            df["importance"] = df["mean_abs_shap"]
        elif source_type in ["S1", "ST"]:
            df = pd.DataFrame({
                "feature": data["feature"],
                "importance": data[source_type]
            }).sort_values("importance", ascending=False).reset_index(drop=True)
        else:
            raise ValueError("source_type must be 'shap', 'S1', or 'ST'.")

        if method == "threshold":
            df["cumulative"] = df["importance"].cumsum()
            total_importance = df["importance"].sum()
            df["cumulative_norm"] = df["cumulative"] / total_importance

            #return(df)
            # Select features until cumulative_norm >= threshold
            selected = []
            for i, row in df.iterrows():
                selected.append(row["feature"])
                if row["cumulative"] >= threshold:
                    break
            
            total_explained = df[df["feature"].isin(selected)]["importance"].sum()

        elif method == "topk":
            selected = df.head(top_k)["feature"].tolist()
            total_explained = df[df["feature"].isin(selected)]["importance"].sum() / df["importance"].sum()
        else:
            raise ValueError("method must be 'threshold' or 'topk'.")

        selected_features_per_region[region] = selected
        print(f"\nRegion: {region}")
        print(f"Selected {len(selected)} features using method='{method}' "
              f"({f'threshold={threshold}' if method=='threshold' else f'top_k={top_k}'})")
        print(f"Total importance explained: {total_explained:.3f}")
        print("Selected features:", selected)

    # Compute union of all selected features
    union_set = set()
    for features in selected_features_per_region.values():
        union_set.update(features)
 
    print(f"\nUnion of all selected features across regions: {sorted(union_set)}")
    print(f"Total unique features selected: {len(union_set)}")

    return selected_features_per_region, union_set

def build_rank_table(rank_dict):
    """
    Builds a parameter ranking table across regions,
    ordered by the region with the fewest ranked features.

    Parameters
    ----------
    rank_dict : dict
        Dictionary mapping region name to ordered list of ranked features.

    Returns
    -------
    pd.DataFrame
        DataFrame with features as rows and regions as columns.
        Values are rankings (1 = most important), NaN if not ranked.
    """
    # Identify all unique features
    all_features = set(f for lst in rank_dict.values() for f in lst)

    # Use the region with the fewest features to determine row order
    base_region = min(rank_dict, key=lambda k: len(rank_dict[k]))
    base_order = rank_dict[base_region]

    # Append remaining features not in the base region
    remaining = [f for f in all_features if f not in base_order]
    ordered_features = base_order + sorted(remaining)

    # Initialize the DataFrame
    df = pd.DataFrame(index=ordered_features, columns=rank_dict.keys())

    # Fill in rankings
    for region, features in rank_dict.items():
        for rank, feature in enumerate(features, start=1):
            df.loc[feature, region] = rank

    return df.astype("Int64")  # Ensure integer display with NA support




def compare_selected_features(dict_a, dict_b, name_a="SHAP", name_b="Sobol"):
    """
    Compare two feature-selection dictionaries and report:
    - Shared features across all regions (intersection per method)
    - Common features across both methods
    - Unique features added by each method

    Parameters
    ----------
    dict_a : dict
        First selection dictionary (e.g., SHAP).
    dict_b : dict
        Second selection dictionary (e.g., Sobol).
    name_a : str
        Name of first method (for reporting).
    name_b : str
        Name of second method (for reporting).

    Returns
    -------
    common_features : set
        Features shared across all regions and both methods.
    additional_features : dict
        Features unique to each method (not in the shared intersection).
    """
    # Features selected across all regions (intersection within method)
    shared_a = set.intersection(*(set(features) for features in dict_a.values()))
    shared_b = set.intersection(*(set(features) for features in dict_b.values()))

    # Common features (intersection across both methods)
    common_features = shared_a & shared_b

    # Additional features unique to each method
    additional_features = {
        name_a: sorted(shared_a - common_features),
        name_b: sorted(shared_b - common_features)
    }

    return sorted(common_features), additional_features


def compute_region_auc(row, ifc_cols, ucell_cols, region_bounds, handle_negative="drop"):
    """
    Compute the discrete AUC for one region of the polarization curve.

    Parameters
    ----------
    row : pd.Series
        One row from the dataset.
    ifc_cols : list of str
        Columns containing current density values.
    ucell_cols : list of str
        Columns containing voltage values.
    region_bounds : tuple
        Tuple (lower_bound, upper_bound) for the region.
    handle_negative : str
        What to do with negative voltages:
        - 'drop': ignore (mask out) those points
        - 'zero': replace them with 0
        - 'keep': use as-is (default behavior)

    Returns
    -------
    float or np.nan
        Computed area under the curve for the given region.
    """
    import numpy as np

    ifcs = row[ifc_cols].values
    volts = row[ucell_cols].values
    mask_region = (ifcs >= region_bounds[0]) & (ifcs < region_bounds[1])

    if not np.any(mask_region):
        return np.nan

    region_ifcs = ifcs[mask_region]
    region_volts = volts[mask_region]

    if handle_negative == "drop":
        mask_valid = region_volts >= 0
        region_ifcs = region_ifcs[mask_valid]
        region_volts = region_volts[mask_valid]
    elif handle_negative == "zero":
        region_volts = np.maximum(region_volts, 0)

    if len(region_ifcs) < 2:
        return np.nan  # Need at least two points to integrate

    return np.trapezoid(region_volts, region_ifcs)


def add_confidence_intervals(df, index_col="S1", conf_col="S1_conf"):
    """
    Adds confidence interval bounds and a flag if 0 is inside the CI.

    Parameters:
        df (pd.DataFrame): Sobol index DataFrame with index and conf columns.
        index_col (str): Name of the Sobol index column (e.g. 'S1', 'ST', 'S2').
        conf_col (str): Name of the confidence column (e.g. 'S1_conf').

    Returns:
        pd.DataFrame: Updated DataFrame with CI bounds and zero flag.
    """
    df = df.copy()
    df["CI_lower"] = df[index_col] - df[conf_col]
    df["CI_upper"] = df[index_col] + df[conf_col]
    df["CI_contains_0"] = (df["CI_lower"] <= 0) & (df["CI_upper"] >= 0)
    return df


def run_sobol_analysis_for_region(Y, problem, region_name="activation"):
    """
    Run Sobol sensitivity analysis on a provided target vector Y.

    Parameters
    ----------
    Y : np.ndarray
        AUC targets for a region.
    problem : dict
        SALib problem definition.
    region_name : str
        Region label for display only.

    Returns
    -------
    Si, total_Si, first_Si, second_Si
    """
    Y = Y.to_numpy()
    
    # Run sobol
    print(f"[INFO] Running Sobol SA on the AUC of the '{region_name}' region.")
    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
    total_Si, first_Si, second_Si = Si.to_df()

    # Convert to DataFrames
    total_Si, first_Si, second_Si = Si.to_df()

    # Add confidence interval processing
    first_Si = add_confidence_intervals(first_Si, "S1", "S1_conf")
    total_Si = add_confidence_intervals(total_Si, "ST", "ST_conf")
    second_Si = add_confidence_intervals(second_Si, "S2", "S2_conf")

    # --- Post-analysis summaries ---
    print("\n[SUMMARY] Sobol S1 (main effects):")
    s1_sum = round(first_Si["S1"].sum(), 4)
    s1_pos_sum = round(first_Si[first_Si["S1"] > 0]["S1"].sum(), 4)
    print("Sum of S1 indices:", s1_sum)
    print("Sum of S1 indices (setting negative indices to 0):", s1_pos_sum)
    if (first_Si["S1"] < 0).any():
        print("[WARNING] Some S1 indices are negative!")

    print("\n[SUMMARY] Sobol S2 (interaction effects):")
    s2_sum = second_Si["S2"].sum()
    s2_pos_mask = (second_Si["S2"] > 0) & (~second_Si["CI_contains_0"])
    s2_pos_sum = second_Si.loc[s2_pos_mask, "S2"].sum()
    print("Sum of second order:", round(s2_sum, 4))
    print("Sum of second order (only significant & > 0):", round(s2_pos_sum, 4))
    if (second_Si["S2"] < 0).any():
        print("[WARNING] Some S2 indices are negative!")

    print("\n[SUMMARY] Combined S1 + S2:")
    print("Sum of S1 and S2:", round(s1_sum + s2_sum,4))
    print("Sum of significant S1 + significant S2:", round(s1_pos_sum + s2_pos_sum, 4))

    return Si, total_Si, first_Si, second_Si


def plot_sobol_region_barplot(
    df: pd.DataFrame,
    region: str,
    index_type: str = "S1",
    top_k: int = 10,
    figsize=(6, 4),
):
    """
    Plot top Sobol indices for a region as horizontal bars with confidence intervals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with Sobol indices and columns: index_type and index_type+'_conf'.
    region : str
        Region name for title.
    index_type : str
        "S1" or "ST".
    top_k : int
        Number of top features to plot.
    figsize : tuple
        Figure size.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert index_type in ["S1", "ST"], "Only 'S1' or 'ST' supported."

    # If feature column is missing, get it from index
    df = df.copy()
    if "feature" not in df.columns:
        df["feature"] = df.index

    df_sorted = df.sort_values(index_type, ascending=False).reset_index(drop=True)
    df_top = df_sorted.head(top_k).copy()
    df_top = df_top[::-1]  # Reverse for horizontal order

    features = df_top["feature"]
    values = df_top[index_type]
    confs = df_top[f"{index_type}_conf"]
    colors = [FEATURE_COLOR_MAP.get(f, "#cccccc") for f in features]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(features, values, color=colors, edgecolor="black")

    ax.errorbar(
        values,
        np.arange(len(features)),
        xerr=confs,
        fmt="none",
        ecolor="black",
        capsize=4,
        elinewidth=1
    )

    ax.set_xlabel(f"{index_type} Value", fontsize=12)
    ax.set_title(f"{index_type}-based Ranking: {region.capitalize()} region", fontsize=14)
    ax.grid(True, axis='x', linestyle="--", alpha=0.6)
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.show()
