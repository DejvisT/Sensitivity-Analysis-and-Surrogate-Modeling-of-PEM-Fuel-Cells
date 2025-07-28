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


def plot_sobol_index_convergence(sobol_results, index_type="ST", top_k=8, title=None, log_x=True, region=None):
    """
    Plot how each feature's Sobol index (S1 or ST) changes as sample size N increases.

    Parameters
    ----------
    sobol_results : dict
        Output from run_sobol_analysis(). Maps N → dict with key 'sobol_df' containing a DataFrame.
    index_type : str
        Which index to plot: "S1" for first-order or "ST" for total-order.
    top_k : int
        Number of top features to show based on the highest N.
    title : str or None
        Custom plot title. If None, defaults to f"{index_type} Index Convergence".
    log_x : bool
        Use log scale on the x-axis (sample size N).
    region : str or None
        Region name for the title, if available.
    """
    
    assert index_type in ["S1", "ST"]

    max_N = max(sobol_results)
    df_max = sobol_results[max_N]['sobol_df']
    top_features = df_max.sort_values(index_type, ascending=False)["feature"].head(top_k).tolist()

    N_vals = sorted(sobol_results.keys())
    data = {f: [] for f in top_features}

    for N in N_vals:
        df = sobol_results[N]['sobol_df'].set_index("feature")
        for f in top_features:
            v = df.loc[f, index_type] if f in df.index else np.nan
            data[f].append(v)

    plt.figure(figsize=(10, 6))
    for f in top_features:
        plt.plot(N_vals, data[f], marker="o", label=f, color=FEATURE_COLOR_MAP.get(f, "#cccccc"))

    plt.ylabel(f"{index_type} Sobol Index")
    plt.xlabel("Sample Size N")
    plt.xticks([256, 512, 1024, 2048, 4096, 8192])
    plt.ylim(0, 1.05)
    if title is None:
        title = f"{index_type} Index Convergence for {region} Region" if region else f"{index_type} Index Convergence"
    plt.title(title)
    if log_x:
        plt.xscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(loc="upper right", bbox_to_anchor=(1.18, 1.0))
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



def load_FE_results(
    region_name,
    save_dir="../results/xgboost",
    tag=None
):
    """
    Load SHAP and Sobol results for a given region.

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
        DataFrame of SHAP values.
    sobol_results : dict or None
        Dictionary of Sobol results (or None if not found).
    """
    suffix = f"_{tag}" if tag else ""
    base = os.path.join(save_dir, f"xgb_{region_name}{suffix}")

    shap_path = f"{base}_shap.csv"
    sobol_path = f"{base}_sobol_results.pkl"

    if not os.path.exists(shap_path):
        raise FileNotFoundError(f"[ERROR] SHAP file not found: {shap_path}")
    shap_df = pd.read_csv(shap_path)
    print(f"[INFO] Loaded SHAP ranking from {shap_path}")

    sobol_results = None
    if os.path.exists(sobol_path):
        sobol_results = joblib.load(sobol_path)
        print(f"[INFO] Loaded Sobol results from {sobol_path}")
    else:
        print(f"[WARN] Sobol results not found: {sobol_path}")

    return shap_df, sobol_results


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

    filtered = ranking_matrix.apply(lambda row: any(r <= top_k for r in row if pd.notna(r)), axis=1)
    ranking_matrix = ranking_matrix[filtered]

    plt.figure(figsize=figsize)
    for feature in ranking_matrix.index:
        y_vals = ranking_matrix.loc[feature].values.astype(float)
        color = FEATURE_COLOR_MAP.get(feature, "#cccccc")
        plt.plot(region_labels, y_vals, marker='o', label=feature, color=color)

    plt.gca().invert_yaxis()
    plt.xticks(rotation=45)
    plt.yticks(range(1, top_k + 1))
    plt.xlabel("Current density region")
    plt.ylabel("Ranking (1 = most important)")
    plt.title(f"Top {top_k} Parameter Rankings across Regions ({source_type})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Parameter")
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
            df["cumulative"] = df["importance"].cumsum() / df["importance"].sum()
            selected = df[df["cumulative"] <= threshold]["feature"].tolist()
            if len(selected) < len(df):
                selected.append(df.loc[len(selected), "feature"])
            total_explained = df[df["feature"].isin(selected)]["importance"].sum() / df["importance"].sum()
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