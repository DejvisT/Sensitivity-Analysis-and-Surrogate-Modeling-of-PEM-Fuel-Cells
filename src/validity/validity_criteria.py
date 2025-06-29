import pandas as pd

def validate_polarization_curves(
    df: pd.DataFrame,
    apply_criteria: dict,
    filter_invalid: bool = False,
    keep_temp_cols: bool = False,
    approx_monotonic_threshold: float = 0.01,
    voltage_range: tuple = (0.0, 1.23),
    early_values_tolerance: int = 3
) -> pd.DataFrame:
    """
    Validate polarization curves in a DataFrame based on customizable criteria.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing polarization curve data (voltage and current), as well as experimental conditions.

    apply_criteria : dict
        Dictionary with keys as criteria names and boolean values indicating whether to apply them.
        Valid keys:
            - "start_in_range": First voltage value must be within `voltage_range`.
            - "early_values_in_range": First N voltage values (controlled by `early_values_tolerance`) must be in range.
            - "monotonic": Voltage must be strictly non-increasing.
            - "approx_monotonic": Voltage must be approximately non-increasing with allowed bumps.

    filter_invalid : bool, optional (default=False)
        If True, filters out rows that fail selected criteria.

    keep_temp_cols : bool, optional (default=False)
        If False, removes intermediate boolean columns used for classification.

    approx_monotonic_threshold : float, optional (default=0.01)
        Threshold used when applying "approx_monotonic" criterion.

    voltage_range : tuple, optional (default=(0.0, 1.23))
        Tuple specifying the valid voltage range as (min_voltage, max_voltage) for the range-based checks.

    early_values_tolerance : int, optional (default=3)
        Number of early voltage values to check in `early_values_in_range`.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with validation results, filtered and/or with temp columns depending on function parameters.

        Example
    -------
    >>> criteria = {
    >>>     "start_in_range": True,
    >>>     "early_values_in_range": True,
    >>>     "approx_monotonic": True
    >>> }
    >>> validated_df = validate_polarization_curves(
    >>>     df_clean,
    >>>     apply_criteria=criteria,
    >>>     filter_invalid=True,
    >>>     voltage_range=(0.05, 1.2),
    >>>     approx_monotonic_threshold=0.015,
    >>>     early_values_tolerance=4
    >>> )
    >>> print(validated_df.head())
    """
    
    df_config = df.copy()
    ucell_columns = [col for col in df.columns if col.startswith("Ucell_")]
    v_min, v_max = voltage_range

    if apply_criteria.get("start_in_range", False):
        df_config["start_in_range"] = df_config[ucell_columns[0]].between(v_min, v_max)

    if apply_criteria.get("early_values_in_range", False):
        df_config["early_values_in_range"] = df_config[ucell_columns[:early_values_tolerance]].apply(
            lambda row: row.between(v_min, v_max).all(), axis=1
        )

    if apply_criteria.get("monotonic", False):
        df_config["monotonic"] = df_config[ucell_columns].apply(
            lambda row: all(x >= y for x, y in zip(row, row[1:])), axis=1
        )

    if apply_criteria.get("approx_monotonic", False):
        def approx_monotonic_until_negative(row, threshold=approx_monotonic_threshold):
            voltages = row.values.astype(float)
            for i in range(len(voltages) - 1):
                if voltages[i] < voltages[i + 1] - threshold:
                    return False
            return True

        df_config["approx_monotonic"] = df_config[ucell_columns].apply(
            approx_monotonic_until_negative, axis=1
        )

    # Combine active criteria into one final classification column
    criteria_cols = [col for col in [
        "start_in_range", "early_values_in_range", "monotonic", "approx_monotonic"
    ] if apply_criteria.get(col, False)]

    df_config["classification"] = df_config[criteria_cols].all(axis=1)
    df_config["classification"] = df_config["classification"].map({True: "valid", False: "invalid"})

    if filter_invalid:
        df_config = df_config[df_config["classification"] == "valid"]
        df_config = df_config.drop(columns=["classification"])

    if not keep_temp_cols:
        df_config = df_config.drop(columns=criteria_cols, errors="ignore")

    return df_config
