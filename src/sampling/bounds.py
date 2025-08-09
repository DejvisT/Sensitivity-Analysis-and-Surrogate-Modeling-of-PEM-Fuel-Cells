# src/sampling/bounds.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import yaml

# -----------------------------
# Public data container
# -----------------------------
@dataclass
class SamplingSpec:
    names: List[str]                 # variable parameter names (order matters)
    bounds: np.ndarray               # (d, 2) low/high for cont/int dims, aligned with names
    types: List[str]                 # 'continuous' | 'integer' | 'categorical' aligned with names
    categories: Dict[str, List[Any]] # name -> allowed categories (only for categoricals)
    fixed_values: Dict[str, Any]     # name -> fixed scalar value
    derived_exprs: Dict[str, str]    # name -> expression string (e.g., "Pa_des - 20000")
    seed: int
    design_cfg: Dict[str, Any]       # e.g., {'method':'sobol', 'n_samples': 128}
    spec_index: Dict[str, Dict[str, Any]]  # raw spec by name (for downstream tooling)


# -----------------------------
# Exceptions
# -----------------------------
class SpecError(ValueError):
    """Raised when the YAML param_config has structural mistakes."""


class SampleValidationError(ValueError):
    """Raised when a sample violates the spec."""


# -----------------------------
# YAML loading/validation
# -----------------------------
def load_param_config(path: str) -> SamplingSpec:
    """
    Load and validate configs/param_config.yaml. Returns a SamplingSpec
    you can use for sampling & per-sample validation.

    This validates:
      - presence of required fields (name, type, fixed),
      - correct combos (low/high, values, value, derived),
      - types for bounds and values,
      - no duplicate parameter names.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    seed = int(cfg.get("seed", 0))
    design_cfg = cfg.get("sampling", {})
    params = cfg.get("parameters", [])
    if not isinstance(params, list) or not params:
        raise SpecError("No parameters declared under 'parameters' (expected a non-empty list).")

    names: List[str] = []
    bounds_list: List[Tuple[float, float]] = []
    types: List[str] = []
    categories: Dict[str, List[Any]] = {}
    fixed_values: Dict[str, Any] = {}
    derived_exprs: Dict[str, str] = {}
    spec_index: Dict[str, Dict[str, Any]] = {}

    seen = set()
    for p in params:
        _validate_param_block(p)
        name = p["name"]
        if name in seen:
            raise SpecError(f"Duplicate parameter name: {name}")
        seen.add(name)
        spec_index[name] = p

        if p["fixed"]:
            # fixed param: can be value OR derived (if derived, no direct value)
            if "derived" in p:
                derived_exprs[name] = str(p["derived"]).strip()
            elif "value" in p:
                fixed_values[name] = p["value"]
            else:
                # _validate_param_block already guarantees one is present—this is safety
                raise SpecError(f"{name}: fixed param must have 'value' or 'derived'.")
            # fixed params are not part of variable names/bounds
            continue

        # variable params
        ptype = p["type"]
        names.append(name)
        types.append(ptype)

        if ptype in ("continuous", "integer"):
            low, high = float(p["low"]), float(p["high"])
            if low > high:
                raise SpecError(f"{name}: low must be <= high (got {low} > {high}).")
            bounds_list.append((low, high))
        elif ptype == "categorical":
            vals = p["values"]
            if not isinstance(vals, list) or len(vals) == 0:
                raise SpecError(f"{name}: 'values' must be a non-empty list.")
            categories[name] = vals
            # For sampling engines, we often map to [0,1) then bin; keep [0,1] placeholder.
            bounds_list.append((0.0, 1.0))
        else:
            raise SpecError(f"{name}: unknown type '{ptype}'.")

    bounds = np.asarray(bounds_list, dtype=float) if bounds_list else np.zeros((0, 2), dtype=float)
    if len(names) != len(bounds):
        raise SpecError("Internal error: names and bounds misaligned.")

    return SamplingSpec(
        names=names,
        bounds=bounds,
        types=types,
        categories=categories,
        fixed_values=fixed_values,
        derived_exprs=derived_exprs,
        seed=seed,
        design_cfg=design_cfg,
        spec_index=spec_index,
    )


def _validate_param_block(p: Dict[str, Any]) -> None:
    required = {"name", "type", "fixed"}
    missing = required - set(p)
    if missing:
        raise SpecError(f"Missing keys {missing} in parameter: {p}")

    t = p["type"]
    fixed = bool(p["fixed"])

    if fixed:
        # fixed param must have either a scalar 'value' or a 'derived' expression
        if ("value" not in p) and ("derived" not in p):
            raise SpecError(f"{p['name']}: fixed parameter must have 'value' or 'derived'.")
        if ("value" in p) and ("derived" in p):
            raise SpecError(f"{p['name']}: fixed parameter cannot have both 'value' and 'derived'.")
        # If it has 'value', don't require low/high or values
        return

    # Non-fixed (variable) parameters:
    if t in ("continuous", "integer"):
        if not {"low", "high"} <= set(p):
            raise SpecError(f"{p['name']}: non-fixed {t} must have 'low' and 'high'.")
        _assert_number(p["low"], f"{p['name']}.low")
        _assert_number(p["high"], f"{p['name']}.high")
    elif t == "categorical":
        if "values" not in p:
            raise SpecError(f"{p['name']}: non-fixed categorical must have 'values'.")
        if not isinstance(p["values"], list) or len(p["values"]) == 0:
            raise SpecError(f"{p['name']}: 'values' must be a non-empty list.")
    else:
        raise SpecError(f"{p['name']}: unknown type '{t}'.")


def _assert_number(x: Any, where: str) -> None:
    try:
        float(x)
    except Exception:
        raise SpecError(f"{where} must be numeric; got {x!r}.")


# -----------------------------
# Sample-level utilities
# -----------------------------
def apply_derived(sample: Dict[str, Any], spec: SamplingSpec) -> Dict[str, Any]:
    """
    Compute derived params declared in the spec (e.g., Pc_des = Pa_des - 20000).
    Returns a NEW dict (doesn't mutate input).

    Currently supports:
      - Pc_des: "Pa_des - 20000"
    """
    out = dict(sample)
    for name, expr in spec.derived_exprs.items():
        expr = expr.strip()
        if name == "Pc_des" and expr == "Pa_des - 20000":
            if "Pa_des" not in out:
                raise SampleValidationError("Derived 'Pc_des' requires 'Pa_des' in the sample.")
            out[name] = float(out["Pa_des"]) - 20000.0
        else:
            raise SampleValidationError(f"Unsupported derived expression for '{name}': {expr}")
    return out


def validate_sample_against_spec(sample: Dict[str, Any], spec: SamplingSpec) -> Dict[str, Any]:
    """
    Validate a single sample against the provided SamplingSpec (bounds, fixed values, categories).
    Also computes derived params first. Returns the augmented sample (with derived + missing fixed).
    """
    # 1) compute derived fields
    augmented = apply_derived(sample, spec)

    # 2) check all present fields
    for name, val in augmented.items():
        if name not in spec.spec_index:
            # allow extra fields to pass through; AlphaPEM might ignore them
            continue
        p = spec.spec_index[name]
        if p.get("fixed", False):
            if "value" in p and val != p["value"]:
                raise SampleValidationError(f"{name} must be fixed to {p['value']}, got {val}.")
            # if derived: value is set by apply_derived; nothing to check here
            continue

        ptype = p["type"]
        if ptype == "continuous":
            low, high = float(p["low"]), float(p["high"])
            v = float(val)
            if not (low <= v <= high):
                raise SampleValidationError(f"{name} out of bounds [{low}, {high}]: {val}")
        elif ptype == "integer":
            low, high = int(p["low"]), int(p["high"])
            if not (isinstance(val, (int, np.integer)) and low <= int(val) <= high):
                raise SampleValidationError(f"{name} must be integer in [{low}, {high}], got {val}")
        elif ptype == "categorical":
            allowed = list(p["values"])
            if val not in allowed:
                raise SampleValidationError(f"{name} must be one of {allowed}, got {val}")
        else:
            raise SampleValidationError(f"{name}: unknown type {ptype}")

    # 3) fill missing fixed-with-value params
    for name, p in spec.spec_index.items():
        if p.get("fixed", False) and "value" in p and name not in augmented:
            augmented[name] = p["value"]

    return augmented
