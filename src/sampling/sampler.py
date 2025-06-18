import numpy as np
import sys
import os
import pandas as pd
# Adjust the path to point to external/AlphaPEM
sys.path.append(os.path.abspath("../external/AlphaPEM"))
# Importing constants' value and functions
from configuration.settings import current_density_parameters, physical_parameters, computing_parameters, operating_inputs
from model.AlphaPEM import AlphaPEM

def get_polarisation_curve_samples(sampled_parameters, fixed_parameters="default", save_path="../data/raw/results.pkl", save_every=10):

    if fixed_parameters == "default":
        fixed_parameters = build_fixed_parameters()
        
    results = []

    for i, sample in enumerate(sampled_parameters):
        try:
            combined_parameters = {**sample, **fixed_parameters}
            Simulator = AlphaPEM(**combined_parameters)

            variables, operating_inputs, parameters = Simulator.variables, Simulator.operating_inputs, Simulator.parameters

            # Extraction of the variables
            t, Ucell_t = np.array(variables['t']), np.array(variables['Ucell'])
            current_density = operating_inputs['current_density']
            t_step, i_step, i_max_pola = parameters['t_step'], parameters['i_step'], parameters['i_max_pola']
            delta_pola = parameters['delta_pola']
            i_EIS, t_EIS, f_EIS = parameters['i_EIS'], parameters['t_EIS'], parameters['f_EIS']
            type_fuel_cell, type_auxiliary = parameters['type_fuel_cell'], parameters['type_auxiliary']
            type_control, type_plot = parameters['type_control'], parameters['type_plot']

            if type_plot == "fixed":
                n = len(t)
                ifc_t = np.zeros(n)
                for j in range(n):
                    ifc_t[j] = current_density(t[j], parameters) / 1e4  # Convert A/m² to A/cm²

                delta_t_load_pola, delta_t_break_pola, delta_i_pola, delta_t_ini_pola = delta_pola
                nb_loads = int(i_max_pola / delta_i_pola + 1)
                ifc_discretized = np.zeros(nb_loads)
                Ucell_discretized = np.zeros(nb_loads)

                for k in range(nb_loads):
                    t_load = delta_t_ini_pola + (k + 1) * (delta_t_load_pola + delta_t_break_pola) - delta_t_break_pola / 10
                    idx = (np.abs(t - t_load)).argmin()
                    ifc_discretized[k] = ifc_t[idx]
                    Ucell_discretized[k] = Ucell_t[idx]

                combined_parameters['ifc'] = ifc_discretized
                combined_parameters['Ucell'] = Ucell_discretized

        except Exception as e:
            print(f"❌ Sample {i} not valid: {sample}")
            print(f"   Error: {e}")
            combined_parameters = {**sample, **fixed_parameters}
            combined_parameters['ifc'] = None
            combined_parameters['Ucell'] = None

        results.append(combined_parameters)

        # Save every `save_every` iterations
        if (i + 1) % save_every == 0 and save_path is not None:
            pd.DataFrame(results).to_pickle(save_path)
            print(f"✅ Saved {i + 1} samples to {save_path}")

    # Final save
    if save_path is not None:
        pd.DataFrame(results).to_pickle(save_path)
        print(f"\n📁 Final save complete: {save_path} with {len(results)} samples.")

    return pd.DataFrame(results)

def build_fixed_parameters():
    type_current="polarization"
    type_fuel_cell="EH-31_2.0"
    t_step, i_step, delta_pola, i_EIS, ratio_EIS, f_EIS, t_EIS, current_density = current_density_parameters(type_current)
    # Operating conditions
    *_, i_max_pola = operating_inputs(type_fuel_cell)
    
    # Physical parameters
    Hcl, epsilon_mc, tau, Hmem, Hgdl, epsilon_gdl, epsilon_c, Hgc, Wgc, Lgc, Aact, e, Re, i0_c_ref, kappa_co, \
        kappa_c, a_slim, b_slim, a_switch, C_scl = physical_parameters(type_fuel_cell)
    # Computing parameters
    max_step, n_gdl, t_purge = computing_parameters(type_current, Hgdl, Hcl)

    return {
        "t_step": t_step,
        "i_step": i_step,
        "delta_pola": delta_pola,
        "i_EIS": i_EIS,
        "ratio_EIS": ratio_EIS,
        "f_EIS": f_EIS,
        "t_EIS": t_EIS,
        "current_density": current_density,
        "max_step": max_step,
        "n_gdl": n_gdl,
        "t_purge": t_purge,
        "type_fuel_cell": "manual_setup", 
        "type_current": "polarization",
        "type_auxiliary": "no_auxiliary",
        "type_control": "no_control",
        "type_purge": "no_purge",
        "type_display": "no_display",
        "type_plot": "fixed",
        "C_scl": C_scl,
        "i_max_pola": i_max_pola,
        "Aact": Aact,
        "Hgdl": Hgdl,
        "Hmem": Hmem,
        "Hcl": Hcl,
        "Hgc": Hgc,
        "Wgc": Wgc,
        "Lgc": Lgc,
        "Sa": 1.3,
        "Phi_a_des": 0.5,
        "a_slim": 0,
        "b_slim": 1,
        "a_switch": 0.99,
    }

def make_exclusive_bounds(bounds_dict, relative_eps=1e-6):
    shrunk_bounds = {}
    for key, value in bounds_dict.items():
        if isinstance(value, list) and len(value) == 2:
            low, high = value
            if low >= high:
                raise ValueError(f"Invalid bounds for '{key}': lower bound must be < upper bound.")
            eps = (high - low) * relative_eps
            shrunk_bounds[key] = [low + eps, high - eps]
        else:
            # Preserve other values (e.g., lists not of length 2, None, etc.)
            shrunk_bounds[key] = value
    return shrunk_bounds

