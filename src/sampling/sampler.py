import numpy as np
import sys
import os
import pandas as pd
# Adjust the path to point to external/AlphaPEM
sys.path.append(os.path.abspath("../external/AlphaPEM"))
# Importing constants' value and functions
from configuration.settings import current_density_parameters, physical_parameters, computing_parameters, operating_inputs
from model.AlphaPEM import AlphaPEM

PARAMETER_RANGES = {
    # Operating conditions
    "Tfc": (333, 363),  # Cell temperature (K)
    "Pa_des": (1.1e5, 3e5),  # Desired cell pressure (bara)
    "Pc_des": None,  # Desired cell pressure (bara)
    "Sa": (1.1, 3),  # Stoichiometry anode
    "Phi_a_des": (0.1, 1),  # Desired entrance humidity anode

    # Undetermined physical parameters
    "epsilon_gdl": (0.55, 0.8),  # GDL porosity
    "tau": (1.0, 4.0),  # Pore structure coefficient
    "epsilon_mc": (0.15, 0.4),  # CL volume fraction of ionomer
    "epsilon_c": (0.15, 0.3),  # GDL compression ratio
    "e": [3, 4, 5],  # Capillary exponent (should be an integer)
    "Re": (5e-7, 5e-6),  # Electron conduction resistance (Ω·m²)
    "i0_c_ref": (0.001, 500),  # Cathode exchange current density (A/m²)
    "kappa_co": (0.01, 40),  # Crossover correction coefficient (mol/(m·s·Pa))
    "kappa_c": (0, 100),  # Overpotential correction exponent
    "a_slim": (0, 1),  # Slim coefficient a
    "b_slim": (0, 1),  # Slim coefficient b
    "a_switch": (0, 1),  # Slim switch parameter
}


def get_polarisation_curve_samples(sampled_parameters, fixed_parameters, save_path="../data/raw/results.pkl", save_every=10):
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

            results.append(combined_parameters)

            # Save every `save_every` iterations
            if (i + 1) % save_every == 0 and save_path != None:
                pd.DataFrame(results).to_pickle(save_path)
                print(f"✅ Saved {i + 1} samples to {save_path}")

        except Exception as e:
            print(f"❌ Sample {i} not valid: {sample}")
            print(f"   Error: {e}")

    # Final save
    if save_path != None:
        pd.DataFrame(results).to_pickle(save_path)
        print(f"\n📁 Final save complete: {save_path} with {len(results)} valid samples.")

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
        "Phi_a_des": 0.5
    }

def sample_parameters(n_samples=100, parameter_ranges=PARAMETER_RANGES):
    samples = {}
    
    for key, val in parameter_ranges.items():
        if key == 'Pa_des':
            low, high = val
            samples['Pa_des'] = np.random.uniform(low, high, n_samples)
            low, high = (np.maximum(1.1e5, samples['Pa_des'] - 0.5e5), np.maximum(1.1e5, samples['Pa_des'] - 0.1e5))
            samples['Pc_des'] = np.random.uniform(low, high)

        elif isinstance(val, tuple):  # Continuous range
            low, high = val
            samples[key] = np.random.uniform(low, high, n_samples)

        elif isinstance(val, list):  # Discrete
            samples[key] = np.random.choice(val, n_samples)
        
    return [{key: float(value) for key, value in zip(samples.keys(), values)} for values in zip(*samples.values())]
