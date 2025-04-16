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
    "Sc": (1.1, 10),  # Stoichiometry cathode
    "Phi_a_des": (0.1, 1),  # Desired entrance humidity anode
    "Phi_c_des": (0.1, 1),  # Desired entrance humidity cathode

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


def get_polarisation_curve_samples(sampled_parameters, fixed_parameters):
    results = []

    for sample in sampled_parameters:
        combined_parameters = {**sample, **fixed_parameters}
        Simulator = AlphaPEM(**combined_parameters)

        variables, operating_inputs, parameters = Simulator.variables, Simulator.operating_inputs, Simulator.parameters

        # Extraction of the variables
        t, Ucell_t = np.array(variables['t']), np.array(variables['Ucell'])
        # Extraction of the operating inputs and the parameters
        current_density = operating_inputs['current_density']
        t_step, i_step, i_max_pola = parameters['t_step'], parameters['i_step'], parameters['i_max_pola']
        delta_pola = parameters['delta_pola']
        i_EIS, t_EIS, f_EIS = parameters['i_EIS'], parameters['t_EIS'], parameters['f_EIS']
        type_fuel_cell, type_auxiliary = parameters['type_fuel_cell'], parameters['type_auxiliary']
        type_control, type_plot = parameters['type_control'], parameters['type_plot']

        if type_plot == "fixed":
            # Creation of ifc_t
            n = len(t)
            ifc_t = np.zeros(n)
            for i in range(n):
                ifc_t[i] = current_density(t[i], parameters) / 1e4  # Conversion in A/cm²

            # Recovery of ifc and Ucell from the model after each stack stabilisation
            delta_t_load_pola, delta_t_break_pola, delta_i_pola, delta_t_ini_pola = delta_pola
            nb_loads = int(i_max_pola / delta_i_pola + 1)  # Number of loads which are made
            ifc_discretized = np.zeros(nb_loads)
            Ucell_discretized = np.zeros(nb_loads)
            for i in range(nb_loads):
                t_load = delta_t_ini_pola + (i + 1) * (delta_t_load_pola + delta_t_break_pola) - delta_t_break_pola / 10
                #                                                                                    # time for measurement
                idx = (np.abs(t - t_load)).argmin()  # the corresponding index
                ifc_discretized[i] = ifc_t[idx]  # the last value at the end of each load
                Ucell_discretized[i] = Ucell_t[idx]  # the last value at the end of each load

        combined_parameters['ifc'] = ifc_discretized
        combined_parameters['Ucell'] = Ucell_discretized

        results.append(combined_parameters)

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
        "Lgc": Lgc
    }

def sample_parameters(n_samples=100, parameter_ranges=PARAMETER_RANGES):
    samples = {}
    
    for key, val in parameter_ranges.items():
        if key == 'Pa_des':
            low, high = val
            samples['Pa_des'] = np.random.uniform(low, high, n_samples)
            low, high = (np.maximum(1, samples['Pa_des'] - 0.5), np.maximum(1, samples['Pa_des'] - 0.1))
            samples['Pc_des'] = np.random.uniform(low, high)

        elif isinstance(val, tuple):  # Continuous range
            low, high = val
            samples[key] = np.random.uniform(low, high, n_samples)

        elif isinstance(val, list):  # Discrete
            samples[key] = np.random.choice(val, n_samples)
        
    return [{key: float(value) for key, value in zip(samples.keys(), values)} for values in zip(*samples.values())]
