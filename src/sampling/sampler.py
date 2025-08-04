
import numpy as np
import sys
import os
import pandas as pd
import sys
import os

# Get absolute path to the AlphaPEM directory
alpha_pem_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "external", "AlphaPEM"))

# Add it to sys.path if not already there
if alpha_pem_path not in sys.path:
    sys.path.append(alpha_pem_path)

from configuration.settings import current_density_parameters, physical_parameters, computing_parameters, operating_inputs
from model.AlphaPEM import AlphaPEM

def get_polarisation_curve_samples(sampled_parameters, fixed_parameters="default", save_path="../data/raw/results.pkl", save_every=10):
    """
    Simulate polarisation curves for a list of sampled parameter configurations using the AlphaPEM model.

    Parameters
    ----------
    sampled_parameters : list of dict or pd.DataFrame
        List of dictionaries or DataFrame containing sampled parameters for the simulation.
    fixed_parameters : dict or str, optional
        Fixed parameters to use in each simulation. If set to "default", default fixed parameters are used via build_fixed_parameters().
    save_path : str or None, optional
        Path to save intermediate and final results as a pickle file. If None, results are not saved.
    save_every : int, optional
        Frequency (in number of samples) at which intermediate results are saved.

    Returns
    -------
    pd.DataFrame
        DataFrame containing input parameters and extracted polarisation curve data (ifc, Ucell) for each valid sample.
    """
    if isinstance(sampled_parameters, pd.DataFrame):
        sampled_parameters = sampled_parameters.to_dict(orient='records')

    # Load default fixed parameters if specified
    if fixed_parameters == "default":
        fixed_parameters = build_fixed_parameters()

    results = []

    for i, sample in enumerate(sampled_parameters):
        try:
            # Handle SHA256 tracking (if present)
            sha256 = sample.get('SHA256', None)
            sample.pop('SHA256', None)  # Remove SHA256 to avoid passing it to the simulator

            # Combine fixed and sampled parameters
            combined_parameters = {**sample, **fixed_parameters}

            # Instantiate the simulation
            Simulator = AlphaPEM(**combined_parameters)
            variables, operating_inputs, parameters = Simulator.variables, Simulator.operating_inputs, Simulator.parameters

            # Extract time and cell voltage over time
            t = np.array(variables['t'])
            Ucell_t = np.array(variables['Ucell'])

            # Unpack relevant parameters and functions
            current_density = operating_inputs['current_density']
            t_step, i_step, i_max_pola = parameters['t_step'], parameters['i_step'], parameters['i_max_pola']
            delta_pola = parameters['delta_pola']
            type_plot = parameters['type_plot']

            # Only extract polarisation curve if type_plot is 'fixed'
            if type_plot == "fixed":
                n = len(t)
                ifc_t = np.zeros(n)

                # Evaluate current density over time
                for j in range(n):
                    ifc_t[j] = current_density(t[j], parameters) / 1e4  # Convert A/m² to A/cm²

                # Compute polarisation curve at discrete current densities
                delta_t_load_pola, delta_t_break_pola, delta_i_pola, delta_t_ini_pola = delta_pola
                nb_loads = int(i_max_pola / delta_i_pola + 1)

                ifc_discretized = np.zeros(nb_loads)
                Ucell_discretized = np.zeros(nb_loads)

                for k in range(nb_loads):
                    t_load = delta_t_ini_pola + (k + 1) * (delta_t_load_pola + delta_t_break_pola) - delta_t_break_pola / 10
                    idx = (np.abs(t - t_load)).argmin()
                    ifc_discretized[k] = ifc_t[idx]
                    Ucell_discretized[k] = Ucell_t[idx]

                # Add simulation outputs to the parameters
                combined_parameters['ifc'] = ifc_discretized
                combined_parameters['Ucell'] = Ucell_discretized

        except Exception as e:
            print(f"❌ Sample {i} not valid: {sample}")
            print(f"   Error: {e}")

            # Save the failed configuration with null outputs
            combined_parameters = {**sample, **fixed_parameters}
            combined_parameters['ifc'] = None
            combined_parameters['Ucell'] = None

        # Append result and track SHA256 if present
        combined_parameters['SHA256'] = sha256 if sha256 else None
        results.append(combined_parameters)

        # Periodically save results
        if (i + 1) % save_every == 0 and save_path is not None:
            pd.DataFrame(results).to_pickle(save_path)
            print(f"✅ Saved {i + 1} samples to {save_path}")

    # Final save
    if save_path is not None:
        pd.DataFrame(results).to_pickle(save_path)
        print(f"\n📁 Final save complete: {save_path} with {len(results)} samples.")

    return pd.DataFrame(results)

def build_fixed_parameters():
    """
    Builds a dictionary of fixed parameters required for simulating the AlphaPEM model
    under polarization curve conditions.

    Returns
    -------
    dict
        Dictionary containing time step parameters, physical properties, operating conditions,
        control flags, and other configuration settings used across all simulation runs.
    """
    # Define the current type and fuel cell model type
    type_current = "polarization"
    type_fuel_cell = "EH-31_2.0"

    # Retrieve current density-related parameters
    t_step, i_step, delta_pola, i_EIS, ratio_EIS, f_EIS, t_EIS, current_density = current_density_parameters(type_current)

    # Get the maximum polarisation current from the operating conditions
    *_, i_max_pola = operating_inputs(type_fuel_cell)

    # Retrieve physical parameters of the membrane electrode assembly and gas channels
    Hcl, epsilon_mc, tau, Hmem, Hgdl, epsilon_gdl, epsilon_c, Hgc, Wgc, Lgc, Aact, e, Re, \
    i0_c_ref, kappa_co, kappa_c, a_slim, b_slim, a_switch, C_scl = physical_parameters(type_fuel_cell)

    # Retrieve simulation-specific computational parameters
    max_step, n_gdl, t_purge = computing_parameters(type_current, Hgdl, Hcl)

    # Assemble and return the dictionary of fixed parameters
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

