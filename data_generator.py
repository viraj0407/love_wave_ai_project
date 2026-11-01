"""
Data Generator for Love Wave Acoustic Propagation in Smart Composites
Generates theoretical dispersion dataset with micro-inertia, flexoelectricity, and imperfect joint defects.
"""

import numpy as np
import pandas as pd
from physics_solver import SmartCompositeLoveWave

def generate_dataset(n_samples=500, random_seed=42):
    np.random.seed(random_seed)
    
    # Physical parameter bounds
    kh1_vals = np.random.uniform(0.5, 4.0, n_samples)
    h1_vals = np.random.uniform(0.2e-6, 3.0e-6, n_samples)          # 0.2 to 3.0 um
    l0_vals = np.random.uniform(0.0, 35.0e-9, n_samples)             # 0 to 35 nm
    mu_flexo_vals = np.random.uniform(0.0, 2.5e-8, n_samples)       # 0 to 25 nC/m
    e15_vals = np.random.uniform(1.5, 5.0, n_samples)               # 1.5 to 5.0 C/m^2
    R_M_vals = np.random.uniform(0.0, 0.15, n_samples)              # joint mechanical slip
    R_E_vals = np.random.uniform(0.0, 0.08, n_samples)              # joint electrical leak
    
    records = []
    
    print(f"Generating {n_samples} physics-grounded smart composite wave samples...")
    for i in range(n_samples):
        solver = SmartCompositeLoveWave(
            h1=h1_vals[i],
            l0=l0_vals[i],
            mu_flexo=mu_flexo_vals[i],
            e15_2=e15_vals[i],
            R_M=R_M_vals[i],
            R_E=R_E_vals[i]
        )
        kh = kh1_vals[i]
        c = solver.solve_phase_velocity(kh)
        
        if not np.isnan(c) and c > 0:
            # Frequency f = omega / (2*pi) = (c * k) / (2*pi) = c * kh / (2*pi*h1)
            f_MHz = (c * kh / (2.0 * np.pi * h1_vals[i])) / 1e6
            
            # Wavelength lambda = 2*pi / k = 2*pi*h1 / kh
            wavelength_um = (2.0 * np.pi * h1_vals[i] / kh) * 1e6
            
            # Group velocity estimate (c - lambda * dc/dlambda)
            # perturbed wavenumber for numerical derivative
            dkh = 0.01
            c_pert = solver.solve_phase_velocity(kh + dkh)
            dc_dkh = (c_pert - c) / dkh if not np.isnan(c_pert) else 0.0
            cg = c + kh * dc_dkh
            
            records.append({
                'kh1': kh,
                'thickness_um': h1_vals[i] * 1e6,
                'micro_inertia_nm': l0_vals[i] * 1e9,
                'flexo_coeff_nC_m': mu_flexo_vals[i] * 1e9,
                'piezo_coupling': e15_vals[i],
                'mechanical_slip_RM': R_M_vals[i],
                'electrical_leak_RE': R_E_vals[i],
                'frequency_MHz': f_MHz,
                'wavelength_um': wavelength_um,
                'phase_velocity_mps': c,
                'group_velocity_mps': cg
            })
            
    df = pd.DataFrame(records)
    print(f"Successfully generated {len(df)} valid dispersion data points.")
    return df

if __name__ == "__main__":
    df = generate_dataset(n_samples=250)
    out_path = "artifacts/file_generation/ttl=63d/intermediate/love_wave_ai_project/love_wave_dataset.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")
    print(df.head())
