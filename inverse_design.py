"""
Inverse Material Design for Smart Acoustic Composites
Calculates optimal layer thickness and material parameters to achieve target wave velocity and frequency.
"""

import numpy as np
from scipy.optimize import minimize
from physics_solver import SmartCompositeLoveWave

class InverseMaterialDesigner:
    def __init__(self, substrate_material='LiNbO3'):
        # Substrate reference properties
        self.substrate = substrate_material

    def design_for_target_velocity(self, target_c_mps, target_freq_MHz, 
                                   allowed_RM=0.02, allowed_RE=0.01):
        """
        Given target phase velocity c* and target frequency f*,
        solves for the optimal layer thickness h1 and flexoelectric coefficient mu_flexo.
        """
        # Relation: f = c * kh1 / (2 * pi * h1) => kh1 = 2 * pi * f * h1 / c
        target_f_Hz = target_freq_MHz * 1e6

        def objective(params):
            h1_val = params[0]        # layer thickness in meters
            mu_val = params[1]        # flexoelectric coefficient in C/m
            
            # calculate expected kh1 from frequency
            kh1 = (2.0 * np.pi * target_f_Hz * h1_val) / target_c_mps
            if kh1 < 0.2 or kh1 > 6.0:
                return 1e6 + (kh1 - 2.0)**2

            solver = SmartCompositeLoveWave(
                h1=h1_val,
                mu_flexo=mu_val,
                R_M=allowed_RM,
                R_E=allowed_RE
            )
            c_calc = solver.solve_phase_velocity(kh1)
            if np.isnan(c_calc):
                return 1e6
            
            # Penalize discrepancy from target velocity + smoothness
            loss = (c_calc - target_c_mps)**2
            return loss

        # Initial guess: h1 = 1.0 um, mu = 1.2e-8 C/m
        x0 = [1.0e-6, 1.2e-8]
        bounds = [(0.1e-6, 5.0e-6), (0.0, 5.0e-8)]

        res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 100})
        
        opt_h1 = res.x[0]
        opt_mu = res.x[1]
        opt_kh1 = (2.0 * np.pi * target_f_Hz * opt_h1) / target_c_mps

        # Verify with final forward solver
        verifier = SmartCompositeLoveWave(h1=opt_h1, mu_flexo=opt_mu, R_M=allowed_RM, R_E=allowed_RE)
        achieved_c = verifier.solve_phase_velocity(opt_kh1)
        achieved_f_MHz = (achieved_c * opt_kh1 / (2.0 * np.pi * opt_h1)) / 1e6

        return {
            'target_velocity_mps': target_c_mps,
            'achieved_velocity_mps': achieved_c,
            'target_frequency_MHz': target_freq_MHz,
            'achieved_frequency_MHz': achieved_f_MHz,
            'optimal_thickness_um': opt_h1 * 1e6,
            'optimal_flexo_nC_m': opt_mu * 1e9,
            'wavenumber_kh1': opt_kh1,
            'velocity_error_pct': abs(achieved_c - target_c_mps) / target_c_mps * 100.0,
            'frequency_error_pct': abs(achieved_f_MHz - target_freq_MHz) / target_freq_MHz * 100.0
        }

if __name__ == "__main__":
    designer = InverseMaterialDesigner()
    print("Testing Reverse Material Design...")
    # User specifies desired phase velocity 3500 m/s at 800 MHz (typical 5G RF filter band)
    result = designer.design_for_target_velocity(target_c_mps=3500.0, target_freq_MHz=800.0)
    print("Optimization Results:")
    for k, v in result.items():
        print(f"  {k:<25}: {v:.3f}" if isinstance(v, float) else f"  {k:<25}: {v}")
