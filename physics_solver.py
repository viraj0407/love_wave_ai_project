"""
Physics Solver: Love-Type Waves in Smart Composites
Flexoelectric Top Layer + Piezoelectric Substrate with Imperfect Interface and Micro-Inertia
"""

import numpy as np
from scipy.optimize import brentq

class SmartCompositeLoveWave:
    def __init__(self, 
                 rho1=2200.0,       # kg/m^3 (flexoelectric top layer, e.g. BST/dielectric composite)
                 c44_1=1.8e10,      # N/m^2 (elastic shear modulus layer 1)
                 h1=1.0e-6,         # m (layer thickness, 1 micron)
                 l0=15.0e-9,        # m (micro-inertia length scale, 15 nm)
                 mu_flexo=1.2e-8,   # C/m (flexoelectric coupling coefficient)
                 eps11_1=1.2e-10,   # F/m (permittivity layer 1)
                 rho2=4700.0,       # kg/m^3 (piezoelectric substrate, e.g. LiNbO3)
                 c44_2=6.0e10,      # N/m^2 (elastic shear modulus substrate)
                 e15_2=3.7,         # C/m^2 (piezoelectric stress constant)
                 eps11_2=3.9e-10,   # F/m (permittivity substrate)
                 R_M=0.05,          # dimensionless mechanical interface slip
                 R_E=0.02):         # dimensionless electrical interface leakage
        
        self.rho1 = float(rho1)
        self.c44_1 = float(c44_1)
        self.h1 = float(h1)
        self.l0 = float(l0)
        self.mu_flexo = float(mu_flexo)
        self.eps11_1 = float(eps11_1)
        
        self.rho2 = float(rho2)
        self.c44_2 = float(c44_2)
        self.e15_2 = float(e15_2)
        self.eps11_2 = float(eps11_2)
        
        self.R_M = float(R_M)
        self.R_E = float(R_E)

    def get_effective_properties(self, kh1):
        """Calculate effective velocities and modulations for a given wavenumber kh1."""
        k = kh1 / self.h1
        M1 = 1.0 + (self.l0 * k)**2
        
        # Flexoelectric nanoscale gradient stiffening
        flexo_stiffening = 1.0 + (self.mu_flexo**2 / (self.eps11_1 * self.c44_1)) * (1.0 / (self.h1**2 * (1.0 + (k * self.l0)**2)))
        c44_1_eff = self.c44_1 * flexo_stiffening
        vs1 = np.sqrt(c44_1_eff / (self.rho1 * M1))
        
        # Piezoelectric substrate electro-acoustic stiffening with electrical leak defect R_E
        piezo_factor = (self.e15_2**2 / self.eps11_2) / (1.0 + self.R_E * kh1)
        c44_2_bar = self.c44_2 + piezo_factor
        vs2 = np.sqrt(c44_2_bar / self.rho2)
        
        return vs1, vs2, c44_1_eff, c44_2_bar, M1

    def solve_phase_velocity(self, kh1):
        """Solve for fundamental Love wave phase velocity c at dimensionless wavenumber kh1."""
        vs1, vs2, c44_1_eff, c44_2_bar, _ = self.get_effective_properties(kh1)
        if vs1 >= vs2:
            return np.nan
        
        def H(c):
            # Safe domain check
            if c <= vs1 or c >= vs2:
                return 0.0
            p1 = np.sqrt((c / vs1)**2 - 1.0)
            p2 = np.sqrt(1.0 - (c / vs2)**2)
            term = c44_1_eff * p1 - self.R_M * kh1 * p1 * c44_2_bar * p2
            return np.sin(p1 * kh1) * term - np.cos(p1 * kh1) * (c44_2_bar * p2)

        grid = np.linspace(vs1 * 1.0001, vs2 * 0.9999, 150)
        vals = [H(c) for c in grid]
        
        # Scan for fundamental mode root (first zero-crossing from vs1 towards vs2)
        for i in range(len(grid) - 1):
            if vals[i] * vals[i+1] <= 0:
                try:
                    return brentq(H, grid[i], grid[i+1], xtol=1e-5)
                except Exception:
                    pass
        return np.nan

    def dispersion_curve(self, kh1_array):
        """Return array of phase velocities over a range of kh1."""
        return np.array([self.solve_phase_velocity(kh) for kh in kh1_array])

    def displacement_profile(self, z_array, kh1, c):
        """
        Calculate normalized transverse acoustic shear displacement u2(z) across:
        - Layer (-h1 <= z <= 0)
        - Substrate (z >= 0)
        """
        vs1, vs2, c44_1_eff, c44_2_bar, _ = self.get_effective_properties(kh1)
        p1 = np.sqrt((c / vs1)**2 - 1.0)
        p2 = np.sqrt(1.0 - (c / vs2)**2)
        k = kh1 / self.h1
        
        # At free surface z = -h1, stress is zero -> u2(z) = A * cos(p1 * k * (z + h1))
        # Amplitude normalized so u2(-h1) = 1.0
        A = 1.0
        u = []
        for z in z_array:
            if z <= 0:
                # Top flexoelectric layer
                val = A * np.cos(p1 * k * (z + self.h1))
            else:
                # Substrate with interface jump:
                # u2_sub(0) = u2_layer(0) + sigma32(0) / K_M
                # where sigma32(0) = - c44_1_eff * p1 * k * A * sin(p1 * kh1)
                u2_layer_0 = A * np.cos(p1 * kh1)
                jump = - self.R_M * kh1 * p1 * A * np.sin(p1 * kh1)
                B = u2_layer_0 - jump
                val = B * np.exp(- p2 * k * z)
            u.append(val)
        return np.array(u)

if __name__ == "__main__":
    solver = SmartCompositeLoveWave()
    kh = np.linspace(0.5, 4.0, 8)
    vels = solver.dispersion_curve(kh)
    print("Physics Solver Ready! Fundamental phase velocities:")
    for k, v in zip(kh, vels):
        print(f"kh1 = {k:.2f} -> c = {v:.2f} m/s")
