"""
End-to-End Execution Pipeline:
Love-Type Waves in Smart Composites - Physics Simulation, AI Prediction,
Feature Ranking, Inverse Design, Real-time Flaw Detection, and Lab Transfer Learning.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from physics_solver import SmartCompositeLoveWave
from data_generator import generate_dataset
from rf_predictor import train_and_evaluate
from inverse_design import InverseMaterialDesigner
from defect_detection import AcousticFlawDetector
from transfer_learning import evaluate_transfer_learning, LabDomainSimulator

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig_dir = "artifacts/file_generation/ttl=63d/intermediate/love_wave_ai_project/figures"
os.makedirs(fig_dir, exist_ok=True)

print("="*75)
print("  LOVE-TYPE WAVES IN SMART COMPOSITES: AI & PHYSICS ANALYSIS PIPELINE")
print("="*75)

# -------------------------------------------------------------
# STEP 1: Analytical Physics Dispersion & Size Effect Curves
# -------------------------------------------------------------
print("\n[Step 1/5] Simulating Physics Dispersion Curves & Interface Jump...")
kh_range = np.linspace(0.4, 4.5, 45)

# Effect of Mechanical Interface Slip RM
solver_perfect = SmartCompositeLoveWave(R_M=0.0, R_E=0.0)
solver_slip1 = SmartCompositeLoveWave(R_M=0.05, R_E=0.0)
solver_slip2 = SmartCompositeLoveWave(R_M=0.12, R_E=0.0)

c_perf = solver_perfect.dispersion_curve(kh_range)
c_slip1 = solver_slip1.dispersion_curve(kh_range)
c_slip2 = solver_slip2.dispersion_curve(kh_range)

# Effect of Micro-Inertia Size Scale l0
solver_macro = SmartCompositeLoveWave(l0=0.0, R_M=0.02)
solver_nano1 = SmartCompositeLoveWave(l0=20e-9, R_M=0.02)
solver_nano2 = SmartCompositeLoveWave(l0=40e-9, R_M=0.02)

c_macro = solver_macro.dispersion_curve(kh_range)
c_nano1 = solver_nano1.dispersion_curve(kh_range)
c_nano2 = solver_nano2.dispersion_curve(kh_range)

# Displacement Depth Profile u2(z)
z_depth = np.linspace(-1.0, 3.0, 200) # -1 to 0 in layer, 0 to 3 in substrate (microns)
z_m = z_depth * 1e-6
u_perf = solver_perfect.displacement_profile(z_m, kh1=2.0, c=3500.0)
u_slip = solver_slip2.displacement_profile(z_m, kh1=2.0, c=3350.0)

# Plot Figure 1: Physics Dispersion & Displacement
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

ax1.plot(kh_range, c_perf, 'b-', lw=2.2, label=r'Perfect Joint ($R_M=0.0$)')
ax1.plot(kh_range, c_slip1, 'g--', lw=2.0, label=r'Moderate Slip ($R_M=0.05$)')
ax1.plot(kh_range, c_slip2, 'r-.', lw=2.0, label=r'Severe Debond ($R_M=0.12$)')
ax1.plot(kh_range, c_nano2, 'm:', lw=2.0, label=r'Nano Micro-Inertia ($l_0=40\,$nm)')
ax1.set_xlabel(r'Dimensionless Wavenumber $k h_1$', fontsize=11, fontweight='bold')
ax1.set_ylabel(r'Phase Velocity $c$ [m/s]', fontsize=11, fontweight='bold')
ax1.set_title('(a) Love Wave Dispersion & Defect Sensitivity', fontsize=12, fontweight='bold')
ax1.legend(frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Depth Profile
ax2.plot(u_perf, -z_depth, 'b-', lw=2.2, label='Perfect Joint (Continuous)')
ax2.plot(u_slip, -z_depth, 'r--', lw=2.2, label='Imperfect Joint (Slip Jump at z=0)')
ax2.axhline(0, color='k', linestyle='-', lw=1.5, alpha=0.7)
ax2.fill_between([-0.2, 1.2], 0, 1.0, color='skyblue', alpha=0.2, label='Flexoelectric Top Layer')
ax2.fill_between([-0.2, 1.2], -3.0, 0, color='lightgreen', alpha=0.2, label='Piezoelectric Substrate')
ax2.set_xlabel('Normalized Shear Displacement $u_2(z)$', fontsize=11, fontweight='bold')
ax2.set_ylabel(r'Normalized Depth $z / h_1$', fontsize=11, fontweight='bold')
ax2.set_title('(b) Shear Wave Displacement & Interface Slip Jump', fontsize=12, fontweight='bold')
ax2.set_xlim(-0.1, 1.1)
ax2.set_ylim(-3.0, 1.0)
ax2.legend(frameon=True, fontsize=9.5, loc='lower left')
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig_path1 = os.path.join(fig_dir, "dispersion_and_profiles.png")
fig.savefig(fig_path1)
plt.close(fig)
print(f"Saved: {fig_path1}")

# -------------------------------------------------------------
# STEP 2: Dataset Generation & Random Forest AI Modeling
# -------------------------------------------------------------
print("\n[Step 2/5] Training Random Forest AI & Ranking Physical Variables...")
df = generate_dataset(n_samples=300, random_seed=42)
rf_model, feat_imp, metrics = train_and_evaluate(df)

# Plot Figure 2: Feature Importance Ranking & Prediction Parity
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=200)

labels = [k.replace('_', ' ').title() for k in feat_imp.keys()]
scores = [v * 100 for v in feat_imp.values()]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

bars = ax1.barh(labels[::-1], scores[::-1], color=colors[:len(labels)][::-1], edgecolor='black', alpha=0.85)
ax1.set_xlabel('Relative Permutation Importance (%)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Random Forest Ranking: Which Physical Variables Matter Most?', fontsize=12, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.6)

for bar in bars:
    w = bar.get_width()
    ax1.text(w + 1.0, bar.get_y() + bar.get_height()/2, f"{w:.1f}%", va='center', fontsize=9.5, fontweight='bold')

# Parity plot (True vs Pred)
feat_cols = ['kh1', 'thickness_um', 'micro_inertia_nm', 'flexo_coeff_nC_m', 
             'piezo_coupling', 'mechanical_slip_RM', 'electrical_leak_RE']
X_all = df[feat_cols].values
y_all = df['phase_velocity_mps'].values
y_pred_all = rf_model.predict(X_all)

ax2.scatter(y_all, y_pred_all, alpha=0.65, edgecolors='k', color='teal', s=40, label='AI Predictions')
lims = [min(y_all.min(), y_pred_all.min()) - 50, max(y_all.max(), y_pred_all.max()) + 50]
ax2.plot(lims, lims, 'r--', lw=2.0, label='Ideal 1:1 Parity')
ax2.set_xlabel('Analytical Physics Velocity [m/s]', fontsize=11, fontweight='bold')
ax2.set_ylabel('Random Forest Predicted Velocity [m/s]', fontsize=11, fontweight='bold')
ax2.set_title(f'(b) Model Parity: $R^2={metrics[0]:.4f}$, RMSE={metrics[1]:.1f} m/s', fontsize=12, fontweight='bold')
ax2.legend(frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig_path2 = os.path.join(fig_dir, "feature_importance_and_parity.png")
fig.savefig(fig_path2)
plt.close(fig)
print(f"Saved: {fig_path2}")

# -------------------------------------------------------------
# STEP 3: Inverse Material Design (Reverse Engineering)
# -------------------------------------------------------------
print("\n[Step 3/5] Executing Inverse Material Design for 5G RF Filter Bands...")
designer = InverseMaterialDesigner()
target_cases = [
    (3200.0, 650.0),   # 650 MHz SAW filter
    (3500.0, 900.0),   # 900 MHz GSM/5G band
    (3800.0, 1200.0),  # 1.2 GHz GPS / L-band
    (4100.0, 1800.0)   # 1.8 GHz LTE band
]

opt_results = [designer.design_for_target_velocity(c, f) for c, f in target_cases]
opt_df = pd.DataFrame(opt_results)
print(opt_df[['target_velocity_mps', 'achieved_velocity_mps', 'target_frequency_MHz', 'optimal_thickness_um', 'velocity_error_pct']])

# -------------------------------------------------------------
# STEP 4: Real-Time Acoustic Flaw & Crack Detection
# -------------------------------------------------------------
print("\n[Step 4/5] Simulating Real-Time Waveform & Spectral Flaw Detection...")
detector = AcousticFlawDetector()
states = [
    ("Pristine Joint", 0.00, 0.00, 'navy'),
    ("Micro-Slip Flaw", 0.04, 0.01, 'darkorange'),
    ("Critical Crack", 0.15, 0.02, 'crimson')
]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), dpi=200)

for name, rm, re, col in states:
    sig = detector.simulate_transmitted_signal(rm, re, noise_level=0.01)
    diag = detector.analyze_signal(sig)
    ax1.plot(diag['time_ns'], sig, color=col, lw=1.8, label=f"{name} (Integrity: {diag['joint_integrity_pct']:.0f}%)")
    ax2.plot(diag['freqs_GHz'], diag['fft_vals'], color=col, lw=1.8, label=f"{name}")

ax1.set_xlabel('Time [ns]', fontsize=11, fontweight='bold')
ax1.set_ylabel('Transmitted Amplitude', fontsize=11, fontweight='bold')
ax1.set_title('(a) Real-Time Acoustic Waveform Pulse Transmission Across Joint', fontsize=12, fontweight='bold')
ax1.legend(frameon=True, fontsize=10)
ax1.set_xlim(0, 8.0)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2.set_xlabel('Frequency [GHz]', fontsize=11, fontweight='bold')
ax2.set_ylabel('FFT Spectral Amplitude', fontsize=11, fontweight='bold')
ax2.set_title('(b) Frequency Response: Resonance Shift & Harmonic Distortion in Flawed Joints', fontsize=12, fontweight='bold')
ax2.legend(frameon=True, fontsize=10)
ax2.set_xlim(0, 3.0)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
fig_path3 = os.path.join(fig_dir, "acoustic_flaw_detection.png")
fig.savefig(fig_path3)
plt.close(fig)
print(f"Saved: {fig_path3}")

# -------------------------------------------------------------
# STEP 5: Math-to-Lab Transfer Learning Calibration
# -------------------------------------------------------------
print("\n[Step 5/5] Benchmarking Math-to-Lab Domain Adaptation Transfer Learning...")
transfer_metrics = evaluate_transfer_learning(df)

# Plot Figure 4: Transfer Learning Calibration Comparison
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
models = ['Raw Theoretical\nAI Model', 'Scratch Trained\n(16 Lab Points)', 'Transfer-Learned\nDomain Adapted']
rmse_vals = [transfer_metrics['rmse_raw'], transfer_metrics['rmse_scratch'], transfer_metrics['rmse_adapted']]
bar_colors = ['#d62728', '#ff7f0e', '#2ca02c']

bars = ax.bar(models, rmse_vals, color=bar_colors, edgecolor='black', width=0.55, alpha=0.85)
ax.set_ylabel('Root Mean Square Error (RMSE) [m/s]', fontsize=11, fontweight='bold')
ax.set_title('Bridging Theory & Lab: Transfer Learning Error Reduction', fontsize=12, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.6, axis='y')

for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 8, f"{h:.1f} m/s", ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
fig_path4 = os.path.join(fig_dir, "transfer_learning_benchmark.png")
fig.savefig(fig_path4)
plt.close(fig)
print(f"Saved: {fig_path4}")

print("\n" + "="*75)
print("  ALL PIPELINE STAGES COMPLETED SUCCESSFULLY!")
print("="*75)
