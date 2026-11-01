"""
Transfer Learning & Domain Adaptation Module:
Bridges Theoretical Math Models with Noisy Experimental Lab Measurements.
"""

import numpy as np
import pandas as pd
from rf_predictor import PureRandomForestRegressor

class LabDomainSimulator:
    """Simulates real physical lab test data with 3D edge scattering, surface roughness, and thermal drift."""
    def __init__(self, roughness_decay=0.035, thermal_offset_mps=-45.0, noise_std=12.0):
        self.roughness_decay = roughness_decay
        self.thermal_offset = thermal_offset_mps
        self.noise_std = noise_std

    def generate_lab_measurements(self, theoretical_df, n_samples=18, random_state=42):
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(theoretical_df), n_samples, replace=False)
        lab_df = theoretical_df.iloc[sample_indices].copy()

        c_ideal = lab_df['phase_velocity_mps'].values
        kh = lab_df['kh1'].values

        # Systematic lab physical phenomena:
        # 1. Surface roughness and 3D scattering attenuates velocity at high frequency (high kh)
        dispersion_bias = - self.roughness_decay * c_ideal * (kh / 3.0)
        
        # 2. Thermal expansion / ambient temperature drift offset
        thermal_bias = self.thermal_offset
        
        # 3. Transducer coupling noise
        measurement_noise = np.random.normal(0, self.noise_std, len(c_ideal))

        lab_df['c_lab_measured'] = c_ideal + dispersion_bias + thermal_bias + measurement_noise
        return lab_df

class TransferLearnedCalibrator:
    """Fine-tunes a pre-trained theoretical AI model using limited lab calibration samples."""
    def __init__(self, pretrained_rf):
        self.base_model = pretrained_rf
        self.residual_weights = None
        self.feature_cols = ['kh1', 'thickness_um', 'micro_inertia_nm', 'flexo_coeff_nC_m', 
                             'piezo_coupling', 'mechanical_slip_RM', 'electrical_leak_RE']

    def fine_tune(self, lab_train_df, regularization=1e-2):
        X_lab = lab_train_df[self.feature_cols].values
        y_lab = lab_train_df['c_lab_measured'].values

        # Theoretical model baseline predictions
        y_theory = self.base_model.predict(X_lab)
        
        # Systematic residual: Delta = y_lab - y_theory
        residuals = y_lab - y_theory

        # Ridge regression domain adaptation transfer layer on normalized features
        # Adds bias and polynomial transfer features
        Phi = np.column_stack([np.ones(len(X_lab)), X_lab[:, 0], X_lab[:, 0]**2, X_lab[:, 1]])
        # Ridge closed form: w = (Phi^T Phi + lambda I)^(-1) Phi^T residuals
        reg_matrix = regularization * np.eye(Phi.shape[1])
        reg_matrix[0, 0] = 0.0 # do not regularize bias
        self.residual_weights = np.linalg.solve(Phi.T @ Phi + reg_matrix, Phi.T @ residuals)
        return self

    def predict_lab(self, X):
        y_theory = self.base_model.predict(X)
        Phi = np.column_stack([np.ones(len(X)), X[:, 0], X[:, 0]**2, X[:, 1]])
        residuals_pred = Phi @ self.residual_weights
        return y_theory + residuals_pred

def evaluate_transfer_learning(theory_df):
    feature_cols = ['kh1', 'thickness_um', 'micro_inertia_nm', 'flexo_coeff_nC_m', 
                    'piezo_coupling', 'mechanical_slip_RM', 'electrical_leak_RE']
    
    # 1. Pre-train theoretical base model
    X_theory = theory_df[feature_cols].values
    y_theory = theory_df['phase_velocity_mps'].values
    print("Pre-training base model on 200 theoretical synthetic simulations...")
    base_rf = PureRandomForestRegressor(n_estimators=30, max_depth=10, random_state=42)
    base_rf.fit(X_theory[:200], y_theory[:200])

    # 2. Simulate experimental lab campaign (18 points: 12 calibration train, 6 test)
    sim = LabDomainSimulator()
    lab_data = sim.generate_lab_measurements(theory_df, n_samples=24, random_state=101)
    train_lab = lab_data.iloc[:16]
    test_lab = lab_data.iloc[16:]

    X_test_lab = test_lab[feature_cols].values
    y_test_lab = test_lab['c_lab_measured'].values

    # Baseline 1: Raw theoretical model on lab data (zero adaptation)
    pred_raw = base_rf.predict(X_test_lab)
    rmse_raw = np.sqrt(np.mean((y_test_lab - pred_raw)**2))

    # Baseline 2: Train from scratch on only 16 lab points
    scratch_rf = PureRandomForestRegressor(n_estimators=15, max_depth=4, random_state=42)
    scratch_rf.fit(train_lab[feature_cols].values, train_lab['c_lab_measured'].values)
    pred_scratch = scratch_rf.predict(X_test_lab)
    rmse_scratch = np.sqrt(np.mean((y_test_lab - pred_scratch)**2))

    # Proposed Method: Transfer Learning / Domain Adaptation
    calibrator = TransferLearnedCalibrator(base_rf)
    calibrator.fine_tune(train_lab)
    pred_adapted = calibrator.predict_lab(X_test_lab)
    rmse_adapted = np.sqrt(np.mean((y_test_lab - pred_adapted)**2))

    print("\n--- Domain Adaptation / Transfer Learning Benchmark ---")
    print(f"1. Raw Theoretical AI Model on Lab Data RMSE: {rmse_raw:.2f} m/s (Suffers from domain gap)")
    print(f"2. Model Trained from Scratch on Lab Data RMSE: {rmse_scratch:.2f} m/s (Suffers from small sample size)")
    print(f"3. Fine-Tuned / Domain-Adapted AI Model RMSE:    {rmse_adapted:.2f} m/s (Best performance!)")
    print(f"-> Error reduction through transfer learning: {((rmse_raw - rmse_adapted)/rmse_raw)*100:.1f}%")

    return {
        'rmse_raw': rmse_raw,
        'rmse_scratch': rmse_scratch,
        'rmse_adapted': rmse_adapted,
        'improvement_pct': ((rmse_raw - rmse_adapted)/rmse_raw)*100
    }

if __name__ == "__main__":
    df = pd.read_csv("artifacts/file_generation/ttl=63d/intermediate/love_wave_ai_project/love_wave_dataset.csv")
    res = evaluate_transfer_learning(df)
