# Love-Type Waves in Smart Composites: Physics & AI Prediction Suite

An end-to-end scientific computing and machine learning framework for analyzing, predicting, and reverse-designing Love-type surface acoustic waves in smart composite structures (Flexoelectric top layer + Piezoelectric substrate) with interface defects and nanoscale micro-inertia.

## 📌 Project Architecture

1. **`physics_solver.py`**:
   - Analytical & continuous dispersion equation root solver for horizontal shear Love waves.
   - Models mechanical interface imperfection via shear compliance ($R_M = c_{44}^{(1)} / (K_M h_1)$).
   - Models electrical interface imperfection via dielectric leakage ($R_E = \epsilon_{11}^{(1)} / (K_E h_1)$).
   - Incorporates strain gradient micro-inertia length scale $l_0$ and flexoelectric gradient coupling $\mu_f$.
   - Computes shear acoustic displacement depth profile $u_2(z)$ showing displacement jumps across imperfect joints.

2. **`data_generator.py`**:
   - Parametric Monte Carlo sampling over physical parameter spaces ($kh_1, h_1, l_0, \mu_f, e_{15}, R_M, R_E$).
   - Computes phase velocity $c$, group velocity $c_g$, resonant frequency $f$, and acoustic wavelength $\lambda$.

3. **`rf_predictor.py`**:
   - Machine Learning Forward Predictor (Random Forest ensemble).
   - Permutation Feature Importance ranking determining which physical variables matter most.
   - Comprehensive test evaluation ($R^2$, RMSE, MAE, Parity plots).

4. **`inverse_design.py`**:
   - Automated Inverse Material Engineering for RF filter / SAW sensor bands.
   - Input: Target phase velocity $c^*$ and operating frequency $f^*$.
   - Output: Optimal layer thickness $h_1$, flexoelectric coefficient $\mu_f$, and defect tolerances.

5. **`defect_detection.py`**:
   - Real-time acoustic wave pulse transmission simulation across pristine vs flawed joints.
   - FFT spectral analysis, non-linear harmonic distortion, time-delay measurement, and structural health integrity scoring.

6. **`transfer_learning.py`**:
   - Domain adaptation bridging idealized mathematical models with noisy, temperature-drifted lab coupon measurements.
   - Validates error reduction using transfer-learned residual adapters with small sample sizes ($N \le 20$).

7. **`run_pipeline.py`**:
   - Master execution script generating high-resolution publication figures.

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run complete end-to-end pipeline
python run_pipeline.py
```
"# love_wave_ai_project" 
