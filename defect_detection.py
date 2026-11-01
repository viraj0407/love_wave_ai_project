"""
Real-Time Acoustic Flaw and Crack Detection in Smart Composite Interfaces
Analyzes transmitted Love-wave pulse signals via FFT and energy degradation to detect debonding and leaks.
"""

import numpy as np

class AcousticFlawDetector:
    def __init__(self, pulse_center_freq_GHz=1.0):
        self.fc = pulse_center_freq_GHz * 1e9   # 1 GHz acoustic pulse
        self.t = np.linspace(0, 10e-9, 1024)    # 10 ns time window
        self.dt = self.t[1] - self.t[0]

    def simulate_transmitted_signal(self, RM, RE, noise_level=0.015):
        """
        Simulates transmitted Love-wave acoustic response through the smart composite interface.
        - RM (mechanical slip) attenuates signal amplitude, introduces delay, and non-linear harmonics.
        - RE (electrical leak) alters electro-acoustic coupling, causing frequency detuning.
        """
        t0 = 2.5e-9 + RM * 3.0e-9
        sigma = 0.7e-9 * (1.0 + RM * 2.5)
        
        # Transmission attenuation factor
        A_trans = np.exp(-12.0 * RM)
        
        # Center frequency detuning from dielectric leak
        f_effective = self.fc * (1.0 - 0.22 * RE)
        
        # Gaussian envelope
        env = np.exp(-((self.t - t0) / sigma)**2)
        carrier = np.sin(2.0 * np.pi * f_effective * (self.t - t0))
        
        sig = A_trans * env * carrier
        
        # Harmonic generation from non-linear contact / kissing bond
        if RM > 0.02:
            sig += 0.20 * (RM / 0.15) * (env**1.5) * np.sin(4.0 * np.pi * f_effective * (self.t - t0))
            
        noise = np.random.normal(0, noise_level, len(self.t))
        return sig + noise

    def analyze_signal(self, signal):
        """Extracts spectral and energy features and diagnoses joint condition."""
        N = len(signal)
        fft_vals = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(N, d=self.dt) / 1e9  # GHz

        peak_amp = float(np.max(np.abs(signal)))
        signal_energy = float(np.sum(signal**2))

        # Peak frequency
        peak_freq_idx = np.argmax(fft_vals)
        peak_freq = freqs[peak_freq_idx]

        # Harmonic content around 2.0 GHz
        fund_band = (freqs >= 0.75) & (freqs <= 1.25)
        harm_band = (freqs >= 1.6) & (freqs <= 2.2)
        e_fund = np.sum(fft_vals[fund_band]**2) or 1e-12
        e_harm = np.sum(fft_vals[harm_band]**2)
        harmonic_ratio = float(e_harm / e_fund)

        # Diagnostics classification
        if peak_amp >= 0.80 and abs(peak_freq - 1.0) < 0.05:
            status = "Pristine Joint (Fully Bonded)"
            severity = "Healthy"
            integrity = 98.5
        elif abs(peak_freq - 1.0) >= 0.05 and peak_amp > 0.5:
            status = "Dielectric Breakdown / Joint Electrical Leak"
            severity = "Electrical Fault"
            integrity = 62.0
        elif peak_amp >= 0.45:
            status = "Incipient Mechanical Slippage / Micro-flaw"
            severity = "Warning"
            integrity = 65.0 + (peak_amp - 0.45) * 50.0
        else:
            status = "Critical Joint Debonding / Open Crack"
            severity = "Severe"
            integrity = max(5.0, peak_amp * 80.0)

        return {
            'peak_amplitude': peak_amp,
            'signal_energy': signal_energy,
            'peak_frequency_GHz': float(peak_freq),
            'harmonic_distortion_ratio': harmonic_ratio,
            'status': status,
            'severity': severity,
            'joint_integrity_pct': float(np.clip(integrity, 0.0, 100.0)),
            'time_ns': self.t * 1e9,
            'freqs_GHz': freqs,
            'fft_vals': fft_vals
        }

if __name__ == "__main__":
    detector = AcousticFlawDetector()
    for name, rm, re in [("Pristine", 0.0, 0.0), ("Micro-slip", 0.04, 0.01), ("Severe crack", 0.15, 0.02), ("Electrical leak", 0.01, 0.09)]:
        s = detector.simulate_transmitted_signal(rm, re)
        res = detector.analyze_signal(s)
        print(f"Condition: {name:<16} | Status: {res['status']:<45} | Integrity: {res['joint_integrity_pct']:5.1f}% | Peak Amp: {res['peak_amplitude']:.3f} | Freq: {res['peak_frequency_GHz']:.2f} GHz")
