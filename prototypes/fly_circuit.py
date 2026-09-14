"""
Phase 2: fly circuit (LPLC2 + LC4 -> GF), tested against a synthetic
expanding disc. No rendering, no drone, no RMO yet.

Switched from a hand-built leaky-integrator model to the actual published
model in Ache et al. 2019, Current Biology 29:1073-1081, "Neural Basis for
Looming Size and Velocity Encoding in the Drosophila Giant Fiber Escape
Pathway" (STAR Methods, "GF Model", equations 1-4 and 7; equations 5-6 are
the two inhibitory components, DROPPED here per project decision -- see
project memory).

IMPORTANT CAVEAT (from the paper itself): this fitted model was built only
from trials where the GF did NOT spike -- it's a subthreshold synaptic-drive
model, not a spike predictor. The threshold-crossing + reset logic below is
OUR OWN addition, needed because our corridor-flight task needs repeated
discrete evasive maneuvers, not a single-trial escape like the paper's
experiments. It is not part of the published model.

TRANSCRIPTION FLAG: equation 4 in the PDF extraction reads as a Gaussian
with only one side of the difference logged, `(|theta| - ln(C3))^2`, which
doesn't square dimensionally. Implemented here as a proper log-normal
Gaussian, `(ln(theta) - ln(C3))^2` -- both sides logged. This is my
inference from an imperfect OCR read, not a confirmed transcription -- worth
checking against the actual PDF equation rendering if precision matters.

All angles in degrees to match the paper's fit constants (C3=42 deg, etc).
"""

import numpy as np
import matplotlib.pyplot as plt


# --- synthetic looming stimulus (paper's Equation 1) -------------------------

def looming_disc(k: float, dt: float, lead_multiple: float = 40):
    """theta(t) in degrees and theta_dot(t) in deg/s, for r/v = k seconds
    (the paper's size-to-speed ratio -- only the ratio matters, per Eq 1,
    not r and v separately). The object must approach from far away (small
    angle) before reaching k's timescale, so the window starts at
    lead_multiple*k before contact, not at k itself."""
    ttc = lead_multiple * k  # time-to-contact from the start of our window
    t = np.arange(0, ttc + 5 * k, dt)
    tau = np.maximum(ttc - t, dt / 10)  # time remaining to contact, clamped away from 0
    theta_rad = 2 * np.arctan(k / tau)
    theta_deg = np.degrees(theta_rad)
    theta_dot_deg = np.gradient(theta_deg, dt)
    return t, theta_deg, theta_dot_deg, ttc


def delay(signal: np.ndarray, dt: float, delay_s: float) -> np.ndarray:
    """Shift a signal later in time by delay_s seconds (sensory latency),
    holding the initial value for the padded region."""
    shift = int(round(delay_s / dt))
    if shift <= 0:
        return signal
    return np.concatenate([np.full(shift, signal[0]), signal[:-shift]])


# --- the two excitatory components (paper's Equations 3 and 4) ---------------

def v_lc4(theta_dot_deg, dt, C1=0.0002567, delta1=0.019):
    """LC4 -> GF: true angular-velocity (rho) encoder, linear through origin."""
    return C1 * delay(theta_dot_deg, dt, delta1)


def v_lplc2(theta_deg, dt, C2=1.7, C3=42.0, C4=0.52, delta2=0.019):
    """LPLC2 -> GF: angular-size (eta) encoder, Gaussian in log(theta),
    peaking at C3=42 degrees. See TRANSCRIPTION FLAG above."""
    theta_delayed = np.clip(delay(theta_deg, dt, delta2), 1e-3, None)
    return C2 * np.exp(-((np.log(theta_delayed) - np.log(C3)) ** 2) / (2 * C4))


# --- full GF drive: paper's Equation 7, inhibitory terms dropped -------------

def gf_drive(theta_deg, theta_dot_deg, dt, w_lplc2=1.45, w_lc4=1.62):
    return w_lplc2 * v_lplc2(theta_deg, dt) + w_lc4 * v_lc4(theta_dot_deg, dt)


# --- OUR OWN addition: threshold-crossing escape event, not from the paper ---

def detect_escapes(t, drive, threshold):
    spikes = []
    armed = True
    for i in range(len(t)):
        if armed and drive[i] >= threshold:
            spikes.append(t[i])
            armed = False
        elif drive[i] < threshold * 0.8:  # simple hysteresis so it can re-trigger
            armed = True
    return spikes


# --- verification run ---------------------------------------------------------

def run_and_plot(r_over_v_ms, dt=0.001, threshold=1.0, out_path=None):
    k = r_over_v_ms / 1000.0
    t, theta_deg, theta_dot_deg, ttc = looming_disc(k, dt)
    drive = gf_drive(theta_deg, theta_dot_deg, dt)
    spikes = detect_escapes(t, drive, threshold)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].plot(t, drive, label="GF drive, V_LPLC2*w + V_LC4*w", color="tab:red")
    ax[0].axhline(threshold, color="black", linestyle="--", linewidth=1, label="our escape threshold")
    for s in spikes:
        ax[0].axvline(s, color="tab:red", alpha=0.3)
    ax[0].axvline(ttc, color="gray", linestyle=":", linewidth=1, label="contact")
    ax[0].set_ylabel("GF drive (mV, modeled)")
    ax[0].set_title(f"r/v={r_over_v_ms}ms, time-to-contact={ttc*1000:.0f}ms")
    ax[0].legend(loc="upper left", fontsize=8)

    ax[1].plot(t, theta_deg, color="tab:green")
    ax[1].set_ylabel("angular size (deg)")
    ax[1].set_xlabel("time (s)")

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=120)
    plt.close(fig)

    first_spike = spikes[0] if spikes else None
    lead_time = (ttc - first_spike) if first_spike is not None else None
    return {
        "r_over_v_ms": r_over_v_ms,
        "time_to_contact_ms": ttc * 1000,
        "first_spike_ms": first_spike * 1000 if first_spike is not None else None,
        "lead_time_ms": lead_time * 1000 if lead_time is not None else None,
        "spike_count": len(spikes),
    }


if __name__ == "__main__":
    results = []
    for r_over_v_ms in (10, 20, 40, 80):  # matches the paper's tested r/v values
        out = f"prototypes/gf_trace_rv_{r_over_v_ms}ms.png"
        results.append(run_and_plot(r_over_v_ms, out_path=out))

    print(f"{'r/v (ms)':>10} {'ttc (ms)':>10} {'1st spike (ms)':>16} {'lead (ms)':>10} {'#spikes':>8}")
    for r in results:
        fs = f"{r['first_spike_ms']:.1f}" if r['first_spike_ms'] is not None else "none"
        lt = f"{r['lead_time_ms']:.1f}" if r['lead_time_ms'] is not None else "n/a"
        print(f"{r['r_over_v_ms']:>10} {r['time_to_contact_ms']:>10.1f} {fs:>16} {lt:>10} {r['spike_count']:>8}")
