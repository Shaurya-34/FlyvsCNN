"""
Phase 2: fly circuit (LPLC2 + LC4 -> GF), tested against a synthetic
expanding disc. No rendering, no drone, no RMO yet.

Switched from a hand-built leaky-integrator model to the actual published
model in Ache et al. 2019, Current Biology 29:1073-1081, "Neural Basis for
Looming Size and Velocity Encoding in the Drosophila Giant Fiber Escape
Pathway" (STAR Methods, "GF Model", equations 1-7, including the two
inhibitory components, equations 5 and 6).

IMPORTANT CAVEAT (from the paper itself): this fitted model was built only
from trials where the GF did NOT spike -- it's a subthreshold synaptic-drive
model, not a spike predictor. The threshold-crossing + reset logic below is
OUR OWN addition, needed because our corridor-flight task needs repeated
discrete evasive maneuvers, not a single-trial escape like the paper's
experiments. It is not part of the published model.

Equation 4, checked against the PDF rendered at 300 dpi: both sides of the
difference are logged, `(ln(theta) - ln(C3))^2`, and the denominator is
2 * C4**2. An earlier version used 2 * C4 from a garbled text extraction,
which made the LPLC2 tuning curve wider than the paper's.

All angles in degrees to match the paper's fit constants (C3=42 deg, etc).
"""

import numpy as np
import matplotlib.pyplot as plt


# --- synthetic looming stimulus (paper's Equation 1) -------------------------

MAX_DISC_DEG = 90  # the paper's discs stop growing at 90 deg and stay on screen
HOLD_S = 0.3  # how long to keep showing the full-size disc, where the tonic inhibition shows up


def looming_disc(k: float, dt: float, lead_multiple: float = 40):
    """theta(t) in degrees and theta_dot(t) in deg/s, for r/v = k seconds
    (the paper's size-to-speed ratio -- only the ratio matters, per Eq 1,
    not r and v separately). The object must approach from far away (small
    angle) before reaching k's timescale, so the window starts at
    lead_multiple*k before contact, not at k itself. Like the paper's
    stimulus, the disc stops at MAX_DISC_DEG and is then held there."""
    ttc = lead_multiple * k  # time-to-contact from the start of our window
    t = np.arange(0, ttc + HOLD_S, dt)
    tau = np.maximum(ttc - t, dt / 10)  # time remaining to contact, clamped away from 0
    theta_rad = 2 * np.arctan(k / tau)
    theta_deg = np.minimum(np.degrees(theta_rad), MAX_DISC_DEG)
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
    peaking at C3=42 degrees."""
    theta_delayed = np.clip(delay(theta_deg, dt, delta2), 1e-3, None)
    return C2 * np.exp(-((np.log(theta_delayed) - np.log(C3)) ** 2) / (2 * C4 ** 2))


# --- the two inhibitory components (paper's Equations 5 and 6) ---------------

def v_i1(theta_deg, dt, C5=-0.53, C6=0.59, C7=66.0, C8=-11.0, delta3=0.0375):
    """Tonic hyperpolarization: a sigmoid in angular size, strongest for large
    objects. Unaffected by LC4 or LPLC2 silencing, so an independent input."""
    theta_delayed = delay(theta_deg, dt, delta3)
    return C5 + C6 / (1 + np.exp(-(theta_delayed - C7) / C8))


def v_i2(theta_deg, dt, C9=-0.52, C10=26.0, C11=7.8, delta4=0.011):
    """Small LC4-dependent inhibition: a Gaussian dip in angular size at 26 deg."""
    theta_delayed = delay(theta_deg, dt, delta4)
    return C9 * np.exp(-((theta_delayed - C10) ** 2) / (2 * C11 ** 2))


# --- full GF drive: paper's Equation 7 ----------------------------------------

def gf_drive(theta_deg, theta_dot_deg, dt, inhibition=True,
             w_lplc2=1.45, w_lc4=1.62, w_i1=2.27, w_i2=1.0):
    drive = w_lplc2 * v_lplc2(theta_deg, dt) + w_lc4 * v_lc4(theta_dot_deg, dt)
    if inhibition:
        drive = drive + w_i1 * v_i1(theta_deg, dt) + w_i2 * v_i2(theta_deg, dt)
    return drive


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

def static_drive(theta_deg, inhibition=True):
    """GF drive for a disc held at a fixed size: no growth, so no LC4 term."""
    theta = np.asarray(theta_deg, dtype=float)
    return gf_drive(theta, np.zeros_like(theta), dt=1.0, inhibition=inhibition)


def half_max_threshold(inhibition=True):
    """Our escape threshold: half the peak of the model's own size tuning."""
    return 0.5 * static_drive(np.arange(1, 181), inhibition).max()


def run_and_plot(r_over_v_ms, dt=0.001, out_path=None):
    k = r_over_v_ms / 1000.0
    t, theta_deg, theta_dot_deg, ttc = looming_disc(k, dt)
    excitatory = gf_drive(theta_deg, theta_dot_deg, dt, inhibition=False)
    drive = gf_drive(theta_deg, theta_dot_deg, dt)
    threshold = half_max_threshold()
    spikes = detect_escapes(t, drive, threshold)

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax[0].plot(t, excitatory, label="excitatory only (LPLC2 + LC4)", color="tab:blue", linestyle="--", linewidth=1)
    ax[0].plot(t, drive, label="full model, with both inhibitory terms", color="tab:blue")
    ax[0].axhline(threshold, color="black", linestyle="--", linewidth=1, label="our escape threshold (half max)")
    for s in spikes:
        ax[0].axvline(s, color="tab:blue", alpha=0.3)
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
        "peak": drive.max(),
        "held_90": drive[-1],  # the full-size disc still on screen at the end
    }


if __name__ == "__main__":
    sizes = np.array([10, 20, 26, 42, 66, 90])
    print("static drive by angular size (deg)")
    print(f"{'':>22}" + "".join(f"{s:>8}" for s in sizes))
    print(f"{'excitatory only':>22}" + "".join(f"{v:>8.2f}" for v in static_drive(sizes, inhibition=False)))
    print(f"{'full model':>22}" + "".join(f"{v:>8.2f}" for v in static_drive(sizes)))
    print(f"half-max threshold: excitatory only {half_max_threshold(False):.3f}, full model {half_max_threshold():.3f}\n")

    results = []
    for r_over_v_ms in (10, 20, 40, 80):  # matches the paper's tested r/v values
        out = f"prototypes/gf_trace_rv_{r_over_v_ms}ms.png"
        results.append(run_and_plot(r_over_v_ms, out_path=out))

    print(f"{'r/v (ms)':>10} {'ttc (ms)':>10} {'1st spike (ms)':>16} {'lead (ms)':>10} {'#spikes':>8} {'peak':>6} {'held 90':>8}")
    for r in results:
        fs = f"{r['first_spike_ms']:.1f}" if r['first_spike_ms'] is not None else "none"
        lt = f"{r['lead_time_ms']:.1f}" if r['lead_time_ms'] is not None else "n/a"
        print(f"{r['r_over_v_ms']:>10} {r['time_to_contact_ms']:>10.1f} {fs:>16} {lt:>10} {r['spike_count']:>8}"
              f" {r['peak']:>6.2f} {r['held_90']:>8.2f}")
