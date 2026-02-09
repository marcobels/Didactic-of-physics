
"""
Rotating-bar Doppler simulation with 2 sources and 1 receiver + animation (Python 3.8+).

- Simulates f1(t), f2(t) and Δf(t) using the linearized Di Laccio et al. model structure (Eq. 6-7).
- Estimates FFT resolution needed to resolve the Doppler shift.
- Animates the rotating bar + time evolution of Δf(t).

Requirements:
    numpy, matplotlib
For saving MP4:
    ffmpeg installed and available in PATH
For saving GIF:
    pillow installed (pip install pillow)

Author: Marco Bellomo
Date: January 2026
"""

from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


# -------------------------
# Utilities
# -------------------------
def rpm_to_omega(rpm: float) -> float:
    return 2.0 * np.pi * rpm / 60.0


def omega_to_rpm(omega: float) -> float:
    return 60.0 * omega / (2.0 * np.pi)


# -------------------------
# Model
# -------------------------
def doppler_term_linearized(
    t: np.ndarray,
    R: float,
    d0: float,
    omega: float,
    phi: float,
    c: float,
    sign_in_denominator: int,
) -> np.ndarray:
    """
    term(t) = (R*omega/c) * sin(omega t + phi) / sqrt(1 + alpha^2 ± 2 alpha cos(omega t + phi))
    sign_in_denominator = -1 (source 1), +1 (source 2) in the paper geometry.
    """
    alpha = R / d0
    ang = omega * t + phi
    denom = np.sqrt(1.0 + alpha * alpha + sign_in_denominator * 2.0 * alpha * np.cos(ang))
    return (R * omega / c) * (np.sin(ang) / denom)


def simulate_doppler_setup(
    f10: float,
    R1: float,
    d0: float,
    omega: float,
    phi: float = 0.0,
    c: float = 343.0,
    f20: Optional[float] = None,
    R2: Optional[float] = None,
    t_end: float = 10.0,
    fs: float = 200.0,
) -> Dict[str, Any]:
    t = np.arange(0.0, t_end, 1.0 / fs)

    # Source 1: Eq.(6) structure -> f1 = f10*(1 - term1) with denom sign -1
    term1 = doppler_term_linearized(t, R1, d0, omega, phi, c, sign_in_denominator=-1)
    f1 = f10 * (1.0 - term1)
    df1 = f1 - f10

    out: Dict[str, Any] = {
        "t": t,
        "f1": f1,
        "df1": df1,
        "params": {"f10": f10, "R1": R1, "d0": d0, "omega": omega, "phi": phi, "c": c, "t_end": t_end, "fs": fs},
    }

    if (f20 is not None) and (R2 is not None):
        # Source 2: Eq.(7) structure -> f2 = f20*(1 + term2) with denom sign +1
        term2 = doppler_term_linearized(t, R2, d0, omega, phi, c, sign_in_denominator=+1)
        f2 = f20 * (1.0 + term2)
        df2 = f2 - f20
        out.update({"f2": f2, "df2": df2, "params2": {"f20": f20, "R2": R2}})

    return out


# -------------------------
# FFT resolution estimate
# -------------------------
def estimate_fft_requirements(
    f0: float,
    R: float,
    d0: float,
    omega: float,
    c: float = 343.0,
    desired_bins_across_shift: float = 5.0,
    safety_factor: float = 1.5,
) -> Dict[str, float]:
    alpha = R / d0
    ang = np.linspace(0.0, 2.0 * np.pi, 20001)

    denom_minus = np.sqrt(1.0 + alpha * alpha - 2.0 * alpha * np.cos(ang))
    denom_plus = np.sqrt(1.0 + alpha * alpha + 2.0 * alpha * np.cos(ang))
    denom_minus = np.clip(denom_minus, 1e-9, None)
    denom_plus = np.clip(denom_plus, 1e-9, None)

    ratio_minus = np.abs(np.sin(ang) / denom_minus)
    ratio_plus = np.abs(np.sin(ang) / denom_plus)
    ratio_max = float(max(ratio_minus.max(), ratio_plus.max()))

    delta_f_max_est = f0 * (R * omega / c) * ratio_max
    delta_f_fft_target = delta_f_max_est / float(desired_bins_across_shift)
    window_T_min = (1.0 / delta_f_fft_target) if delta_f_fft_target > 0 else np.inf
    window_T_suggested = safety_factor * window_T_min

    return {
        "delta_f_max_est_hz": float(delta_f_max_est),
        "delta_f_fft_target_hz": float(delta_f_fft_target),
        "window_T_min_s": float(window_T_min),
        "window_T_suggested_s": float(window_T_suggested),
        "omega_rpm": float(omega_to_rpm(omega)),
        "alpha": float(alpha),
        "ratio_max": float(ratio_max),
    }


def print_fft_report(label: str, report: Dict[str, float]) -> None:
    print(f"\n=== FFT resolution estimate: {label} ===")
    print(f"alpha = R/d0 = {report['alpha']:.4f}")
    print(f"omega = {report['omega_rpm']:.2f} rpm")
    print(f"Estimated max |sin/denom| over phase = {report['ratio_max']:.4f}")
    print(f"Estimated max Doppler shift |Δf|max ≈ {report['delta_f_max_est_hz']:.3f} Hz")
    print(f"Target FFT resolution Δf_fft <= {report['delta_f_fft_target_hz']:.3f} Hz")
    print(f"Minimum window length T >= {report['window_T_min_s']:.3f} s")
    print(f"Suggested window length T ≈ {report['window_T_suggested_s']:.3f} s")
    print("Rule of thumb: Δf_fft ≈ 1/Twindow (longer window => better freq resolution, worse time resolution)\n")


# -------------------------
# Animation
# -------------------------
def animate_rotating_bar(
    sim: Dict[str, Any],
    f20: Optional[float] = None,
    R2: Optional[float] = None,
    save_path: Optional[str] = None,
    fps: int = 30,
    every_n: int = 1,
    blit: bool = True,   # puoi forzare False se il backend fa storie
) -> None:
    t = sim["t"][::every_n]
    df1 = sim["df1"][::every_n]
    omega = sim["params"]["omega"]
    phi = sim["params"]["phi"]
    d0 = sim["params"]["d0"]
    R1 = sim["params"]["R1"]
    # Fixed colors for sources (used consistently in both panels)
    color1 = "tab:red"
    color2 = "tab:blue"


    has2 = ("df2" in sim) and (f20 is not None) and (R2 is not None)
    if has2:
        df2 = sim["df2"][::every_n]

    obs = np.array([d0, 0.0])

    def pos_sources(tt: float):
        ang = omega * tt + phi
        s1 = np.array([R1 * np.cos(ang), R1 * np.sin(ang)])
        if has2:
            s2 = np.array([R2 * np.cos(ang + np.pi), R2 * np.sin(ang + np.pi)])
        else:
            s2 = None
        return s1, s2

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    ax_geom = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[0, 1])

    # --- Left panel: geometry
    ax_geom.set_aspect("equal", adjustable="box")
    lim = max(d0, R1, (R2 if (R2 is not None) else 0.0)) * 0.2
    ax_geom.set_xlim(-lim - 4.2 * lim, lim + 4.2 * lim)
    ax_geom.set_ylim(-lim - 4.2 * lim, lim + 4.2 * lim)
    ax_geom.set_xlabel("x [m]")
    ax_geom.set_ylabel("y [m]")
    ax_geom.set_title("Rotating bar + observer")

    center_point, = ax_geom.plot([0.0], [0.0], marker="o")
    obs_point, = ax_geom.plot([obs[0]], [obs[1]], marker="o")
    bar_line, = ax_geom.plot([], [], linewidth=2)
    s1_point, = ax_geom.plot([], [], marker="o", color=color1, markersize=8)
    if has2:
        s2_point, = ax_geom.plot([], [], marker="o", color=color2, markersize=8)
    else:
        s2_point = None
    los_line, = ax_geom.plot([0.0, obs[0]], [0.0, obs[1]], linestyle="--", linewidth=1)

    # --- Right panel: time series
    ax_ts.set_xlabel("t [s]")
    ax_ts.set_ylabel(r"$\Delta f(t)$ [Hz]")
    ax_ts.set_title("Doppler shift vs time")

    # PLOT CURVES and keep handles
    line_df1, = ax_ts.plot(t, df1, color="red", label=r"$\Delta f_1(t)$")
    if has2:
        line_df2, = ax_ts.plot(t, df2, color="blue", label=r"$\Delta f_2(t)$")
        cursor_line = ax_ts.axvline(t[0], linewidth=1)

    # Markers: force marker colors to match curve colors
    df1_marker, = ax_ts.plot([t[0]], [df1[0]], marker="o", linestyle="None",
                         color="red")

    if has2:
        df2_marker, = ax_ts.plot([t[0]], [df2[0]], marker="o", linestyle="None",
                                 color="blue")
    else:
        df2_marker = None
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend()

    y_min = df1.min()
    y_max = df1.max()
    if has2:
        y_min = min(y_min, df2.min())
        y_max = max(y_max, df2.max())
    pad = 0.15 * (y_max - y_min + 1e-9)
    ax_ts.set_ylim(y_min - pad, y_max + pad)


    def _artists():
        arts = [bar_line, s1_point, center_point, obs_point, los_line, cursor_line, df1_marker]
        if has2 and (s2_point is not None) and (df2_marker is not None):
            arts.extend([s2_point, df2_marker])
        return arts

    def init():
        bar_line.set_data([], [])
        s1_point.set_data([], [])
        if has2 and s2_point is not None:
            s2_point.set_data([], [])
        # FIX: set_xdata wants a sequence in some matplotlib versions
        cursor_line.set_xdata([t[0], t[0]])
        df1_marker.set_data([t[0]], [df1[0]])
        if has2 and df2_marker is not None:
            df2_marker.set_data([t[0]], [df2[0]])
        return _artists()

    def update(i):
        tt = t[i]
        s1, s2 = pos_sources(tt)

        if has2 and (s2 is not None):
            x = [s2[0], s1[0]]
            y = [s2[1], s1[1]]
        else:
            x = [-s1[0], s1[0]]
            y = [-s1[1], s1[1]]

        bar_line.set_data(x, y)
        s1_point.set_data([s1[0]], [s1[1]])
        if has2 and (s2 is not None) and (s2_point is not None):
            s2_point.set_data([s2[0]], [s2[1]])

        # FIX: sequence, not scalar
        cursor_line.set_xdata([tt, tt])
        df1_marker.set_data([tt], [df1[i]])
        if has2 and (df2_marker is not None):
            df2_marker.set_data([tt], [df2[i]])

        return _artists()

    interval_ms = 1000.0 / float(fps)

    # Robust fallback: if blit causes backend issues, set blit=False
    try:
        ani = FuncAnimation(
            fig, update, frames=len(t), init_func=init, blit=blit, interval=interval_ms
        )
    except Exception:
        ani = FuncAnimation(
            fig, update, frames=len(t), init_func=init, blit=False, interval=interval_ms
        )

    if save_path:
        if save_path.lower().endswith(".mp4"):
            writer = FFMpegWriter(fps=fps)
            ani.save(save_path, writer=writer)
            print(f"Saved animation to: {save_path}")
        elif save_path.lower().endswith(".gif"):
            writer = PillowWriter(fps=fps)
            ani.save(save_path, writer=writer)
            print(f"Saved animation to: {save_path}")
        else:
            raise ValueError("Unsupported extension. Use .mp4 or .gif")

    plt.show()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    # -----------------------
    # EDIT THESE PARAMETERS
    # -----------------------
    # Emitted frequencies [Hz]
    f10 = 4000.0
    f20 = 4400.0  # set to None if only one source

    # Geometry [m]
    R1 = 0.60
    R2 = 0.36  # set to None if only one source
    d0 = 0.70

    # Rotation rate
    rpm = 95.0
    omega = rpm_to_omega(rpm)

    # Phase [rad]
    phi = 0.0

    # Sound speed [m/s]
    c = 343.0

    # Simulation duration (seconds) and sampling for the *model* time series
    t_end = 8.0
    fs = 400.0

    # -----------------------
    # FFT REPORT
    # -----------------------
    rep1 = estimate_fft_requirements(f0=f10, R=R1, d0=d0, omega=omega, c=c)
    print_fft_report("Source 1", rep1)

    if (f20 is not None) and (R2 is not None):
        rep2 = estimate_fft_requirements(f0=f20, R=R2, d0=d0, omega=omega, c=c)
        print_fft_report("Source 2", rep2)

    # -----------------------
    # SIMULATE
    # -----------------------
    sim = simulate_doppler_setup(
        f10=f10,
        f20=f20,
        R1=R1,
        R2=R2,
        d0=d0,
        omega=omega,
        phi=phi,
        c=c,
        t_end=t_end,
        fs=fs,
    )

    # -----------------------
    # QUICK STATIC PLOTS
    # -----------------------
    t = sim["t"]
    plt.figure()
    plt.plot(t, sim["df1"], label=r"$\Delta f_1(t)$")
    if "df2" in sim:
        plt.plot(t, sim["df2"], label=r"$\Delta f_2(t)$")
    plt.xlabel("t [s]")
    plt.ylabel(r"$\Delta f(t)$ [Hz]")
    plt.title("Doppler shift in frequency vs time (linearized model)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    # -----------------------
    # ANIMATE
    # -----------------------
    # If you want to save, set e.g. save_path="doppler_bar.mp4" or "doppler_bar.gif"
    animate_rotating_bar(
        sim=sim,
        f20=f20,
        R2=R2,
        save_path="/Users/marcobellomo/Documents/Master_Udine/Moti relativi/Doppler/doppler_bar.mp4",  # e.g. "doppler_bar.mp4"
        fps=30,
        every_n=2,        # increase to reduce frames if saving is slow/heavy
    )
