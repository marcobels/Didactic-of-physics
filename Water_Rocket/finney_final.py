"""
Water-rocket (Finney/Gowdy-style) — Euler + air phase + phase shading
--------------------------------------------------------------------
- Fase acqua: Eqs. (4,7,10,15,16,17), aria isoterma durante espansione; burnout quando Vw-->0 o P-->Pa.
- Fase aria: gas ideale isoterma a T_air, nozzle choked/subsonico, thrust = m_dot*v_e + (P_e-Pa)*A_e.
- Coast: nessuna spinta; solo gravità + drag quadratico.
- Patch: preset coerenti, D fisico tipico, dt adattivo, interpolazione all’atterraggio,
         scalari extra e tracking "phase" (0=water,1=air,2=coast).
- Plot: figura 2x2 (T,P,v,y) + grafico Thrust con regioni colorate.

Nota: modello didattico ma plausibile; aria nel serbatoio assunta isoterma.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List

# ----------------------------
# Costanti fisiche
# ----------------------------
Pa = 101_325.0            # Pressione atmosferica [Pa]
g  = 9.80665              # Gravità [m/s^2]
rho_w = 1000.0            # Densità acqua [kg/m^3]

# ----------------------------
# Data classes
# ----------------------------
@dataclass
class RocketGeom:
    VT: float          # Volume interno totale [m^3]
    Ae: float          # Area dell'ugello [m^2]

@dataclass
class DragModel:
    D: float = 9.0e-4  # F_d = D * v * |v| [N / (m/s)^2]

@dataclass
class Masses:
    mr: float = 0.039  # Massa a vuoto [kg]

@dataclass
class GasParams:
    gamma: float = 1.4       # Rapporto calori specifici aria
    R: float = 287.05        # Costante specifica aria [J/(kg K)]
    T_air: float = 293.0     # Temperatura aria nel serbatoio [K] (isoterma)
    Cd_nozzle: float = 0.95  # Coefficiente di scarico dell'ugello

@dataclass
class SimConfig:
    dt: float = 2.0e-5
    t_max: float = 30.0
    stop_at_ground: bool = True

# ----------------------------
# Helper: pompa -> pressione (Eq. 18 Finney)
# ----------------------------
def pressure_after_pumps(P_init: float, V0_air: float, n_pumps: int, Vc: float, Ve: float) -> float:
    P = P_init
    for _ in range(int(n_pumps)):
        P = (P * V0_air + Pa * (Vc + Ve)) / (V0_air + Ve)
    return P

# ----------------------------
# Nozzle gas outflow (choked / subsonic), isentropic
# ----------------------------
def gas_massflow_exit_with_area(P_res: float, Pa_out: float, Ae: float, gas: GasParams) -> Tuple[float, float, float]:
    """
    Calcola (m_dot, v_exit, P_exit) per getto d'aria da serbatoio:
    - P_res: pressione serbatoio (stagnazione) [Pa]
    - Pa_out: pressione ambiente [Pa]
    - Ae: area dell'ugello [m^2]
    - gas: parametri gas (gamma, R, T_air, Cd_nozzle)
    Ritorna:
      m_dot [kg/s], v_exit [m/s], P_exit [Pa]
    """
    gamma, R, T0, Cd = gas.gamma, gas.R, gas.T_air, gas.Cd_nozzle
    P_res = max(P_res, 1.0)  # evita divisioni strane
    pr = Pa_out / P_res
    pr_crit = (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))

    if pr <= pr_crit:
        # Choked
        m_dot = Cd * Ae * P_res * np.sqrt(gamma / (R * T0)) * (2.0 / (gamma + 1.0)) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        P_exit = P_res * (2.0 / (gamma + 1.0)) ** (gamma / (gamma - 1.0))
        v_e = np.sqrt(gamma * R * T0) * np.sqrt(2.0 / (gamma + 1.0))
    else:
        # Non-choked (subsonico)
        term = pr ** ((gamma - 1.0) / gamma)
        m_dot = Cd * Ae * P_res * np.sqrt(gamma / (R * T0)) * (pr ** (1.0 / gamma)) * np.sqrt((2.0 / (gamma - 1.0)) * (1.0 - term))
        P_exit = Pa_out
        v_e = np.sqrt((2.0 * gamma / (gamma - 1.0)) * R * T0 * (1.0 - term))

    return max(m_dot, 0.0), max(v_e, 0.0), P_exit

# ----------------------------
# Core simulator
# ----------------------------
def simulate_one(
    geom: RocketGeom,
    masses: Masses,
    drag: DragModel,
    Vw0: float,
    P0_abs: float = 4.0 * Pa,
    cfg: SimConfig = SimConfig(),
    gas: GasParams = GasParams(),
    y0: float = 0.0,
    v0: float = 0.0,
) -> Dict[str, np.ndarray | float | int]:
    dt_base = cfg.dt

    # Volume aria iniziale (prima di espellere acqua)
    V0_air = geom.VT - Vw0
    if V0_air <= 0.0:
        raise ValueError("Il volume iniziale d'aria deve essere positivo (riduci Vw0 o aumenta VT).")

    # Stato
    t = 0.0
    y = y0
    v = v0
    P = P0_abs  # pressione interna assoluta [Pa]

    # Massa aria (fase gas)
    m_air = None

    # Storage serie
    T_series: List[float] = []
    P_series: List[float] = []
    M_series: List[float] = []
    v_series: List[float] = []
    y_series: List[float] = []
    t_series: List[float] = []
    phase_series: List[int] = []  # 0=water, 1=air, 2=coast

    # Eventi
    burned_out = False
    t_burnout = None
    v_at_burn = np.nan
    y_at_burn = np.nan
    t_apex = np.nan

    # Helper Eq. (15) per fase acqua
    def water_volume_from_P(P_now: float) -> float:
        # Isoterma: P V_air = P0 V0  => V_air = P0*V0 / P
        return geom.VT - (P0_abs * V0_air) / P_now

    def mass_from_P_water(P_now: float) -> float:
        Vw_now = water_volume_from_P(P_now)
        if Vw_now > 0.0 and P_now > Pa:
            return rho_w * Vw_now + masses.mr
        return masses.mr

    ymax = y
    t_of_flight = None

    for _ in range(int(np.ceil(cfg.t_max / dt_base))):
        # Determina fase
        Vw_now = water_volume_from_P(P)
        in_water_thrust = (Vw_now > 1e-12) and (P > Pa + 1e-6)
        in_air_phase    = (not in_water_thrust) and (P > Pa + 1e-6)
        phase_id = 0 if in_water_thrust else (1 if in_air_phase else 2)

        # dt adattivo
        dt_use = dt_base if (in_water_thrust or in_air_phase) else min(20.0 * dt_base, 5e-3)

        # Aggiorna pressione/massa/spinta secondo fase
        if in_water_thrust:
            # Fase acqua: dP/dt Eq. (7)
            dPdt = - (P**2 / (P0_abs * V0_air)) * geom.Ae * np.sqrt(max(2.0 * (P - Pa) / rho_w, 0.0))
            P_next = max(P + dPdt * dt_use, Pa)
            M = mass_from_P_water(P)
            T = 2.0 * (P - Pa) * geom.Ae

        elif in_air_phase:
            # Fase aria: inizializza massa aria al primo step
            if m_air is None:
                m_air = P * geom.VT / (gas.R * gas.T_air)
                burned_out = True
                t_burnout = t
                v_at_burn = v
                y_at_burn = y

            # Portata e velocità di uscita (choked/subsonica)
            m_dot, v_e, P_exit = gas_massflow_exit_with_area(P, Pa, geom.Ae, gas)
            # Aggiorna massa aria e pressione (isoterma)
            m_air_next = max(m_air - m_dot * dt_use, 0.0)
            P_next = m_air_next * gas.R * gas.T_air / geom.VT
            # Spinta residua (momentum + pressure thrust)
            T = m_dot * v_e + (P_exit - Pa) * geom.Ae
            M = masses.mr

        else:
            # Coast: niente spinta, pressione pari a Pa
            P_next = Pa
            T = 0.0
            M = masses.mr

        # Dinamica con drag quadratico
        a = (T / M) - g - (drag.D * v * abs(v)) / M

        # Euler cinematico
        v_next = v + a * dt_use
        y_next = y + v * dt_use
        t_next = t + dt_use

        # Log
        T_series.append(T)
        P_series.append(P)
        M_series.append(M)
        v_series.append(v)
        y_series.append(y)
        t_series.append(t)
        phase_series.append(phase_id)

        # Apice
        if (v > 0.0) and (v_next <= 0.0) and np.isnan(t_apex):
            frac = abs(v) / (abs(v) + max(abs(v_next), 1e-16))
            t_apex = t + frac * dt_use

        # Atterraggio con interpolazione
        if cfg.stop_at_ground and (t > 0.05) and (y_next <= 0.0) and (v_next < 0.0):
            denom = (y - y_next)
            if denom != 0.0:
                frac = y / denom
                t_of_flight = t + frac * dt_use
                # aggiungi ultimo punto coerente opzionale
                t_series.append(t_of_flight)
                y_series.append(0.0)
                v_series.append(v + (v_next - v) * frac)
                P_series.append(P + (P_next - P) * frac)
                M_series.append(M)
                T_series.append(T + (0.0 - T) * frac)
                phase_series.append(phase_id)
            # commit e break
            P, v, y, t = P_next, v_next, y_next, t_next
            break

        # Commit stato
        if in_air_phase:
            m_air = m_air_next
        P, v, y, t = P_next, v_next, y_next, t_next
        if y > ymax:
            ymax = y

    else:
        t_of_flight = t

    out = {
        "t": np.array(t_series),
        "y": np.array(y_series),
        "v": np.array(v_series),
        "P": np.array(P_series),
        "T": np.array(T_series),
        "M": np.array(M_series),
        "phase": np.array(phase_series, dtype=int),  # 0=water,1=air,2=coast
        "ymax": float(ymax),
        "t_burnout": float(t_burnout) if t_burnout is not None else np.nan,
        "y_at_burnout": float(y_at_burn),
        "v_at_burnout": float(v_at_burn),
        "t_apex": float(t_apex),
        "tof": float(t_of_flight) if t_of_flight is not None else np.nan,
        "burned_out": bool(burned_out or (m_air is not None)),
    }
    return out

# ----------------------------
# Sweep su frazione d’acqua
# ----------------------------
def sweep_water_volume(
    geom: RocketGeom,
    masses: Masses,
    drag: DragModel,
    water_fracs: np.ndarray,
    P0_abs: float = 4.0 * Pa,
    cfg: SimConfig = SimConfig(),
    gas: GasParams = GasParams(),
) -> Dict[str, np.ndarray]:
    fracs = np.clip(np.array(water_fracs, dtype=float), 1e-4, 0.9999)
    Vw0s = fracs * geom.VT
    y_max = np.zeros_like(fracs)
    tofs = np.zeros_like(fracs)
    tbo = np.zeros_like(fracs)
    for i, Vw0 in enumerate(Vw0s):
        res = simulate_one(geom, masses, drag, Vw0, P0_abs=P0_abs, cfg=cfg, gas=gas)
        y_max[i] = res["ymax"]
        tofs[i] = res["tof"]
        tbo[i] = res["t_burnout"]
    return {"frac": fracs, "Vw0": Vw0s, "ymax": y_max, "tof": tofs, "t_burnout": tbo}

# ----------------------------
# Preset coerenti
# ----------------------------
PRESETS = {
    "A_2L_bottle": {
        "geom": RocketGeom(VT=2.0e-3, Ae=np.pi*(0.0045)**2),   # 2.0 L, ugello ~9 mm
        "masses": Masses(mr=0.039),
        "drag": DragModel(D=9.0e-4),
        "cfg": SimConfig(dt=2e-5, t_max=30.0, stop_at_ground=True),
        "P0_abs": 3.0*Pa,
        "fill_fracs": np.linspace(0.05, 0.90, 30),
        "single_fill_frac": 0.33,
        "label": "Config A — 2.0 L bottle, 9 mm nozzle, D≈9e-4, mr=39 g",
        "gas": GasParams(gamma=1.4, R=287.05, T_air=293.0, Cd_nozzle=0.95),
    },
    "B_0p75L_small": {
        "geom": RocketGeom(VT=7.5e-4, Ae=np.pi*(0.0031)**2),   # 0.75 L, ~6.2 mm
        "masses": Masses(mr=0.039),
        "drag": DragModel(D=9.0e-4),
        "cfg": SimConfig(dt=2e-5, t_max=30.0, stop_at_ground=True),
        "P0_abs": 3.0*Pa,
        "fill_fracs": np.linspace(0.05, 0.90, 30),
        "single_fill_frac": 0.35,
        "label": "Config B — 0.75 L bottle, 6.2 mm nozzle, D≈9e-4, mr=39 g",
        "gas": GasParams(gamma=1.4, R=287.05, T_air=293.0, Cd_nozzle=0.95),
    },
}

# ----------------------------
# Utility di run/diagnostica
# ----------------------------
def run_single_and_report(preset_key: str) -> Dict[str, float]:
    ps = PRESETS[preset_key]
    geom, masses, drag = ps["geom"], ps["masses"], ps["drag"]
    cfg, P0_abs, gas = ps["cfg"], ps["P0_abs"], ps["gas"]
    Vw0 = ps["single_fill_frac"] * geom.VT
    res = simulate_one(geom, masses, drag, Vw0, P0_abs=P0_abs, cfg=cfg, gas=gas)
    out = {
        "ymax": float(res["ymax"]),
        "t_burnout": float(res["t_burnout"]),
        "tof": float(res["tof"]),
        "fill_frac": float(ps["single_fill_frac"]),
    }
    print(f"[{ps['label']}] "
          f"fill={out['fill_frac']*100:.1f}% | "
          f"ymax={out['ymax']:.2f} m | "
          f"burn={out['t_burnout']:.3f} s | "
          f"ToF={out['tof']:.2f} s")
    return out

def sweep_and_report(preset_key: str) -> Tuple[np.ndarray, np.ndarray]:
    ps = PRESETS[preset_key]
    geom, masses, drag = ps["geom"], ps["masses"], ps["drag"]
    cfg, P0_abs, gas = ps["cfg"], ps["P0_abs"], ps["gas"]
    fracs = ps["fill_fracs"]
    sweep = sweep_water_volume(geom, masses, drag, fracs, P0_abs=P0_abs, cfg=cfg, gas=gas)
    i_opt = int(np.argmax(sweep["ymax"]))
    print(f"Optimal water fraction ~ {sweep['frac'][i_opt]*100:.1f}% -> ymax = {sweep['ymax'][i_opt]:.2f} m")
    return sweep["frac"], sweep["ymax"]

# ----------------------------
# Helper per colorare le fasi nei grafici
# ----------------------------
def _phase_spans(t: np.ndarray, phase: np.ndarray):
    """Restituisce liste di (t_start, t_end, phase_id) per tratti contigui della stessa fase."""
    spans = []
    if len(t) == 0: return spans
    start = t[0]; cur = phase[0]
    for i in range(1, len(t)):
        if phase[i] != cur:
            spans.append((start, t[i], cur))
            start = t[i]
            cur = phase[i]
    spans.append((start, t[-1], cur))
    return spans

def _shade_phases(ax, t: np.ndarray, phase: np.ndarray, alpha=0.12):
    """Colora le tre regioni: acqua (azzurro C0), aria (verde C2), coast (arancio C1)."""
    spans = _phase_spans(t, phase)
    labels_done = set()
    colors = {0: "C0", 1: "C2", 2: "C1"}
    names  = {0: "Fase acqua", 1: "Fase aria", 2: "Coast"}
    for t0, t1, ph in spans:
        if t1 <= t0: continue
        label = names[ph] if names[ph] not in labels_done else None
        ax.axvspan(t0, t1, facecolor=colors.get(ph, "0.8"), alpha=alpha, label=label)
        if label: labels_done.add(names[ph])

def _plot_panel(ax, t, y, phase, ylabel, title):
    ax.plot(t, y)
    _shade_phases(ax, t, phase)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", frameon=False)

# ----------------------------
# Esempio d’uso e plotting con regioni colorate
# ----------------------------
if __name__ == "__main__":
    preset = "A_2L_bottle"  # oppure "B_0p75L_small"

    # Singolo run
    _ = run_single_and_report(preset)

    # Sweep
    fracs, y_max = sweep_and_report(preset)

    # Plot
    try:
        import matplotlib.pyplot as plt
        ps = PRESETS[preset]
        geom, masses, drag = ps["geom"], ps["masses"], ps["drag"]
        cfg, P0_abs, gas = ps["cfg"], ps["P0_abs"], ps["gas"]
        Vw0 = ps["single_fill_frac"] * geom.VT
        res = simulate_one(geom, masses, drag, Vw0, P0_abs=P0_abs, cfg=cfg, gas=gas)

        # Figura 2x2 con regioni colorate
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))
        _plot_panel(axs[0,0], res["t"], res["T"], res["phase"], "T [N]", "Thrust vs time")
        _plot_panel(axs[0,1], res["t"], res["P"], res["phase"], "P [Pa]", "Internal pressure vs time")
        _plot_panel(axs[1,0], res["t"], res["v"], res["phase"], "v [m/s]", "Velocity vs time")
        _plot_panel(axs[1,1], res["t"], res["y"], res["phase"], "y [m]", "Height vs time")
        fig.tight_layout()

        # Grafico singolo: Thrust con bande di fase
        plt.figure()
        ax = plt.gca()
        ax.plot(res["t"], res["T"])
        _shade_phases(ax, res["t"], res["phase"])
        ax.set_xlabel("t [s]"); ax.set_ylabel("T [N]"); ax.set_title("Thrust vs time (phases shaded)")
        ax.legend(loc="best", frameon=False)

        # Altezza max vs frazione d'acqua (trend tipo Fig.4)
        plt.figure()
        plt.plot(fracs*100.0, y_max)
        plt.xlabel("Water volume [% of VT]"); plt.ylabel("Max height [m]")
        plt.title(ps["label"])
        plt.tight_layout(); plt.show()
    except Exception:
        pass
