import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import requests
import pandas as pd

# ============================================================
# Artemis II (JPL Horizons) vs Apollo 8 CSM
# Apollo 8 è ricostruita come traiettoria crewed plausibile
# basata su eventi ufficiali di missione, non sul target S-IVB.
#
# PATCH COERENTE:
# - Apollo 8 è immersa in un piano orbitale 3D coerente.
# - La quota sopra la Luna è calcolata dalla distanza 3D reale.
# - Il grafico moon-centered usa una proiezione sul piano locale
#   della traiettoria lunare, così resta coerente con la quota.
# - Il grafico Earth-centered usa un piano comune best-fit.
# - Il grafico "Distance from Earth" usa la distanza proiettata
#   nello stesso identico piano del plot Earth-centered, così i
#   due grafici risultano coerenti tra loro.
# ============================================================

HORIZONS_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"

R_EARTH = 6378.1363       # km
R_MOON = 1737.4           # km
MU_MOON = 4902.800066     # km^3/s^2
NMI_TO_KM = 1.852
ENTRY_ALT_KM = 121.92     # km


# -----------------------------
# Horizons utilities
# -----------------------------
def parse_horizons_datetime(s):
    month_map = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
        "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
        "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }

    m = re.match(
        r"(\d{4})-([A-Z]{3})-(\d{1,2})\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?$",
        s.strip()
    )
    if not m:
        raise ValueError(f"Formato data Horizons non riconosciuto: {s}")

    year = int(m.group(1))
    month = month_map[m.group(2)]
    day = int(m.group(3))
    hour = int(m.group(4))
    minute = int(m.group(5))
    second = int(m.group(6))
    frac = m.group(7) or "0"
    micro = int((frac + "000000")[:6])

    return datetime(year, month, day, hour, minute, second, micro)


def format_horizons_datetime(dt):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def horizons_vectors(command, start, stop, step="20 min", center="500@399"):
    """
    Scarica vettori cartesiani da JPL Horizons.
    Restituisce un DataFrame con colonne:
    jd, x, y, z, vx, vy, vz
    """
    params = {
        "format": "text",
        "COMMAND": f"'{command}'",
        "OBJ_DATA": "'NO'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTORS'",
        "CENTER": f"'{center}'",
        "START_TIME": f"'{start}'",
        "STOP_TIME": f"'{stop}'",
        "STEP_SIZE": f"'{step}'",
        "TIME_TYPE": "'UT'",
        "OUT_UNITS": "'KM-S'",
        "REF_PLANE": "'FRAME'",
        "REF_SYSTEM": "'J2000'",
        "VEC_CORR": "'NONE'",
        "VEC_TABLE": "'2'",
        "CSV_FORMAT": "'YES'",
    }

    r = requests.get(HORIZONS_URL, params=params, timeout=120)
    r.raise_for_status()
    txt = r.text

    if "$$SOE" not in txt or "$$EOE" not in txt:
        m = re.search(r'prior to A\.D\. ([0-9A-Z\-:\. ]+?) UT', txt)
        if m:
            first_valid = m.group(1).strip()
            raise RuntimeError(
                f"QUERY_TOO_EARLY|{first_valid}|La query per il target {command} parte troppo presto."
            )

        raise RuntimeError(
            f"Risposta Horizons non parseabile per il target {command}.\n{txt[:1500]}"
        )

    block = txt.split("$$SOE", 1)[1].split("$$EOE", 1)[0]

    rows = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("!"):
            continue

        parts = [p.strip() for p in line.split(",")]
        numeric = []
        for p in parts:
            try:
                numeric.append(float(p.replace("D", "E")))
            except ValueError:
                pass

        if len(numeric) >= 7:
            jd = numeric[0]
            x, y, z, vx, vy, vz = numeric[-6:]
            rows.append((jd, x, y, z, vx, vy, vz))

    if not rows:
        raise RuntimeError(f"Nessuna riga numerica parsata per il target {command}.")

    return pd.DataFrame(rows, columns=["jd", "x", "y", "z", "vx", "vy", "vz"])


def safe_horizons_vectors(command, start, stop, step="20 min", center="500@399"):
    try:
        return horizons_vectors(command, start, stop, step=step, center=center), start
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("QUERY_TOO_EARLY|"):
            first_valid = msg.split("|", 2)[1].strip()
            retry_dt = parse_horizons_datetime(first_valid) + timedelta(minutes=5)
            retry_start = format_horizons_datetime(retry_dt)
            print(f"Target {command}: start troppo presto, riprovo da {retry_start} UT")
            return horizons_vectors(command, retry_start, stop, step=step, center=center), retry_start
        raise


# -----------------------------
# Math helpers
# -----------------------------
def norm_rows(a):
    a = np.asarray(a, dtype=float)
    return np.sqrt(np.sum(a * a, axis=1))


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Vettore nullo.")
    return v / n


def rot_ccw(v):
    return np.array([-v[1], v[0]])


def jd_to_datetime_utc(jd):
    unix_seconds = (jd - 2440587.5) * 86400.0
    return pd.DatetimeIndex(pd.to_datetime(unix_seconds, unit="s", utc=True))


def seconds_from_epoch(dt_series, epoch):
    out = []
    for t in dt_series:
        if hasattr(t, "to_pydatetime"):
            t = t.to_pydatetime()
        if getattr(t, "tzinfo", None) is not None:
            t = t.replace(tzinfo=None)
        out.append((t - epoch).total_seconds())
    return np.array(out)


def interpolate_xyz(times_seconds, ref_seconds, ref_xyz):
    out = np.empty((len(times_seconds), 3))
    for k in range(3):
        out[:, k] = np.interp(times_seconds, ref_seconds, ref_xyz[:, k])
    return out


def kepler_E(M, e, tol=1e-12, maxiter=50):
    """
    Solve Kepler's equation M = E - e sin E for elliptical orbit.
    """
    M = np.asarray(M, dtype=float)
    E = np.where(e < 0.8, M.copy(), np.pi * np.ones_like(M))

    for _ in range(maxiter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = -f / fp
        E = E + dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def ellipse_positions_from_E(a, e, E_values):
    """
    Orbital ellipse in perifocal frame:
      x = a (cosE - e)
      y = a sqrt(1-e^2) sinE
    """
    E_values = np.asarray(E_values)
    x = a * (np.cos(E_values) - e)
    y = a * np.sqrt(1 - e**2) * np.sin(E_values)
    return np.column_stack([x, y])


def transform_2d_to_inertial(xy_pf, e_p, e_q):
    """
    Generalized 2D perifocal -> inertial plane embedding.
    e_p, e_q can be 2D or 3D orthonormal basis vectors.
    """
    return xy_pf[:, [0]] * e_p + xy_pf[:, [1]] * e_q


def lunar_altitude(xyz, moon_xyz):
    """
    Quota fisica sopra la superficie lunare:
        |r_sc - r_moon|_3D - R_MOON
    """
    n = min(len(xyz), len(moon_xyz))
    rel = xyz[:n] - moon_xyz[:n]
    return norm_rows(rel) - R_MOON


def project_to_local_plane(rel_xyz):
    """
    Proietta una traiettoria moon-centered 3D sul piano locale del moto
    vicino al closest approach, invece che sul piano XY globale.
    """
    rel_xyz = np.asarray(rel_xyz, dtype=float)
    r = norm_rows(rel_xyz)

    idx_ca = int(np.argmin(r))
    i_lo = max(0, idx_ca - 2)
    i_hi = min(len(rel_xyz) - 1, idx_ca + 2)

    r0 = rel_xyz[idx_ca]
    v0 = rel_xyz[i_hi] - rel_xyz[i_lo]

    ex = unit(r0)
    h = np.cross(r0, v0)

    if np.linalg.norm(h) < 1e-10:
        nfit = min(200, len(rel_xyz))
        idx = np.argsort(r)[:nfit]
        pts = rel_xyz[idx] - rel_xyz[idx].mean(axis=0)
        _, _, vh = np.linalg.svd(pts, full_matrices=False)
        ex = unit(vh[0])
        ey = unit(vh[1])
    else:
        ez = unit(h)
        ey = unit(np.cross(ez, ex))

    x = rel_xyz @ ex
    y = rel_xyz @ ey
    return x, y


def best_fit_plane_basis(points):
    """
    Restituisce una base ortonormale (e1, e2) per il piano che approssima
    meglio, in senso least-squares, un insieme di vettori 3D passanti
    per l'origine.
    """
    pts = np.asarray(points, dtype=float)
    _, _, vh = np.linalg.svd(pts, full_matrices=False)

    e1 = unit(vh[0])
    e2 = vh[1] - np.dot(vh[1], e1) * e1
    e2 = unit(e2)
    return e1, e2


def project_to_plane(xyz, e1, e2):
    """
    Proietta vettori 3D sul piano generato da e1, e2.
    """
    xyz = np.asarray(xyz, dtype=float)
    u = xyz @ e1
    v = xyz @ e2
    return u, v


# -----------------------------
# Apollo 8 reconstruction
# -----------------------------
def build_apollo8_crewed_reconstruction():
    """
    Reconstruct Apollo 8 CSM using crewed-mission times and orbital parameters.

    Returns:
        dict with keys:
            t_dt, xyz, moon_xyz, meta
    """

    # -------------------------
    # Event times (UTC)
    # -------------------------
    t_sep = datetime(1968, 12, 21, 16, 11, 59, 300000)
    t_loi1_cut = datetime(1968, 12, 24, 10, 3, 26, 500000)
    t_loi2_cut = datetime(1968, 12, 24, 14, 26, 16)
    t_tei_cut = datetime(1968, 12, 25, 6, 13, 39)
    t_entry = datetime(1968, 12, 27, 15, 37, 13)

    # -------------------------
    # Orbital / trajectory parameters
    # -------------------------
    parking_r = R_EARTH + 0.5 * (98.0 + 103.0) * NMI_TO_KM

    rp1 = R_MOON + 60.0 * NMI_TO_KM
    ra1 = R_MOON + 168.5 * NMI_TO_KM

    # near-circular second lunar orbit
    r_circ = R_MOON + 0.5 * (59.7 + 60.7) * NMI_TO_KM

    # Moon ephemeris during Apollo 8
    moon_df, moon_start_used = safe_horizons_vectors(
        "301",
        "1968-12-21 16:00:00",
        "1968-12-27 16:00:00",
        step="10 min",
        center="500@399"
    )

    moon_dt = jd_to_datetime_utc(moon_df["jd"].to_numpy())
    moon_xyz = moon_df[["x", "y", "z"]].to_numpy()

    epoch = moon_dt[0].to_pydatetime().replace(tzinfo=None)
    moon_seconds = seconds_from_epoch(moon_dt, epoch)

    def moon_at(dt):
        t_sec = np.array([(dt - epoch).total_seconds()])
        return interpolate_xyz(t_sec, moon_seconds, moon_xyz)[0]

    def moon_vel_at(dt, h_seconds=60.0):
        p_minus = moon_at(dt - timedelta(seconds=h_seconds))
        p_plus = moon_at(dt + timedelta(seconds=h_seconds))
        return (p_plus - p_minus) / (2.0 * h_seconds)

    # -------------------------
    # Segment 1: Earth-centered translunar coast (3D plane)
    # -------------------------
    n1 = 700
    t1 = pd.date_range(t_sep, t_loi1_cut, periods=n1)

    moon_loi = moon_at(t_loi1_cut)
    moon_vel_loi = moon_vel_at(t_loi1_cut)

    # Radial direction Earth -> Moon at LOI
    u_loi = unit(moon_loi)

    # Tangential direction in the transfer plane, using Moon velocity
    t_loi_vec = moon_vel_loi - np.dot(moon_vel_loi, u_loi) * u_loi
    t_loi_vec = unit(t_loi_vec)

    # Perilune location at LOI
    p_loi = moon_loi + u_loi * rp1

    ra_trans = np.linalg.norm(p_loi)
    a_trans = 0.5 * (parking_r + ra_trans)
    e_trans = (ra_trans - parking_r) / (ra_trans + parking_r)

    e_p_trans = -unit(p_loi)
    e_q_trans = t_loi_vec - np.dot(t_loi_vec, e_p_trans) * e_p_trans
    e_q_trans = unit(e_q_trans)

    xyz1_pf = ellipse_positions_from_E(a_trans, e_trans, np.linspace(0.0, np.pi, n1))
    xyz1 = transform_2d_to_inertial(xyz1_pf, e_p_trans, e_q_trans)

    # Arrival velocity direction at encounter
    v_arr_hat = unit(-e_q_trans)

    # -------------------------
    # Segment 2: initial lunar orbit (168.5 x 60.0 nmi)
    # -------------------------
    n2 = 350
    t2 = pd.date_range(t_loi1_cut, t_loi2_cut, periods=n2)
    t2_sec = seconds_from_epoch(t2, epoch)
    moon2 = interpolate_xyz(t2_sec, moon_seconds, moon_xyz)

    a1 = 0.5 * (rp1 + ra1)
    e1 = (ra1 - rp1) / (ra1 + rp1)
    T1 = 2.0 * np.pi * np.sqrt(a1**3 / MU_MOON)

    dt2 = np.array([(tt.to_pydatetime() - t_loi1_cut).total_seconds() for tt in t2])
    M2 = 2.0 * np.pi * dt2 / T1
    E2 = kepler_E(M2, e1)
    rel2_pf = ellipse_positions_from_E(a1, e1, E2)

    e_r_lunar = unit(p_loi - moon_loi)
    e_t_lunar = v_arr_hat - np.dot(v_arr_hat, e_r_lunar) * e_r_lunar
    e_t_lunar = unit(e_t_lunar)

    rel2 = transform_2d_to_inertial(rel2_pf, e_r_lunar, e_t_lunar)
    xyz2 = moon2 + rel2

    # -------------------------
    # Segment 3: circularized lunar orbit (~60 nmi circular)
    # -------------------------
    n3 = 850
    t3 = pd.date_range(t_loi2_cut, t_tei_cut, periods=n3)
    t3_sec = seconds_from_epoch(t3, epoch)
    moon3 = interpolate_xyz(t3_sec, moon_seconds, moon_xyz)

    T3 = 2.0 * np.pi * np.sqrt(r_circ**3 / MU_MOON)

    x_last = rel2_pf[-1, 0]
    y_last = rel2_pf[-1, 1]
    phi0 = np.arctan2(y_last / r_circ, x_last / r_circ)

    dt3 = np.array([(tt.to_pydatetime() - t_loi2_cut).total_seconds() for tt in t3])
    phi3 = phi0 + 2.0 * np.pi * dt3 / T3

    rel3_pf = np.column_stack([r_circ * np.cos(phi3), r_circ * np.sin(phi3)])
    rel3 = transform_2d_to_inertial(rel3_pf, e_r_lunar, e_t_lunar)
    xyz3 = moon3 + rel3

    # Velocity direction at TEI cutoff
    phi_end = phi3[-1]
    v3_pf = np.array([-np.sin(phi_end), np.cos(phi_end)])
    v_tei_hat = unit(v3_pf[0] * e_r_lunar + v3_pf[1] * e_t_lunar)

    # -------------------------
    # Segment 4: Earth-centered transearth coast (3D plane)
    # -------------------------
    n4 = 900
    t4 = pd.date_range(t_tei_cut, t_entry, periods=n4)

    p_tei = xyz3[-1]
    ra_ret = np.linalg.norm(p_tei)
    rp_ret = R_EARTH + ENTRY_ALT_KM

    a_ret = 0.5 * (ra_ret + rp_ret)
    e_ret = (ra_ret - rp_ret) / (ra_ret + rp_ret)

    e_p_ret = -unit(p_tei)
    e_q_ret = -v_tei_hat
    e_q_ret = e_q_ret - np.dot(e_q_ret, e_p_ret) * e_p_ret
    e_q_ret = unit(e_q_ret)

    xyz4_pf = ellipse_positions_from_E(a_ret, e_ret, np.linspace(np.pi, 2.0 * np.pi - 0.01, n4))
    xyz4 = transform_2d_to_inertial(xyz4_pf, e_p_ret, e_q_ret)

    # -------------------------
    # Stitch together
    # -------------------------
    t_all = np.concatenate([
        t1.to_pydatetime(),
        t2.to_pydatetime()[1:],
        t3.to_pydatetime()[1:],
        t4.to_pydatetime()[1:]
    ])

    xyz_all = np.vstack([xyz1, xyz2[1:], xyz3[1:], xyz4[1:]])

    t_all_sec = seconds_from_epoch(t_all, epoch)
    moon_all = interpolate_xyz(t_all_sec, moon_seconds, moon_xyz)

    meta = {
        "name": "Apollo 8 CSM (NASA report reconstruction)",
        "moon_query_start_used": moon_start_used,
        "t_sep": t_sep,
        "t_loi1_cut": t_loi1_cut,
        "t_loi2_cut": t_loi2_cut,
        "t_tei_cut": t_tei_cut,
        "t_entry": t_entry,
        "parking_orbit_nmi": (98.0, 103.0),
        "initial_lunar_orbit_nmi": (168.5, 60.0),
        "circular_lunar_orbit_nmi": (59.7, 60.7),
    }

    return {
        "t_dt": np.array(t_all),
        "xyz": xyz_all,
        "moon_xyz": moon_all,
        "meta": meta,
    }


# -----------------------------
# Artemis II data
# -----------------------------
def load_artemis_ii():
    start_art = "2026-04-02 02:00:00"
    stop_art = "2026-04-11 00:00:00"
    step_art = "20 min"

    artemis, start_used = safe_horizons_vectors(
        "-1024",
        start_art,
        stop_art,
        step=step_art,
        center="500@399"
    )
    moon_art, _ = safe_horizons_vectors(
        "301",
        start_used,
        stop_art,
        step=step_art,
        center="500@399"
    )

    art_xyz = artemis[["x", "y", "z"]].to_numpy()
    moon_xyz = moon_art[["x", "y", "z"]].to_numpy()

    return {
        "name": "Artemis II (Horizons)",
        "t_dt": jd_to_datetime_utc(artemis["jd"].to_numpy()).to_pydatetime(),
        "xyz": art_xyz,
        "moon_xyz": moon_xyz,
        "raw_df": artemis,
    }


# -----------------------------
# Metrics / plotting
# -----------------------------
def print_metrics(name, xyz, moon_xyz=None):
    r_earth = norm_rows(xyz)
    print(f"=== {name} ===")
    print(f"Samples                       : {len(xyz)}")
    print(f"Max geocentric dist (3D)      : {r_earth.max():,.1f} km")
    print(f"Min geocentric dist (3D)      : {r_earth.min():,.1f} km")

    if moon_xyz is not None:
        n = min(len(xyz), len(moon_xyz))
        rel = xyz[:n] - moon_xyz[:n]
        r_moon = norm_rows(rel)

        print(f"Min lunar altitude (3D)      : {r_moon.min() - R_MOON:,.1f} km")
        print(f"Max lunar distance (3D)      : {r_moon.max():,.1f} km")
    print()


def plot_comparison(artemis, ap8):
    art_xyz = artemis["xyz"]
    ap8_xyz = ap8["xyz"]

    moon_art = artemis["moon_xyz"]
    moon_ap8 = ap8["moon_xyz"]

    art_rel = art_xyz - moon_art
    ap8_rel = ap8_xyz - moon_ap8

    art_alt = lunar_altitude(art_xyz, moon_art)
    ap8_alt = lunar_altitude(ap8_xyz, moon_ap8)

    # Proiezione locale coerente per il confronto attorno alla Luna
    art_moon_u, art_moon_v = project_to_local_plane(art_rel)
    ap8_moon_u, ap8_moon_v = project_to_local_plane(ap8_rel)

    # Piano comune Earth-centered per rendere coerenti
    # il grafico delle orbite e quello della distanza dalla Terra
    earth_plane_points = np.vstack([art_xyz, ap8_xyz, moon_art, moon_ap8])
    e1_earth, e2_earth = best_fit_plane_basis(earth_plane_points)

    art_earth_u, art_earth_v = project_to_plane(art_xyz, e1_earth, e2_earth)
    ap8_earth_u, ap8_earth_v = project_to_plane(ap8_xyz, e1_earth, e2_earth)
    moon_earth_u, moon_earth_v = project_to_plane(moon_art, e1_earth, e2_earth)

    # Distanze proiettate nello stesso piano del plot Earth-centered
    art_r_earth_proj = np.sqrt(art_earth_u**2 + art_earth_v**2)
    ap8_r_earth_proj = np.sqrt(ap8_earth_u**2 + ap8_earth_v**2)

    # Cartella dove si trova questo file .py
    outdir = Path(__file__).resolve().parent

    th = np.linspace(0, 2 * np.pi, 500)

    # --------------------------------
    # Figure 1: Earth-centered in common best-fit plane
    # --------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 10))

    ax1.fill(
        R_EARTH * np.cos(th),
        R_EARTH * np.sin(th),
        alpha=0.35,
        label="Earth"
    )

    moon_r_mean = np.mean(norm_rows(moon_art))
    ax1.plot(
        moon_r_mean * np.cos(th),
        moon_r_mean * np.sin(th),
        ":",
        lw=1.0,
        alpha=0.7,
        color="gray",
        label="Mean lunar distance"
    )

    ax1.plot(
        art_earth_u,
        art_earth_v,
        lw=2.4,
        color="black",
        label=artemis["name"]
    )

    ax1.plot(
        ap8_earth_u,
        ap8_earth_v,
        "--",
        lw=2.0,
        color="#1f77b4",
        label=ap8["meta"]["name"]
    )

    n_art = min(len(art_xyz), len(moon_art))
    r_moon_art_3d = norm_rows(art_xyz[:n_art] - moon_art[:n_art])
    idx_ca_art = np.argmin(r_moon_art_3d)

    moon_ca_u = moon_earth_u[idx_ca_art]
    moon_ca_v = moon_earth_v[idx_ca_art]

    ax1.fill(
        moon_ca_u + R_MOON * np.cos(th),
        moon_ca_v + R_MOON * np.sin(th),
        alpha=0.35,
        label="Moon (at Artemis closest approach)"
    )

    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("u [km]")
    ax1.set_ylabel("v [km]")
    ax1.set_title("Artemis II vs Apollo 8 CSM (Earth-centered best-fit plane)")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(outdir / "01_earth_centered_comparison.png", dpi=300, bbox_inches="tight")

    # --------------------------------
    # Figure 2: Moon-centered zoom in local plane
    # --------------------------------
    fig2, ax2 = plt.subplots(figsize=(9, 9))

    ax2.fill(
        R_MOON * np.cos(th),
        R_MOON * np.sin(th),
        alpha=0.35,
        label="Moon"
    )

    ax2.plot(
        art_moon_u,
        art_moon_v,
        lw=2.4,
        color="black",
        label="Artemis II in Moon frame (local-plane projection)"
    )

    ax2.plot(
        ap8_moon_u,
        ap8_moon_v,
        "--",
        lw=2.0,
        color="#1f77b4",
        label="Apollo 8 CSM in Moon frame (local-plane projection)"
    )

    ax2.set_xlim(-120000, 120000)
    ax2.set_ylim(-120000, 120000)
    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel("u [km]")
    ax2.set_ylabel("v [km]")
    ax2.set_title("Moon-centered comparison (local plane projection)")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(outdir / "02_moon_centered_zoom.png", dpi=300, bbox_inches="tight")

    # --------------------------------
    # Figure 3: distance from Earth in the same display plane
    # --------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    t_art = np.linspace(0, 1, len(art_xyz))
    t_ap8 = np.linspace(0, 1, len(ap8_xyz))

    ax3.plot(
        t_art,
        art_r_earth_proj,
        lw=2.4,
        color="black",
        label=f"{artemis['name']} (projected in display plane)"
    )

    ax3.plot(
        t_ap8,
        ap8_r_earth_proj,
        "--",
        lw=2.0,
        color="#1f77b4",
        label="Apollo 8 CSM (projected in display plane)"
    )

    ax3.set_xlabel("Normalized mission time")
    ax3.set_ylabel("Projected distance from Earth center [km]")
    ax3.set_title("Distance from Earth (same projection as Earth-centered orbit plot)")
    ax3.grid(alpha=0.25)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(outdir / "03_distance_from_earth.png", dpi=300, bbox_inches="tight")

    # --------------------------------
    # Figure 4: altitude above Moon (3D)
    # --------------------------------
    fig4, ax4 = plt.subplots(figsize=(10, 6))

    ax4.plot(
        np.linspace(0, 1, len(art_alt)),
        art_alt,
        lw=2.4,
        color="black",
        label="Artemis II"
    )

    ax4.plot(
        np.linspace(0, 1, len(ap8_alt)),
        ap8_alt,
        "--",
        lw=2.0,
        color="#1f77b4",
        label="Apollo 8 CSM"
    )

    ax4.set_xlabel("Normalized mission time")
    ax4.set_ylabel("Altitude above lunar surface [km]")
    ax4.set_title("Altitude above Moon (3D distance)")
    ax4.grid(alpha=0.25)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(outdir / "04_altitude_above_moon.png", dpi=300, bbox_inches="tight")

    print(f"Figure salvate in: {outdir}")

    plt.show()


def main():
    artemis = load_artemis_ii()
    ap8 = build_apollo8_crewed_reconstruction()

    print_metrics(artemis["name"], artemis["xyz"], artemis["moon_xyz"])
    print_metrics(ap8["meta"]["name"], ap8["xyz"], ap8["moon_xyz"])

    art_alt = lunar_altitude(artemis["xyz"], artemis["moon_xyz"])
    ap8_alt = lunar_altitude(ap8["xyz"], ap8["moon_xyz"])

    print("Moon-relative minima (3D):")
    print(f"  Artemis II min lunar altitude: {art_alt.min():,.1f} km")
    print(f"  Apollo 8 min lunar altitude  : {ap8_alt.min():,.1f} km")
    print()

    print("Apollo 8 reconstruction details:")
    print(f"  Moon ephemeris query start used: {ap8['meta']['moon_query_start_used']} UT")
    print(f"  Separation: {ap8['meta']['t_sep']} UTC")
    print(f"  LOI1 cutoff: {ap8['meta']['t_loi1_cut']} UTC")
    print(f"  LOI2 cutoff: {ap8['meta']['t_loi2_cut']} UTC")
    print(f"  TEI cutoff: {ap8['meta']['t_tei_cut']} UTC")
    print(f"  Entry interface: {ap8['meta']['t_entry']} UTC")
    print()

    plot_comparison(artemis, ap8)


if __name__ == "__main__":
    main()