"""
Single-file Python translation of:
  - main.m
  - animate_t_handle.m
  - (operationally) rotation.m -> torque-free rigid body ODEs + 3-1-3 kinematics

Dependencies:
  numpy, scipy, matplotlib, pillow (for GIF fallback), ffmpeg (optional for MP4)

Run:
  python t_handle_single.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from pathlib import Path


# ------------------------- Rotation / Kinematics (3-1-3) -------------------------

def rot_313_matrices(psi: float, theta: float, phi: float):
    """
    3-1-3 Euler set:
      R1 = Rz(psi), R2 = Rx(theta), R3 = Rz(phi)
    Using the same sign convention as the MATLAB scripts.
    """
    c, s = np.cos, np.sin

    R1 = np.array([[c(psi),  s(psi), 0.0],
                   [-s(psi), c(psi), 0.0],
                   [0.0,     0.0,    1.0]], dtype=float)

    R2 = np.array([[1.0, 0.0,       0.0],
                   [0.0, c(theta),  s(theta)],
                   [0.0, -s(theta), c(theta)]], dtype=float)

    R3 = np.array([[c(phi),  s(phi), 0.0],
                   [-s(phi), c(phi), 0.0],
                   [0.0,     0.0,    1.0]], dtype=float)

    return R1, R2, R3


def euler313_kinematics_matrix(psi: float, theta: float, phi: float) -> np.ndarray:
    """
    Build A(psi,theta,phi) such that:
        omega_body = A @ [psi_dot, theta_dot, phi_dot]
    consistent with the MATLAB construction in rotation.m:
        omega = (R3*R2*R1)*[0;0;psi_dot] + (R3*R2)*[theta_dot;0;0] + R3*[0;0;phi_dot]
    """
    R1, R2, R3 = rot_313_matrices(psi, theta, phi)

    e3 = np.array([0.0, 0.0, 1.0], dtype=float)
    e1 = np.array([1.0, 0.0, 0.0], dtype=float)

    u_psi = (R3 @ R2 @ R1) @ e3  # coefficient of psi_dot
    u_the = (R3 @ R2) @ e1       # coefficient of theta_dot
    u_phi = R3 @ e3              # coefficient of phi_dot

    A = np.column_stack([u_psi, u_the, u_phi])  # 3x3
    return A


# ----------------------------- Dynamics (Torque-free) ----------------------------

def t_handle_ode(t: float, y: np.ndarray, lambda1: float, lambda2: float, lambda3: float) -> np.ndarray:
    """
    State y = [omega1, omega2, omega3, psi, theta, phi]
    Torque-free rigid body in principal axes + 3-1-3 Euler kinematics.
    """
    omega1, omega2, omega3, psi, theta, phi = y
    omega = np.array([omega1, omega2, omega3], dtype=float)

    # Euler equations (torque-free), in principal inertia frame
    domega1 = ((lambda2 - lambda3) / lambda1) * omega2 * omega3
    domega2 = ((lambda3 - lambda1) / lambda2) * omega3 * omega1
    domega3 = ((lambda1 - lambda2) / lambda3) * omega1 * omega2

    # Kinematics: solve A * qdot = omega
    A = euler313_kinematics_matrix(psi, theta, phi)

    # robust solve near singular configurations
    try:
        qdot = np.linalg.solve(A, omega)
    except np.linalg.LinAlgError:
        qdot, *_ = np.linalg.lstsq(A, omega, rcond=None)

    dpsi, dtheta, dphi = qdot
    return np.array([domega1, domega2, domega3, dpsi, dtheta, dphi], dtype=float)


def integrate_t_handle(
    lambda1: float,
    lambda2: float,
    lambda3: float,
    omega0: np.ndarray,
    euler0: np.ndarray,
    tf: float,
    dt: float,
    rtol: float,
    atol: float,
):
    """
    solve_ivp(RK45) ~ MATLAB ode45
    Returns (t, Y) with Y columns: [omega1 omega2 omega3 psi theta phi]
    """
    omega0 = np.asarray(omega0, dtype=float).reshape(3,)
    euler0 = np.asarray(euler0, dtype=float).reshape(3,)
    y0 = np.concatenate([omega0, euler0])

    t_eval = np.arange(0.0, tf + dt, dt)

    sol = solve_ivp(
        fun=lambda t, y: t_handle_ode(t, y, lambda1, lambda2, lambda3),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    return sol.t, sol.y.T


# ----------------------------------- Animation ----------------------------------

def animate_t_handle(
    psi, theta, phi, dt,
    save_video: bool = True,
    video_filename: str = "t_handle.mp4",
    fps_out: int = 30,
    slow_factor: float = 1.0,   # >1 => video più lento, <1 => più veloce
):
    """
    Translation of animate_t_handle.m
    Geometry in cm.

    Video export:
      - MP4 via ffmpeg (preferred)
      - GIF fallback via PillowWriter
    Files are written into ./exports next to the script.
    """
    psi = np.asarray(psi, dtype=float)
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # Dimensions (cm), as in MATLAB
    LAG = 0.5
    LBC = 4.0
    LAD = 2.0

    n = len(psi)

    # Precompute points over time (fast + deterministic rendering)
    e1 = np.zeros((3, n))
    e2 = np.zeros((3, n))
    e3 = np.zeros((3, n))

    xA = np.zeros(n); yA = np.zeros(n); zA = np.zeros(n)
    xB = np.zeros(n); yB = np.zeros(n); zB = np.zeros(n)
    xC = np.zeros(n); yC = np.zeros(n); zC = np.zeros(n)
    xD = np.zeros(n); yD = np.zeros(n); zD = np.zeros(n)

    for k in range(n):
        PSI = psi[k]
        THE = theta[k]
        PHI = phi[k]

        # Same R1,R2,R3 as MATLAB animate script
        R1, R2, R3 = rot_313_matrices(PSI, THE, PHI)
        R = R3 @ R2 @ R1

        # basis vectors (as in MATLAB): (row_vector * R)^T
        e1[:, k] = (np.array([1.0, 0.0, 0.0]) @ R).T
        e2[:, k] = (np.array([0.0, 1.0, 0.0]) @ R).T
        e3[:, k] = (np.array([0.0, 0.0, 1.0]) @ R).T

        # Points A,B,C,D
        xA[k] = -LAG * e2[0, k]
        yA[k] = -LAG * e2[1, k]
        zA[k] = -LAG * e2[2, k]

        xB[k] = xA[k] + (LBC / 2.0) * e1[0, k]
        yB[k] = yA[k] + (LBC / 2.0) * e1[1, k]
        zB[k] = zA[k] + (LBC / 2.0) * e1[2, k]

        xC[k] = xA[k] - (LBC / 2.0) * e1[0, k]
        yC[k] = yA[k] - (LBC / 2.0) * e1[1, k]
        zC[k] = zA[k] - (LBC / 2.0) * e1[2, k]

        xD[k] = xA[k] + LAD * e2[0, k]
        yD[k] = yA[k] + LAD * e2[1, k]
        zD[k] = zA[k] + LAD * e2[2, k]

    # Set up animation figure
    fig = plt.figure()
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel(r"$X$ (cm)")
    ax.set_ylabel(r"$Y$ (cm)")
    ax.set_zlabel(r"$Z$ (cm)")
    ax.grid(True)

    # mimic MATLAB "set(gca,'XDir','reverse','YDir','reverse')"
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(LBC * np.array([-1.0, 1.0]))
    ax.set_ylim(LBC * np.array([-1.0, 1.0]))
    ax.set_zlim(LAD * np.array([-1.0, 1.0]))

    (AD_line,) = ax.plot([xA[0], xD[0]], [yA[0], yD[0]], [zA[0], zD[0]], linewidth=5)
    (BC_line,) = ax.plot([xB[0], xC[0]], [yB[0], yC[0]], [zB[0], zC[0]], linewidth=5)

    # ---- Build frame indices to control playback speed without changing physics ----
    # Simulation sampling frequency (approx)
    fps_sim = 1.0 / max(1e-12, float(dt))

    # If slow_factor > 1, we want fewer sim-steps per rendered frame => smaller stride
    # stride ~ fps_sim / (fps_out * slow_factor)
    stride = max(1, int(round(fps_sim / (float(fps_out) * float(slow_factor)))))
    frame_idx = np.arange(0, n, stride, dtype=int)

    def update(frame_number: int):
        k = int(frame_idx[frame_number])
        AD_line.set_data([xA[k], xD[k]], [yA[k], yD[k]])
        AD_line.set_3d_properties([zA[k], zD[k]])

        BC_line.set_data([xB[k], xC[k]], [yB[k], yC[k]])
        BC_line.set_3d_properties([zB[k], zC[k]])

        return AD_line, BC_line

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_idx),
        interval=1000.0 / float(fps_out),
        blit=False,
    )

    # ---- Export ----
    if save_video:
        out_dir = Path(__file__).resolve().parent / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)

        out_mp4 = out_dir / video_filename
        out_gif = out_dir / (Path(video_filename).stem + ".gif")

        try:
            writer = animation.FFMpegWriter(fps=fps_out, bitrate=1800)
            ani.save(str(out_mp4), writer=writer, dpi=200)
            print(f"Saved video to: {out_mp4}")
        except Exception as exc:
            print(f"FFmpeg export failed ({exc}). Falling back to GIF: {out_gif}")
            writer = animation.PillowWriter(fps=fps_out)
            ani.save(str(out_gif), writer=writer, dpi=150)
            print(f"Saved GIF to: {out_gif}")

    plt.show()


# ------------------------------------- Main -------------------------------------

def main():
    # Parameters from MATLAB main.m
    lambda1 = 0.2  # kg-m^2
    lambda2 = 0.3  # kg-m^2
    lambda3 = 0.4  # kg-m^2

    tf = 10.0
    dt = 0.005
    tol = 1e-6

    omega0 = np.array([0.1, 15.0, 0.1], dtype=float)
    psi0 = 0.0
    theta0 = np.deg2rad(90.0)
    phi0 = 0.0
    euler0 = np.array([psi0, theta0, phi0], dtype=float)

    t, Y = integrate_t_handle(
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3,
        omega0=omega0,
        euler0=euler0,
        tf=tf,
        dt=dt,
        rtol=tol,
        atol=tol,
    )

    omega1 = Y[:, 0]
    omega2 = Y[:, 1]
    omega3 = Y[:, 2]
    psi = Y[:, 3]
    theta = Y[:, 4]
    phi = Y[:, 5]

    # Energy plot
    E = 0.5 * (lambda1 * omega1**2 + lambda2 * omega2**2 + lambda3 * omega3**2)
    span = (0.8, 1.2)

    figE = plt.figure()
    figE.patch.set_facecolor("white")
    plt.plot(t, E, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Total mechanical energy (J)")
    plt.ylim([np.min(E) * span[0], np.max(E) * span[1]])

    # Angular momentum plot
    H1 = lambda1 * omega1
    H2 = lambda2 * omega2
    H3 = lambda3 * omega3
    H = np.sqrt(H1**2 + H2**2 + H3**2)

    figH = plt.figure()
    figH.patch.set_facecolor("white")
    plt.plot(t, H1, linewidth=2, label=r"$\mathbf{H}\cdot\mathbf{e}_1$")
    plt.plot(t, H2, linewidth=2, label=r"$\mathbf{H}\cdot\mathbf{e}_2$")
    plt.plot(t, H3, linewidth=2, label=r"$\mathbf{H}\cdot\mathbf{e}_3$")
    plt.plot(t, H, linewidth=2, label=r"$\|\mathbf{H}\|$")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular momentum about mass center (kg-m^2/s)")
    plt.legend()

    plt.show()

    # Animate + save video
    animate_t_handle(
        psi, theta, phi, dt,
        save_video=True,
        video_filename="t_handle.mp4",
        fps_out=30,
        slow_factor=1.0,  # e.g. 3.0 => 3x più lento
    )


if __name__ == "__main__":
    main()
