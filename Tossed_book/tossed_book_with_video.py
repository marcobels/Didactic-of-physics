import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from pathlib import Path


def eom(t, y, m, g, lambda1, lambda2, lambda3):
    """
    Equations of motion, translated from MATLAB EOM().
    State y has length 12:
      [x1dot, x2dot, x3dot, x1, x2, x3, omega1, omega2, omega3, psi, theta, phi]
    """
    x1dot, x2dot, x3dot = y[0:3]
    x1, x2, x3 = y[3:6]
    omega1, omega2, omega3 = y[6:9]
    psi, theta, phi = y[9:12]

    # Build the "mass matrix" M (12x12), matching MATLAB blkdiag structure
    M = np.zeros((12, 12), dtype=float)

    # m*I3 for acceleration block (x_ddot)
    M[0:3, 0:3] = m * np.eye(3)

    # I3 for position kinematics (x_dot)
    M[3:6, 3:6] = np.eye(3)

    # Diagonal inertia matrix for angular acceleration block (omega_dot)
    M[6:9, 6:9] = np.diag([lambda1, lambda2, lambda3])

    # Euler-angle kinematics block (3-2-1): matrix multiplying [psi_dot, theta_dot, phi_dot]^T
    M[9:12, 9:12] = np.array([
        [-np.sin(theta), 0.0, 1.0],
        [np.cos(theta) * np.sin(phi), np.cos(phi), 0.0],
        [np.cos(theta) * np.cos(phi), -np.sin(phi), 0.0],
    ])

    # Right-hand side f (12,)
    f = np.array([
        0.0,
        0.0,
        -m * g,
        x1dot,
        x2dot,
        x3dot,
        -(lambda3 - lambda2) * omega2 * omega3,
        -(lambda1 - lambda3) * omega1 * omega3,
        -(lambda2 - lambda1) * omega1 * omega2,
        omega1,
        omega2,
        omega3
    ], dtype=float)

    # Solve M * dy = f
    dy = np.linalg.solve(M, f)
    return dy


def rotmat_321(psi, theta, phi):
    """3-2-1 Euler rotation matrix, matching the MATLAB product."""
    c1, s1 = np.cos(psi), np.sin(psi)
    c2, s2 = np.cos(theta), np.sin(theta)
    c3, s3 = np.cos(phi), np.sin(phi)

    Rz = np.array([[c1, -s1, 0],
                   [s1,  c1, 0],
                   [0,   0,  1]], dtype=float)
    Ry = np.array([[c2, 0, s2],
                   [0,  1, 0],
                   [-s2, 0, c2]], dtype=float)
    Rx = np.array([[1, 0,  0],
                   [0, c3, -s3],
                   [0, s3,  c3]], dtype=float)
    return Rz @ Ry @ Rx


def main():
    # Physical parameters
    L = 25/100     # m
    w = 18/100     # m
    b = 2.5/100    # m
    m = 0.7        # kg
    g = 9.8        # m/s^2

    # Principal moments of inertia
    lambda1 = (m/12) * (w**2 + b**2)
    lambda2 = (m/12) * (L**2 + b**2)
    lambda3 = (m/12) * (w**2 + L**2)

    # Initial conditions
    xdot0 = np.array([0.0, 0.0, 5.0])
    x0 = np.array([0.0, 0.0, 0.0])
    omega0 = np.array([0.1, 20.0, 0.1])
    Q0 = np.array([0.0, 0.0, 0.0])  # [psi, theta, phi]
    y0 = np.concatenate([xdot0, x0, omega0, Q0])

    # Simulation parameters
    dt = 0.005
    tf = 1.0
    t_eval = np.arange(0.0, tf + dt, dt)
    tol = 1e-8

    # Integrate using solve_ivp (RK45 ~ ode45)
    sol = solve_ivp(
        fun=lambda t, y: eom(t, y, m, g, lambda1, lambda2, lambda3),
        t_span=(t_eval[0], t_eval[-1]),
        y0=y0,
        t_eval=t_eval,
        rtol=tol,
        atol=tol,
        method="RK45"
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    T = sol.t
    Y = sol.y.T  # shape (N, 12)

    # Extract solutions
    x1, x2, x3 = Y[:, 3], Y[:, 4], Y[:, 5]
    psi, theta, phi = Y[:, 9], Y[:, 10], Y[:, 11]

    # Compute corotational basis vectors e1,e2,e3 over time
    e1seq = np.zeros((3, len(T)))
    e2seq = np.zeros((3, len(T)))
    e3seq = np.zeros((3, len(T)))

    for i in range(len(T)):
        Qmat = rotmat_321(psi[i], theta[i], phi[i])
        e1seq[:, i] = Qmat[:, 0]
        e2seq[:, i] = Qmat[:, 1]
        e3seq[:, i] = Qmat[:, 2]

    # ----------------------------------------------------------------------
    # Plot mass center position and orientation
    fig1 = plt.figure()
    fig1.patch.set_facecolor("white")
    fig1.canvas.manager.set_window_title("Mass center position and orientation over time")

    ax1 = fig1.add_subplot(1, 2, 1)
    ax1.plot(T, x1, linewidth=2, label=r"$x_1$")
    ax1.plot(T, x2, linewidth=2, label=r"$x_2$")
    ax1.plot(T, x3, linewidth=2, label=r"$x_3$")
    ax1.set_title("Mass center position over time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Mass center position (m)")
    ax1.legend()

    ax2 = fig1.add_subplot(1, 2, 2)
    ax2.plot(T, np.degrees(psi), linewidth=2, label=r"$\psi$")
    ax2.plot(T, np.degrees(theta), linewidth=2, label=r"$\theta$")
    ax2.plot(T, np.degrees(phi), linewidth=2, label=r"$\phi$")
    ax2.set_title("Orientation over time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Angle (deg)")
    ax2.legend()

    # ----------------------------------------------------------------------
    # Plot trajectories of basis vectors tips on unit sphere
    fig2 = plt.figure()
    fig2.patch.set_facecolor("white")
    fig2.canvas.manager.set_window_title("Evolution of the corotational basis")
    ax = fig2.add_subplot(111, projection="3d")

    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    Xs = np.outer(np.cos(u), np.sin(v))
    Ys = np.outer(np.sin(u), np.sin(v))
    Zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(Xs, Ys, Zs, alpha=0.1, linewidth=0.2)

    ax.set_title("Trajectories traced by the corotational basis on the unit sphere\n"
                 "e1=blue, e2=red, e3=black")
    ax.set_xlabel(r"$E_1$")
    ax.set_ylabel(r"$E_2$")
    ax.set_zlabel(r"$E_3$")
    ax.set_box_aspect((1, 1, 1))

    ax.quiver(0, 0, 0, e1seq[0, 0], e1seq[1, 0], e1seq[2, 0], linewidth=2)
    ax.quiver(0, 0, 0, e2seq[0, 0], e2seq[1, 0], e2seq[2, 0], linewidth=2)
    ax.quiver(0, 0, 0, e3seq[0, 0], e3seq[1, 0], e3seq[2, 0], linewidth=2)

    ax.plot(e1seq[0, :], e1seq[1, :], e1seq[2, :], linewidth=1.5)
    ax.plot(e2seq[0, :], e2seq[1, :], e2seq[2, :], linewidth=1.5)
    ax.plot(e3seq[0, :], e3seq[1, :], e3seq[2, :], linewidth=1.5)

    # ----------------------------------------------------------------------
    # Animation of tossed book using Poly3DCollection
    fig3 = plt.figure()
    fig3.patch.set_facecolor("white")
    fig3.canvas.manager.set_window_title("Animation of the tossed book")
    ax3 = fig3.add_subplot(111, projection="3d")
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xlabel(r"$x_1$ (m)")
    ax3.set_ylabel(r"$x_2$ (m)")
    ax3.set_zlabel(r"$x_3$ (m)")
    ax3.grid(True)
    ax3.view_init(elev=30, azim=135)

    ax3.set_xlim([np.min(x1) - L, np.max(x1) + L])
    ax3.set_ylim([np.min(x2) - L, np.max(x2) + L])
    ax3.set_zlim([np.min(x3) - L, np.max(x3) + L])

    rCM = np.vstack([x1, x2, x3])  # (3,N)

    r1 = rCM + 0.5*L*e1seq + 0.5*w*e2seq + 0.5*b*e3seq
    r2 = rCM - 0.5*L*e1seq + 0.5*w*e2seq + 0.5*b*e3seq
    r3p = rCM - 0.5*L*e1seq - 0.5*w*e2seq + 0.5*b*e3seq
    r4 = rCM + 0.5*L*e1seq - 0.5*w*e2seq + 0.5*b*e3seq
    r5 = rCM + 0.5*L*e1seq + 0.5*w*e2seq - 0.5*b*e3seq
    r6 = rCM - 0.5*L*e1seq + 0.5*w*e2seq - 0.5*b*e3seq
    r7 = rCM - 0.5*L*e1seq - 0.5*w*e2seq - 0.5*b*e3seq
    r8 = rCM + 0.5*L*e1seq - 0.5*w*e2seq - 0.5*b*e3seq

    # faces defined by 4 vertices each
    def faces_at(j):
        v1 = np.array([r1[:, j], r2[:, j], r3p[:, j], r4[:, j]])
        v2 = np.array([r5[:, j], r6[:, j], r7[:, j], r8[:, j]])
        v3 = np.array([r3p[:, j], r4[:, j], r8[:, j], r7[:, j]])
        v4 = np.array([r1[:, j], r2[:, j], r6[:, j], r5[:, j]])
        v5 = np.array([r1[:, j], r4[:, j], r8[:, j], r5[:, j]])
        v6 = np.array([r2[:, j], r3p[:, j], r7[:, j], r6[:, j]])
        return [v1, v2, v3, v4, v5, v6]

    faces0 = faces_at(0)
    poly = Poly3DCollection(faces0, alpha=0.5)
    ax3.add_collection3d(poly)

    # highlight vertex r6
    point6, = ax3.plot([r6[0, 0]], [r6[1, 0]], [r6[2, 0]], marker="o")

    # ------------------------------------------------------------------
    # Video export (MP4 preferred; GIF fallback)  --- PATCHED ---
    save_video = True
    fps_out = 30

    # output directory (writeable)
    out_dir = Path(__file__).resolve().parent / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_mp4 = out_dir / "tossed_book.mp4"
    out_gif = out_dir / "tossed_book.gif"

    # Build a frame index list so we do not generate a 1000+ FPS movie
    dt_sim = float(np.mean(np.diff(T))) if len(T) > 1 else dt
    fps_sim = 1.0 / dt_sim
    slow_factor = 5.0   # 3 = video 5 volte più lento
    stride = max(1, int(round(fps_sim / (fps_out * slow_factor))))
    frame_idx = np.arange(0, len(T), stride, dtype=int)

    def update(frame_number):
        j = int(frame_idx[frame_number])
        poly.set_verts(faces_at(j))
        point6.set_data([r6[0, j]], [r6[1, j]])
        point6.set_3d_properties([r6[2, j]])
        return poly, point6

    ani = animation.FuncAnimation(
        fig3,
        update,
        frames=len(frame_idx),
        interval=1000.0 / fps_out,
        blit=False,
    )

    if save_video:
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


if __name__ == "__main__":
    main()
