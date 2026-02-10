# slinky_drop.py
# Python port of: "Slinky drop - Lumped-element model" (MATLAB)
# ---------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from matplotlib.animation import FFMpegWriter
from scipy.integrate import solve_ivp
from pathlib import Path

# -----------------------------
# Model helpers
# -----------------------------
def create_model(k, m, N, c, g):
    """
    Build state-space matrices A, B for z = [positions(0..N-1); velocities(0..N-1)]
    c = number of collisions already occurred (top c+1 masses stick together)
    """
    # Base tridiagonal for accelerations (A_model)
    A1 = (-2.0 * k / m) * np.eye(N)
    A2 = (k / m) * np.diag(np.ones(N - 1), +1)
    A3 = (k / m) * np.diag(np.ones(N - 1), -1)
    A_model = A1 + A2 + A3

    # End masses have only one spring
    A_model[0, 0] = -k / m
    A_model[-1, -1] = -k / m

    # Update after c >= 1 collisions (top block moves together)
    if c >= 1:
        A_model[:, :c] = 0.0
        A_model[0 : (c + 1), c] = -k / ((1 + c) * m)
        if c + 1 < N:
            A_model[0 : (c + 1), c + 1] = +k / ((1 + c) * m)

    # Assemble full state matrix
    Z = np.zeros((N, N))
    A = np.block([[Z, np.eye(N)], [A_model, Z]])

    # Gravity input
    B = np.concatenate([np.zeros(N), g * np.ones(N)])
    return A, B


def model_springs(t, z, A, B):
    return A @ z + B


def mass_collision_factory(c):
    """
    Event: first two *free* masses collide: y[c] - y[c+1] == 0
    (0-based indexing here)
    """
    def event(t, y):
        return y[c] - y[c + 1]
    event.terminal = True
    event.direction = 0
    return event


# -----------------------------
# Main simulation
# -----------------------------
def main():
    # --- Parameters
    # Slinky
    N = 10         # number of elements
    k = 10.0       # spring constant [N/m]
    m = 0.01       # mass [kg]
    g = 9.81       # gravity [m/s^2]

    # Video / timing (simulazione)
    playback_speed = 0.005
    tF = 5.0
    fR = 30.0 / playback_speed        # simulation "frame rate"
    dt = 1.0 / fR
    time = np.linspace(0.0, tF, int(tF * fR))

    # ---------- First phase ----------
    c = 0  # collisions so far
    A, B = create_model(k, m, N, c, g)

    # Static initial condition: solve for positions 2..N (MATLAB slices mapped to 0-based)
    # Rows N+2..2N  ->  N+1 : 2N
    # Cols 2..N     ->  1   : N
    A_static = A[(N + 1) : (2 * N), 1:N]
    B_static = B[(N + 1) : (2 * N)]
    X = np.linalg.solve(A_static, -B_static)  # positions for masses 2..N

    Z0 = np.zeros(2 * N)
    Z0[0] = 0.0           # y1 = 0 (top reference)
    Z0[1:N] = X           # y2..yN = X
    # velocities are zero initially

    X_CG0 = np.mean(np.concatenate(([0.0], X)))  # initial CG position

    # Simulate first phase until first collision
    ev = mass_collision_factory(c)
    sol = solve_ivp(
        fun=lambda t, z: model_springs(t, z, A, B),
        t_span=(time[0], time[-1]),
        y0=Z0,
        t_eval=None,
        events=ev,
        rtol=1e-8, atol=1e-10, max_step=dt
    )
    T_total = sol.t.copy()
    Z_total = sol.y.T.copy()

    # ---------- Subsequent phases (collisions among top masses) ----------
    for c in range(1, N - 1):  # up to N-2 inclusive
        A, B = create_model(k, m, N, c, g)

        z_end = Z_total[-1, :]
        # Conservation of momentum at collision:
        # MATLAB: v = (c*ZOUT(end,N+c) + ZOUT(end,N+c+1))/(c+1)
        # Python (0-based): vel indices N..2N-1, so N+(c-1) and N+c
        v_c = z_end[N + (c - 1)]     # velocity of the stuck block (mass c)
        v_next = z_end[N + c]        # velocity of next mass (c+1)
        v = (c * v_c + v_next) / (c + 1)

        # next phase starts from last state, unify velocities of (c+1) masses
        Z0 = z_end.copy()
        Z0[N : N + (c + 1)] = v  # velocities of masses 1..(c+1)

        ev = mass_collision_factory(c)
        sol = solve_ivp(
            fun=lambda t, z: model_springs(t, z, A, B),
            t_span=(0.0, time[-1]),
            y0=Z0,
            t_eval=None,
            events=ev,
            rtol=1e-8, atol=1e-10, max_step=dt
        )

        # Concatenate with time shift
        T_total = np.concatenate([T_total, T_total[-1] + sol.t])
        Z_total = np.vstack([Z_total, sol.y.T])

    # Extend simulation ~10% for fully collapsed slinky free-fall
    extra_T = 0.1 * T_total[-1]
    steps_extra = max(1, int(extra_T * fR))
    time_add = np.linspace(0.0, extra_T, steps_extra)

    # Fourth phase: collapsed slinky falling as a rigid mass
    c = N - 1
    z_last = Z_total[-1, :]
    v_c = z_last[N + (c - 1)]
    v_next = z_last[N + c - 0] if (N + c) < (2 * N) else v_c  # guard
    v_rigid = (c * v_c + v_next) / (c + 1)

    y1_0 = z_last[0]  # first element position at start of rigid fall
    Z_pos = y1_0 + v_rigid * time_add + 0.5 * g * time_add**2
    Z_vel = v_rigid + g * time_add
    Z_pos_all = np.tile(Z_pos[:, None], (1, N))
    Z_vel_all = np.tile(Z_vel[:, None], (1, N))
    Z_final_phase = np.hstack([Z_pos_all, Z_vel_all])

    T_total = np.concatenate([T_total, T_total[-1] + time_add])
    Z_total = np.vstack([Z_total, Z_final_phase])

    # Center of gravity motion (UAM)
    x_cg = X_CG0 + 0.5 * g * T_total**2
    v_cg = g * T_total

    # -----------------------------
    # Plots
    # -----------------------------
    # Positions
    fig1, ax1 = plt.subplots()
    ax1.grid(True)
    ax1.set_ylim(0.0, np.max(Z_total[:, 0]))
    ax1.invert_yaxis()  # y downwards
    ax1.plot(T_total, Z_total[:, :N])
    ax1.plot(T_total, x_cg, 'c')
    ax1.set_xlabel('time [s]')
    ax1.set_ylabel('position [m]')
    fig1.tight_layout()

    # Velocities
    fig2, ax2 = plt.subplots()
    ax2.grid(True)
    ax2.plot(T_total, Z_total[:, N:])  # velocities
    ax2.plot(T_total, v_cg, 'c')
    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('speed [m/s]')
    fig2.tight_layout()

    # -----------------------------
    # Animation (MP4 or GIF, safe path + fallback)
    # -----------------------------
    SHOW_LIVE = True             # True per vedere la finestra aggiornarsi live
    colors = cm.cool(np.linspace(0, 1, N))
    encode_fps = 30               # FPS del video in uscita

    # >>> Nuovo controllo velocità video:
    #    VIDEO_TIME_SCALE > 1 -> più lento (video più lungo)
    #    VIDEO_TIME_SCALE = 1 -> tempo reale
    #    VIDEO_TIME_SCALE < 1 -> più veloce
    VIDEO_TIME_SCALE = 20.0        # <-- regola qui per rallentare/accelerare il video
    n_frames = max(2, int(encode_fps * T_total[-1] * VIDEO_TIME_SCALE))
    frame_indices = np.linspace(0, len(T_total) - 1, n_frames).astype(int)

    # Percorso di salvataggio sicuro (macOS/Linux/Win): ~/Movies/slinky_drop.*
    save_dir = Path.home() / "Movies"
    save_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = save_dir / "slinky_drop.mp4"
    gif_path = save_dir / "slinky_drop.gif"

    fig = plt.figure(figsize=(12.8, 7.2))  # 1280x720

    def draw_frame(i):
        fig.clf()
        idx = frame_indices[i]

        # Subplot 1: posizioni vs tempo
        ax = fig.add_subplot(2, 2, 1)
        ax.grid(True)
        ax.set_ylim(0.0, float(np.max(Z_total[:, 0])))
        ax.invert_yaxis()
        for j in range(N):
            ax.plot(T_total, Z_total[:, j], color=colors[j], linewidth=2)
        ax.plot(T_total, x_cg, 'k')
        ax.plot([T_total[idx], T_total[idx]], [0, np.max(Z_total[:, 0])], 'k--')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position [m]')

        # Subplot 2: velocità vs tempo
        ax = fig.add_subplot(2, 2, 3)
        ax.grid(True)
        vmax = float(np.max(Z_total[:, N:]))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        ax.set_ylim(0.0, vmax)
        for j in range(N):
            ax.plot(T_total, Z_total[:, N + j], color=colors[j], linewidth=2)
        ax.plot(T_total, v_cg, 'k')
        ax.plot([T_total[idx], T_total[idx]], [0, ax.get_ylim()[1]], 'k--')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Speed [m/s]')

                # Subplot 3: geometria istantanea (TUTTE le molle e TUTTE le masse)
        ax = fig.add_subplot(2, 2, (2, 4))
        ax.grid(True)
        ymax = max(1e-6, 1.1 * float(np.max(Z_total[:, N - 1])))
        ax.set_ylim(0.0, ymax)     # 0 in alto, ymax in basso
        ax.invert_yaxis()          # così l’asse y parte dall’alto


        # Disegna tutte le molle (segmenti verticali)
        spring_handles = []
        for j in range(N - 1):
            h, = ax.plot([0, 0], [Z_total[idx, j], Z_total[idx, j + 1]],
                         'k:', linewidth=1.5)
            spring_handles.append(h)

        # Disegna tutte le masse con etichetta
        mass_handles = []
        mass_labels = []
        for j in range(N):
            (h,) = ax.plot(0, Z_total[idx, j], 'o',
                           color=colors[j], markerfacecolor=colors[j],
                           markersize=6, label=f"m{j+1}")
            mass_handles.append(h)
            mass_labels.append(f"m{j+1}")

        # Centro di massa
        (cgpt,) = ax.plot(0, x_cg[idx], 'ko', markersize=4, label="CG")

        # Legenda: tutte le molle (un'unica voce), tutte le masse, e CG
        ax.legend([spring_handles[0]] + mass_handles + [cgpt],
                  ["Spring"] + mass_labels + ["CG"],
                  loc="best", fontsize=8, ncol=2)

        ax.set_ylabel('Position [m]')
        ax.set_title(f"Time={T_total[idx]:.3f} s (Video scale={VIDEO_TIME_SCALE}x)")


    # Opzione live
    if SHOW_LIVE:
        for i in range(len(frame_indices)):
            draw_frame(i)
            plt.pause(0.001)

    # Salvataggio: MP4 se ffmpeg disponibile, altrimenti GIF (Pillow)
    saved = False
    try:
        if animation.writers.is_available('ffmpeg'):
            writer = FFMpegWriter(fps=encode_fps,
                                  metadata=dict(artist='Python Slinky'),
                                  bitrate=18000)
            with writer.saving(fig, str(mp4_path), dpi=100):
                for i in range(len(frame_indices)):
                    draw_frame(i)
                    writer.grab_frame()
            print(f"✅ Video salvato: {mp4_path}")
            saved = True
        else:
            print("ℹ️ ffmpeg non trovato, passo al GIF…")
    except Exception as e:
        print(f"⚠️ Errore con ffmpeg: {e}\nPasso al GIF…")

    if not saved:
        try:
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=15)
            with writer.saving(fig, str(gif_path), dpi=100):
                for i in range(len(frame_indices)):
                    draw_frame(i)
                    writer.grab_frame()
            print(f"✅ GIF salvata: {gif_path}")
            saved = True
        except Exception as e:
            print(f"❌ Errore anche con GIF: {e}")
            print("Suggerimenti: 1) installa ffmpeg (es. `brew install ffmpeg` oppure `conda install -c conda-forge ffmpeg`), "
                  "2) `pip install pillow`, 3) verifica permessi di scrittura in ~/Movies.")

    # (facoltativo) mostra i grafici statici
    # plt.show()


if __name__ == "__main__":
    main()
