#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Porting in Python (file singolo) del codice MATLAB:
- script.m
- pendolo.m
- fitnonlin.m
- fitnonlin_ev.m
- my.m
- vetIsto.m

Dipendenze: numpy, scipy, matplotlib
    pip install numpy scipy matplotlib

Note:
- Ho mantenuto la logica e (dove presente) anche alcune scelte "strane" del codice MATLAB
  per essere il più fedele possibile (es. deltaT2 = 2*deltaT come in pendolo.m).
- Se vuoi correggere la propagazione dell'errore su T^2 (che fisicamente sarebbe ~ 2*T*deltaT),
  dimmelo e ti preparo una variante "fisicamente corretta".
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2_dist


# -----------------------------------------------------------------------------
# MODELLO (my.m): y = a*x + b con jacobiana rispetto ai parametri e derivata y'(x)
# -----------------------------------------------------------------------------
def my_model(p: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Equivalente di MATLAB:
        function [y,dydp,dydx]=my(p,x)
            y=p(1).*x+p(2);
            dydp(:,1)=x;
            dydp(:,2)=1;
            dydx=p(1);
        end

    Ritorna:
        y     : (m,)
        dydp  : (m, npar) jacobiana dy/dp
        dydx  : (m,) derivata dy/dx
    """
    p = np.asarray(p, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    y = p[0] * x + p[1]
    dydp = np.column_stack([x, np.ones_like(x)])
    dydx = np.full_like(x, p[0], dtype=float)
    return y, dydp, dydx


# -----------------------------------------------------------------------------
# FIT NON LINEARE (fitnonlin.m) con errori su y
# -----------------------------------------------------------------------------
@dataclass
class FitResult:
    p: np.ndarray          # parametri best-fit
    dp: np.ndarray         # errori stimati (come in MATLAB: moltiplicati per chi^2_red)
    chi2_red: float        # chi-quadro ridotto
    prob: float            # 1 - CDF(chi2; dof)
    corr: np.ndarray       # matrice correlazione


def fitnonlin(
    myfun: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pin: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dy: Optional[np.ndarray] = None,
) -> FitResult:
    """
    Porting di fitnonlin.m:
        [pout, dp, chiqnor, prob, corr] = fitnonlin(@myfun, pin, x, y, dy)

    dy se None -> ones
    """
    pin = np.asarray(pin, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    n = pin.size
    m = x.size
    if n > m:
        raise ValueError("FIT IMPOSSIBILE: num parametri > num dati")

    nlib = m - n  # gradi di liberta'

    if dy is None:
        dy = np.ones_like(x, dtype=float)
    else:
        dy = np.abs(np.asarray(dy, dtype=float).ravel())
        if dy.size != m:
            raise ValueError("dy deve avere la stessa lunghezza di x")

    def residuals_and_jac(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        yc, dydp, _dydx = myfun(p, x)  # yc:(m,), dydp:(m,n)
        res = (yc - y) / dy
        jac = dydp / dy[:, None]       # (m,n)
        return res, jac

    def fun(p: np.ndarray) -> np.ndarray:
        res, _ = residuals_and_jac(p)
        return res

    def jac(p: np.ndarray) -> np.ndarray:
        _, J = residuals_and_jac(p)
        return J

    # options analoghe a MATLAB (limiti più alti)
    lsq = least_squares(
        fun, pin, jac=jac, method="trf",
        max_nfev=50000, ftol=1e-5, xtol=1e-6, gtol=1e-8
    )

    if not lsq.success:
        # per fedeltà al MATLAB, restituiamo zeri
        return FitResult(
            p=np.zeros_like(pin),
            dp=np.zeros_like(pin),
            chi2_red=0.0,
            prob=0.0,
            corr=np.zeros((n, n), dtype=float),
        )

    # In MATLAB: chiqmin è la somma dei residui^2.
    chi2_min = float(np.sum(lsq.fun ** 2))
    chi2_red = chi2_min / nlib
    prob = float(1.0 - chi2_dist.cdf(chi2_min, nlib))

    # Covarianza come in MATLAB: inv(J^T J)
    J = lsq.jac
    alfa = J.T @ J
    try:
        cova = np.linalg.inv(alfa)
    except np.linalg.LinAlgError:
        cova = np.linalg.pinv(alfa)

    dp = np.sqrt(np.diag(cova))

    # matrice correlazioni
    corr = cova.copy()
    for i in range(n):
        for j in range(i, n):
            denom = dp[i] * dp[j] if dp[i] * dp[j] != 0 else np.nan
            corr[i, j] = corr[i, j] / denom
            corr[j, i] = corr[i, j]

    # correzione errori (fedelmente al MATLAB: dp = chi2_red * dp)
    dp = chi2_red * dp

    return FitResult(p=lsq.x.reshape(-1), dp=dp.reshape(-1), chi2_red=chi2_red, prob=prob, corr=corr)


# -----------------------------------------------------------------------------
# FIT CON ERRORI SU x e y: Metodo della varianza effettiva (fitnonlin_ev.m)
# -----------------------------------------------------------------------------
def fitnonlin_ev(
    myfun: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    pin: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    nstep: int = 6,
) -> FitResult:
    """
    Porting di fitnonlin_ev.m:
        [pout, dp, chiqnor, prob, corr] = fitnonlin_ev(@myfun, pin, x, y, dx, dy)

    Implementa l'iterazione su varianza effettiva:
        effvar = dy^2 + (dydx^2)*dx^2
        fit su y con sigma = sqrt(effvar)
        aggiornamento xf (non indispensabile ai parametri, ma lo manteniamo)
    """
    pin = np.asarray(pin, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    dx = np.asarray(dx, dtype=float).ravel()
    dy = np.asarray(dy, dtype=float).ravel()

    if not (x.size == y.size == dx.size == dy.size):
        raise ValueError("x, y, dx, dy devono avere la stessa lunghezza")

    piter = pin.copy()
    out = None

    for _k in range(nstep):
        f, _dfdp, derf = myfun(piter, x)   # derf = dy/dx
        effvar = dy**2 + (derf**2) * (dx**2)

        out = fitnonlin(myfun, piter, x, y, np.sqrt(effvar))
        piter = out.p

        # Aggiornamento "xf" come in MATLAB (non usato direttamente nel fit successivo,
        # ma lo manteniamo per fedeltà)
        xf = x + ((dx**2) / effvar) * (y - f) * derf
        yf, _dydp2, _dydx2 = myfun(piter, xf)
        _S = np.sum(((x - xf)**2) / (dx**2) + ((y - yf)**2) / (dy**2))
        # (S è diagnostico; nel MATLAB è commentato)

    assert out is not None
    return out


# -----------------------------------------------------------------------------
# ISTOGRAMMA: vetIsto.m
# -----------------------------------------------------------------------------
def vetIsto(vet: np.ndarray, bin: int, width: float) -> Tuple[np.ndarray, float]:
    """
    Porting di vetIsto.m:
        [coord,larg]=vetIsto(vet,bin,width)

    Se bin==0 usa BinWidth=width; altrimenti usa numero di bin = bin.
    Ritorna:
        coord: array shape (2, nbins) con [centri; conteggi]
        larg : larghezza bin
    """
    vet = np.asarray(vet, dtype=float).ravel()

    if bin == 0:
        if width <= 0:
            raise ValueError("width deve essere > 0 quando bin==0")
        vmin, vmax = float(np.min(vet)), float(np.max(vet))
        edges = np.arange(vmin, vmax + width, width)
        if edges.size < 2:
            edges = np.array([vmin, vmin + width], dtype=float)
        counts, edges = np.histogram(vet, bins=edges)
        larg = float(width)
    else:
        counts, edges = np.histogram(vet, bins=int(bin))
        larg = float(edges[1] - edges[0])

    centers = 0.5 * (edges[:-1] + edges[1:])
    coord = np.vstack([centers, counts])
    return coord, larg


# -----------------------------------------------------------------------------
# SIMULAZIONE PENDOLO: pendolo.m
# -----------------------------------------------------------------------------
def pendolo(
    campione: int,
    periodi: int,
    dL: float,
    dT: float,
    incr: float,
    peri: float,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    """
    Porting di pendolo.m:
        [chi,media,dev,g,dg]=pendolo(campione,periodi,dL,dT,incr,peri)

    Unità: cm e s. g_true = 981.7 cm/s^2
    """
    rng = np.random.default_rng(seed)

    g_true = 981.7
    g = np.zeros(campione, dtype=float)
    dg = np.zeros(campione, dtype=float)
    chi = np.zeros(campione, dtype=float)

    L0 = np.arange(50.0, 100.0 + 1e-12, float(incr))

    for j in range(campione):
        L = L0.copy()
        deltaL = float(dL) * np.ones_like(L)

        L = L + 0.1 * (1.0 - 2.0 * rng.random(L.size))

        T = np.zeros_like(L)
        deltaT = np.zeros_like(L)

        for i in range(L.size):
            noise = rng.normal(loc=0.0, scale=1.0, size=periodi)

            base = 2.0 * np.pi * np.sqrt(L[i] / g_true)
            corr = 1.0 + (np.pi / 4.0) ** 2 / 16.0
            Tsim = base * corr + noise / float(peri)

            T[i] = float(np.mean(Tsim))
            deltaT[i] = float(np.sqrt((np.std(Tsim, ddof=1) / np.sqrt(periodi)) ** 2 + dT**2))

        deltaT2 = 2.0 * deltaT
        T2 = T**2

        if dL == 0:
            out = fitnonlin(my_model, np.array([0.05, 0.0]), L, T2, deltaT2)
        else:
            out = fitnonlin_ev(my_model, np.array([0.05, 0.0]), L, T2, deltaL, deltaT2)

        chi[j] = out.chi2_red
        a = out.p[0]
        da = out.dp[0]
        g[j] = 4.0 * np.pi**2 / a
        dg[j] = g[j] * da / a

    media = float(np.mean(chi))
    dev = float(np.std(chi, ddof=1) / np.sqrt(campione))

    return chi, media, dev, g, dg


# -----------------------------------------------------------------------------
# "script.m" -> demo opzionale (grafici)
# -----------------------------------------------------------------------------
def demo_script(seed: int = 12345) -> None:
    """
    Replica fedele di script.m:
    - Figura 1: 4 pannelli (CASO 1..4)
    - Figura 2: CAMPIONE MISTO
    - Figura 3: dipendenza dal numero di periodi misurati (come in Periodi.png)
    """
    import matplotlib.pyplot as plt

    # -------------------------
    # CASI base (come script.m)
    # -------------------------
    chi, mean, std, *_ = pendolo(10000, 30, 0.0, 0.02, 5, 10, seed=seed)          # 11 lunghezze, no errore su L
    ideale, _ = vetIsto(chi, 60, 0.0)

    chiL2, meanL2, stdL2, *_ = pendolo(10000, 30, 0.1, 0.02, 25, 10, seed=seed+1)  # 3 lunghezze
    idealeL2, _ = vetIsto(chiL2, 60, 0.0)

    chiL3, meanL3, stdL3, *_ = pendolo(10000, 30, 0.1, 0.02, 10, 10, seed=seed+2)  # 6 lunghezze
    idealeL3, _ = vetIsto(chiL3, 60, 0.0)

    chibis, meanbis, stdbis, g, dg = pendolo(10000, 30, 0.1, 0.02, 5, 10, seed=seed+3)  # errori su L e T
    Idealedx, _ = vetIsto(chibis, 60, 0.0)

    chibis2, meanbis2, stdbis2, *_ = pendolo(10000, 30, 0.1, 0.2, 5, 10, seed=seed+4)  # sovrastima su T
    Idedx2, _ = vetIsto(chibis2, 60, 0.0)

    chibis3, meanbis3, stdbis3, *_ = pendolo(10000, 30, 0.1, 0.002, 5, 10, seed=seed+5)  # sottostima su T
    Idedx3, _ = vetIsto(chibis3, 60, 0.0)

    chibau, meanbau, stdbau, *_ = pendolo(10000, 30, 0.5, 0.2, 5, 10, seed=seed+6)  # sovrastima su tutto
    bau, _ = vetIsto(chibau, 0, 60)  # binwidth=60 (come MATLAB)

    chiari, meanari, stdari, *_ = pendolo(10000, 30, 0.1, 0.02, 5, 9.5, seed=seed+7)   # aritmia: 9.5 periodi
    arit, _ = vetIsto(chiari, 60, 0.0)

    chiari2, meanari2, stdari2, *_ = pendolo(10000, 30, 0.1, 0.02, 5, 10.5, seed=seed+8)  # aritmia: 10.5 periodi
    arit2, _ = vetIsto(chiari2, 60, 0.0)

    # -------------------------
    # FIGURA 1 (2x2) - CASO 1..4
    # -------------------------
    fig1 = plt.figure(figsize=(13, 9))
    fig1.suptitle("Valore atteso per la media χ²: 1 ± 0.47   (9 gradi di libertà)", y=0.98)

    ax1 = fig1.add_subplot(2, 2, 1)
    ax1.plot(ideale[0], ideale[1] / np.max(ideale[1]), "k--",
             idealeL2[0], idealeL2[1] / np.max(idealeL2[1]), "r--",
             idealeL3[0], idealeL3[1] / np.max(idealeL3[1]), "b--")
    ax1.set_xlim(0, 6)
    ax1.set_title("CASO 1: Diversi gradi di libertà")
    ax1.legend([
        f"9 gradi di libertà - Mean {mean:.3f} ± {std:.3f}",
        f"1 grado di libertà - Mean {meanL2:.3f} ± {stdL2:.3f}",
        f"4 gradi di libertà - Mean {meanL3:.3f} ± {stdL3:.3f}",
    ], fontsize=9)

    ax2 = fig1.add_subplot(2, 2, 2)
    ax2.plot(ideale[0], ideale[1] / np.max(ideale[1]), "k--",
             Idealedx[0], Idealedx[1] / np.max(Idealedx[1]), "r--",
             Idedx2[0], Idedx2[1] / np.max(Idedx2[1]), "b--",
             bau[0], bau[1] / np.max(bau[1]), "g--",
             Idedx3[0], Idedx3[1] / np.max(Idedx3[1]), "m--")
    ax2.set_xlim(-0.5, 7)
    ax2.set_title("CASO 2: DIFFERENTI ERRORI")
    ax2.legend([
        f"Errori solo su T - Mean {mean:.3f} ± {std:.3f}",
        f"Errori su L e T - Mean {meanbis:.3f} ± {stdbis:.3f}",
        f"Sovrastima errori su T - Mean {meanbis2:.4f} ± {stdbis2:.4f}",
        f"Sovrastima errori su T e L - Mean {meanbau:.4f} ± {stdbau:.4f}",
        f"Sottostima errori su T - Mean {meanbis3:.3f} ± {stdbis3:.3f}",
    ], fontsize=8)

    ax3 = fig1.add_subplot(2, 2, 3)
    ax3.plot(ideale[0], ideale[1] / np.max(ideale[1]), "k--",
             arit[0], arit[1] / np.max(arit[1]), "r--",
             arit2[0], arit2[1] / np.max(arit2[1]), "b--")
    ax3.set_xlim(-1, 6)
    ax3.set_title("CASO 3: ARITMIA")
    ax3.legend([
        f"Errori solo su T - Mean {mean:.3f} ± {std:.3f}",
        f"Misura di 9.5 periodi anziché 10 - Mean {meanari:.3f} ± {stdari:.3f}",
        f"Misura di 10.5 periodi anziché 10 - Mean {meanari2:.3f} ± {stdari2:.3f}",
    ], fontsize=8)

    ax4 = fig1.add_subplot(2, 2, 4)
    diff = 981.7 - g
    ax4.hist(diff, bins=40)
    ax4.set_title(r"CASO 4: $\theta=\pi/4$ in piccole oscillazioni → histogram(g ottenuta - 981.7)")
    acc = float(np.mean(g))
    dacc = float(np.std(g, ddof=1) / np.sqrt(diff.size))
    ax4.text(0.72, 0.82, f"Mean {acc:.2e}\nStd  {dacc:.2e}",
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle="square", facecolor="white", alpha=0.9))

    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    # -------------------------
    # FIGURA 2: CAMPIONE MISTO (come script.m)
    # -------------------------
    ch_i, *_ = pendolo(6000, 30, 0.1, 0.02, 5, 10, seed=seed+20)   # 6000 ideali con errori su L e T (dL=0.1)
    ch_sovT, *_ = pendolo(1000, 30, 0.1, 0.2, 5, 10, seed=seed+21) # 1000 sovrastima su T
    ch_sotT, *_ = pendolo(1000, 30, 0.1, 0.002, 5, 10, seed=seed+22)# 1000 sottostima su T
    ch_sovAll, *_ = pendolo(1000, 30, 0.5, 0.2, 5, 10, seed=seed+23)# 1000 sovrastima su tutto
    ch_ar, *_ = pendolo(1000, 30, 0.1, 0.02, 5, 9.5, seed=seed+24) # 1000 aritmia

    chitot = np.concatenate([ch_i, ch_sovT, ch_sotT, ch_sovAll, ch_ar])

    fig2 = plt.figure(figsize=(12, 7))
    ax = fig2.add_subplot(1, 1, 1)
    ax.hist(chitot, bins=60)
    ax.set_title("CAMPIONE MISTO")
    ax.set_xlim(0, 12)

    fine = float(np.mean(chitot))
    dfine = float(np.std(chitot, ddof=1) / np.sqrt(chitot.size))
    ax.text(0.60, 0.70,
            f"Mean {fine:.3e} std {dfine:.2e}\n"
            f"6000 IDEALI\n"
            f"1000 SOVRASTIMA SU T\n"
            f"1000 SOTTOSTIMA SU T\n"
            f"1000 SOVRASTIMA SU TUTTO\n"
            f"1000 ARITMIA",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="square", facecolor="white", alpha=0.9))

    fig2.tight_layout()

    # -------------------------
    # FIGURA 3: Periodi per ogni lunghezza (ispirata al tuo Periodi.png)
    # -------------------------
    periodi_list = [4, 8, 12, 18, 25, 30]
    fig3 = plt.figure(figsize=(13, 8))
    fig3.suptitle("Errori nulli sulle lunghezze, chi-quadro normalizzato atteso: 1 ± 0.47   (11 - 2 = 9 gradi di libertà)",
                  y=0.98)

    for idx, per in enumerate(periodi_list, start=1):
        axp = fig3.add_subplot(2, 3, idx)
        chi_p, mean_p, std_p, *_ = pendolo(1000, per, 0.0, 0.02, 5, 10, seed=seed+100+per)
        axp.hist(chi_p, bins=25)
        axp.set_title(f"{per} Periodi per ogni lunghezza")
        axp.text(0.70, 0.80, f"mean   {mean_p:.4f}\n dev.std. {std_p:.4f}",
                 transform=axp.transAxes, fontsize=9,
                 bbox=dict(boxstyle="square", facecolor="white", alpha=0.9))

    fig3.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()


if __name__ == "__main__":
    demo_script()
