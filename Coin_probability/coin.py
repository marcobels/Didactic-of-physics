import time
import numpy as np
import matplotlib.pyplot as plt
from math import comb
from typing import Optional


def binomial_pmf(n: int, p: float) -> tuple[np.ndarray, np.ndarray]:
    """PMF Binomiale: X=0..n, P(X=k) = C(n,k) p^k (1-p)^(n-k)."""
    x = np.arange(n + 1, dtype=int)
    y = np.array([comb(n, k) * (p ** k) * ((1 - p) ** (n - k)) for k in x], dtype=float)
    return x, y


def coin1(ncoins: int, nprove: int, p: float = 0.5, seed: Optional[int] = None):
    """
    Simula nprove prove di lancio simultaneo di ncoins monete.
    Restituisce:
      - teste: array (nprove,) numero di teste per prova
      - croci: array (nprove,) numero di croci per prova
      - X, Y: supporto e PMF della binomiale teorica Bin(ncoins, p)
    """
    rng = np.random.default_rng(seed)

    # True = testa, False = croce, con probabilità p di testa
    tosses = rng.random((nprove, ncoins)) < p

    teste = tosses.sum(axis=1).astype(int)
    croci = (ncoins - teste).astype(int)

    X, Y = binomial_pmf(ncoins, p)
    return teste, croci, X, Y


def main():
    t0 = time.perf_counter()

    teste1, croci1, X1, Y1 = coin1(10, 10, 0.5, seed=123)
    teste2, croci2, X2, Y2 = coin1(10, 100, 0.5, seed=123)
    teste3, croci3, X3, Y3 = coin1(10, 1000, 0.5, seed=123)
    teste4, croci4, X4, Y4 = coin1(10, 10000, 0.5, seed=123)

    t1 = time.perf_counter()
    print(f"Elapsed: {t1 - t0:.4f} s")

    # 2x2 histogram come in MATLAB
    plt.figure(figsize=(10, 7))

    plt.subplot(2, 2, 1)
    plt.hist(teste1, bins=np.arange(-0.5, 10.6, 1), edgecolor="black")
    plt.title("Istogramma del numero di teste per 10 lanci di 10 monete")

    plt.subplot(2, 2, 2)
    plt.hist(teste2, bins=np.arange(-0.5, 10.6, 1), edgecolor="black")
    plt.title("Istogramma del numero di teste per 100 lanci di 10 monete")

    plt.subplot(2, 2, 3)
    plt.hist(teste3, bins=np.arange(-0.5, 10.6, 1), edgecolor="black")
    plt.title("Istogramma del numero di teste per 1000 lanci di 10 monete")

    plt.subplot(2, 2, 4)
    plt.hist(teste4, bins=np.arange(-0.5, 10.6, 1), edgecolor="black")
    plt.title("Istogramma del numero di teste per 10000 lanci di 10 monete")

    plt.tight_layout()

    # figura 2: binomiale teorica
    plt.figure()
    plt.plot(X1, Y1, marker="o")
    plt.title("Distribuzione binomiale")
    plt.xlabel("Numero di teste")
    plt.ylabel("Probabilità")

    plt.show()


if __name__ == "__main__":
    main()
