# Python port dei moduli MATLAB (pendolo / chi^2)

Questa cartella contiene una traduzione 1:1 (per quanto sensato) dei file MATLAB
che mi hai caricato:

- `gennumgauss.m`  -> `gennumgauss.py`
- `vettoregauss.m` -> `vettoregauss.py`
- `erpen.m`        -> `erpen.py`
- `periodo_pendolo.m` -> `periodo_pendolo.py`
- `pendolo.m`      -> `pendolo.py`
- `mtit.m`         -> `mtit.py` (analogo matplotlib di `mtit` via `suptitle`)
- `histpenderr.m`  -> `histpenderr.py`
- `histchiperf.m`  -> `histchiperf.py`

## Dipendenze

- `numpy`
- `matplotlib`
- `scipy` (solo per `histchiperf.py`, per lo smoothing spline)

## Esecuzione

Da una directory che contiene `python_pendolo/`:

```bash
python -m python_pendolo.histpenderr
python -m python_pendolo.histchiperf
```

Oppure importando:

```python
import numpy as np
from python_pendolo import pendolo

Xs1 = np.arange(50, 101, 5)
MC2, MC2r, CQM, CQMr, SDC2, SDC2r = pendolo(Xs1, Theta0_deg=5, nmis=10, ntavoli=1000, dx=0.1)
```

## Nota sullo smoothing spline

MATLAB usa `fit(..., 'smoothingspline', 'SmoothingParam', p)` con un significato
specifico di `p`. In Python ho usato una mappatura euristica verso il parametro
`s` di `scipy.interpolate.UnivariateSpline`. Se vuoi una sovrapposizione più
fedele, posso implementare una conversione più rigorosa (ma serve fissare
precisamente: numero di bin, normalizzazione dei conteggi, criterio di smoothing).
