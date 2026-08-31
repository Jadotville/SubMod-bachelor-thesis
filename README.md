# Subgruppensuche durch lokales Modelltraining

Dieses Repo enthält die **modifizierte `pysubgroup`-Bibliothek** aus der Arbeit und die Skripte, mit denen die fünf Experimente nachvollzogen werden können.

## Warum eine lokale `pysubgroup`?

Die Arbeit erweitert [pysubgroup](https://github.com/flemmerich/pysubgroup) (Apache 2.0).
Die zusätzlichen Module liegen nur in `./pysubgroup`, nicht auf PyPI:

- `pysubgroup.model_adaptability_target`
- `pysubgroup.parallel_model_adaptability_dfs`

`requirements.txt` installiert `pysubgroup` daher **nicht** von PyPI.

## Einrichtung (Python 3.10+)

Unter Debian/Ubuntu zuerst `python3-venv` (`sudo apt install python3.10-venv`
bzw. das Paket zur verwendeten Version).

```bash
git clone https://github.com/Jadotville/submod-bachelor-thesis.git
cd submod-bachelor-thesis

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -U pip "setuptools>=46.1.0,<81" "setuptools_scm[toml]>=5" wheel
pip install -r requirements.txt

# Lokale Bibliothek. Die Versionsvariable ist nötig, weil setuptools-scm
# in diesem Unterordner keine Version erkennen kann.
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYSUBGROUP=0.9.0 pip install -e ./pysubgroup
```

`setuptools<81`: Upstream-`pysubgroup` importiert noch `pkg_resources`.

```bash
python -c "import pysubgroup; print(pysubgroup.__file__)"
```

Der Pfad muss `pysubgroup/src/pysubgroup` enthalten. Zeigt er ins
`.venv/.../site-packages` ohne diesen Ordner: `pip uninstall pysubgroup`
und die editierbare Installation erneut.

`tabpfn` steht in `requirements.txt`. Für die GPU-Läufe **zuerst** ein
CUDA-PyTorch passend zur Maschine installieren (sonst zieht `pip` oft eine
CPU-Build), zum Beispiel CUDA 13.0:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

Der erste TabPFN-Fit lädt die Hugging-Face-Gewichte. Dafür einmal die Lizenz
unter https://ux.priorlabs.ai akzeptieren und `export TABPFN_TOKEN='...'`
setzen, oder den Schlüssel in `tabpfn_api_token.txt` im Repo-Root legen
(gitignoriert). Das ist kein API-Inferenzlauf.

Ergebnisse unter `results/` sind gitignoriert. Kernel für das Demo-Notebook:
**`.venv (submod-bachelor-thesis)`**.


## Demo-Notebook (`demo_method.ipynb`)

Kein Experiment der Arbeit, sondern ein kurzer End-to-End-Lauf der Methode aus Kapitel 3–4 über die **direkte `pysubgroup`-API**.

Inhalt:

1. **Synthetisch** — Generator `make_1_1_1` (Vorzeichenwechsel in `a`).
   Logistische Regression, Tiefe 1, ein Selektor. Zeigt `quality_raw`,
   `quality`, `quality_ga` und `interesting`.
2. **Real** — Datensatz **adult**, Tiefe 2, `ProcessModelAdaptabilityDFS`.
   Dafür muss `adult.csv` liegen (einmal Experiment 2, `download_data.py`).

Die Notebook-Zellen sind unabhängig von den fünf Experimentordnern.

## Die fünf Experimente

| # | Ordner | Was es tut | Braucht |
|---|--------|------------|---------|
| 1 | `experiments/synthetical_data/` | Generatoren A0–G1 | nur Setup |
| 2 | `experiments/experiments_real_world_data/` | 13 Datensätze, Offline-Suche (+ optional TabPFN) | Download |
| 3 | `experiments/characterization of the quality function/` | α-/β-Sweeps, Größe, Redundanz | 2, Offline |
| 4 | `experiments/comparison_subroc/` | φ nachträglich auf denselben Kandidaten | 2 |
| 5 | `experiments/performance improvement/` | Worker-Sweep (Threads/Prozesse vs. DFS) | 2, CSV |

### 1. Synthetische Daten

```bash
python experiments/synthetical_data/run.py --list
python experiments/synthetical_data/run.py --all
```
### 2. Reale Datensätze

Die dreizehn CSVs gehören nicht zum Repo. Einmal herunterladen
(OpenML über scikit-learn, ACS über `folktables`):

```bash
cd experiments/experiments_real_world_data
python download_data.py
```

**Offline-Suche** (lr / rf / lgbm / mlp, Tiefe 2, volle Hälften,
`results/offline/`):

```bash
bash run_batch.sh
```

**TabPFN** (eine CUDA-GPU, nicht die Prior-Labs-API).

```bash
cd experiments/experiments_real_world_data
bash run_tabpfn.sh
```

### 3. Charakterisierung der Qualitätsfunktion

```bash
python experiments/characterization\ of\ the\ quality\ function/run.py
```

### 4. Vergleich mit SubROC

```bash
python experiments/comparison_subroc/score_phi.py
```

### 5. Laufzeit / Worker-Sweep

**Offline** (lr / rf / lgbm / mlp; Worker 1 … 64):

```bash
cd experiments/performance\ improvement
bash run_worker_sweep.sh
```

**TabPFN** (lokal CUDA, Train/Test-Cap 1024, ein Estimator, Worker
1 2 4 8 16 — dasselbe Protokoll wie `run_tabpfn.sh`):

```bash
cd experiments/performance\ improvement
bash run_worker_sweep_tabpfn.sh
```

Figuren aus den Summary-CSVs:

```bash
python experiments/performance\ improvement/plot_process_speedup.py
```

