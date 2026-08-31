#!/usr/bin/env bash
#
# Local TabPFN batch on the thirteen real-world datasets of chapter 4.
#
# Same search protocol as ``run_batch.sh`` (offline): depth 2, seed 42,
# size weight 0.7, generalization awareness, min_support = max(50, 0.02 n),
# stratified 50/50 split — except each half is capped (default 1024/1024)
# and TabPFN uses one ensemble member. Full 50k halves with 8 estimators
# do not finish overnight on one GPU. Results go to ``results/tabpfn/``, not
# ``results/offline/``. After the last cell, ``summarize_runs.py --csv``
# writes findings.csv, runtimes.csv and all_subgroups.csv there.
#
# TabPFN runs locally on one CUDA GPU (package ``tabpfn``, not the API client).
# The search is sequential (dfs, one worker) so the foundation model is not
# loaded several times on the same card.
#
#   bash run_tabpfn.sh                 # start in a detached tmux session
#   bash run_tabpfn.sh --foreground    # run in this terminal
#   bash run_tabpfn.sh --datasets mushroom   # extra args go to run.py
#
# Python: ``TABPFN_PYTHON`` if set, else ``.venv`` if it has CUDA TabPFN.
# A Korrektor installs CUDA PyTorch first, then ``pip install -r requirements.txt``.
#
# Restarting the same command resumes: --skip-existing skips finished cells.
#
# Watch:     tmux attach -t sd_tabpfn     (detach: Ctrl-b d)
# Follow:    tail -f results/tabpfn/run.log
# Stop:      tmux kill-session -t sd_tabpfn
#
# (`chmod +x run_tabpfn.sh` once if you prefer `./run_tabpfn.sh`.)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
RESULTS_DIR="$HERE/results/tabpfn"
SESSION="sd_tabpfn"
FOREGROUND=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) FOREGROUND=1; shift ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

CANDIDATES="covertype electricity Diabetes130US road-safety ACSPublicCoverage"
BASELINES="ACSIncome ACSMobility ACSTravelTime adult bank-marketing default-of-credit-card-clients"
CONTROLS="PhishingWebsites mushroom"
DATASETS="$CANDIDATES $BASELINES $CONTROLS"

has_local_tabpfn() {
  local py="$1"
  [[ -x "$py" ]] || return 1
  "$py" -c "import torch; from tabpfn import TabPFNClassifier; raise SystemExit(0 if torch.cuda.is_available() else 1)" \
    >/dev/null 2>&1
}

PYTHON=""
if [[ -n "${TABPFN_PYTHON:-}" ]]; then
  PYTHON="$TABPFN_PYTHON"
elif has_local_tabpfn "$PROJECT_ROOT/.venv/bin/python"; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  cat >&2 <<EOF
Kein Python mit lokalem TabPFN + CUDA gefunden.

Im Projekt-venv, nach einem CUDA-PyTorch passend zur Maschine:

  source $PROJECT_ROOT/.venv/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cu130
  pip install -r $PROJECT_ROOT/requirements.txt

Oder explizit:  TABPFN_PYTHON=/pfad/zu/python bash run_tabpfn.sh
EOF
  exit 1
fi

# Public pysubgroup of this repo, even if PYTHON comes from another venv.
export PYTHONPATH="$PROJECT_ROOT/pysubgroup/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# First fit downloads official weights; license is checked then, not per predict.
export TABPFN_NO_BROWSER=1

# License token for that download (not the old tabpfn_client inference API).
TOKEN_FILE=""
if [[ -n "${TABPFN_TOKEN:-}" ]]; then
  :
else
  for candidate in \
    "$HOME/.cache/tabpfn/auth_token" \
    "$HOME/.tabpfn/token" \
    "$PROJECT_ROOT/tabpfn_api_token.txt"
  do
    if [[ -f "$candidate" ]]; then
      TOKEN_FILE="$candidate"
      export TABPFN_TOKEN
      TABPFN_TOKEN="$(tr -d '[:space:]' < "$candidate")"
      echo "Lizenzschlüssel geladen (nur Gewichtsdownload): $candidate"
      break
    fi
  done
fi
if [[ -z "${TABPFN_TOKEN:-}" ]]; then
  cat >&2 <<EOF
TABPFN_TOKEN fehlt. Einmalig Lizenz akzeptieren und Schlüssel setzen:

  1. https://ux.priorlabs.ai  → Licenses akzeptieren
  2. Schlüssel unter Account kopieren
  3. export TABPFN_TOKEN='...'
     oder Datei $PROJECT_ROOT/tabpfn_api_token.txt (nicht committen)

Danach läuft die Suche lokal auf der GPU, ohne API-Inferenz.
EOF
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1 && [[ "$FOREGROUND" -eq 0 ]]; then
  echo "tmux ist nicht installiert. Mit --foreground im aktuellen Terminal starten." >&2
  exit 1
fi
if [[ "$FOREGROUND" -eq 0 ]] && tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' läuft bereits. Anhängen: tmux attach -t $SESSION" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
# tmux inherits the server environment, not this shell. Re-read from a 0600 file.
umask 077
printf '%s' "$TABPFN_TOKEN" >"$RESULTS_DIR/.tabpfn_token"
chmod 600 "$RESULTS_DIR/.tabpfn_token"
TOKEN_FILE="$RESULTS_DIR/.tabpfn_token"

if ! "$PYTHON" -c "
import sys
from pathlib import Path
import torch
import tabpfn
import pysubgroup
ok = Path(pysubgroup.__file__).resolve().as_posix().find('submod-bachelor-thesis') >= 0
print('python     :', sys.executable)
print('tabpfn     :', tabpfn.__version__)
print('torch      :', torch.__version__)
print('cuda       :', torch.cuda.get_device_name(0))
print('pysubgroup :', pysubgroup.__file__)
raise SystemExit(0 if ok and torch.cuda.is_available() else 1)
"; then
  echo "Preflight fehlgeschlagen (CUDA oder pysubgroup aus diesem Repo)." >&2
  exit 1
fi

"$PYTHON" - <<PY
import json, platform, sys
from pathlib import Path
import pysubgroup
import tabpfn
import torch

out = Path(r"$RESULTS_DIR") / "environment.json"
out.write_text(json.dumps({
    "recorded_at": __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat(),
    "host": platform.node(),
    "python": platform.python_version(),
    "executable": sys.executable,
    "pysubgroup": pysubgroup.__file__,
    "tabpfn": tabpfn.__version__,
    "torch": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "protocol": {
        "datasets": """$DATASETS""".split(),
        "models": ["tabpfn"],
        "depth": 2,
        "seed": 42,
        "size_weight": 0.7,
        "balance_weight": 0.0,
        "generalization_awareness": True,
        "algorithm": "dfs",
        "max_workers": 1,
        "tabpfn": {
            "backend": "local",
            "n_estimators": 1,
            "device": "cuda",
            "ignore_pretraining_limits": True,
            "memory_saving_mode": "auto",
            "train_cap": 1024,
            "test_cap": 1024,
        },
    },
}, indent=2) + "\n", encoding="utf-8")
print(f"geschrieben: {out}")
PY

"$PYTHON" -m pip freeze >"$RESULTS_DIR/pip_freeze.txt" 2>/dev/null || true

CMD=(
  "$PYTHON" "$HERE/run.py"
  --datasets $DATASETS
  --models tabpfn
  --depth 2
  --algorithm dfs
  --max-workers 1
  --train-cap 1024
  --test-cap 1024
  --tabpfn-n-estimators 1
  --results-dir "$RESULTS_DIR"
  --skip-existing
  "${EXTRA[@]+"${EXTRA[@]}"}"
)

LAUNCHER="$RESULTS_DIR/launch_${SESSION}.sh"
{
  echo '#!/usr/bin/env bash'
  echo "# Generiert von run_tabpfn.sh am $(date -Is). Direkt ausführbar."
  echo 'set -uo pipefail'
  echo
  echo "export PYTHONPATH=$(printf '%q' "$PYTHONPATH")"
  echo 'export PYTHONHASHSEED=0'
  echo 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1'
  echo 'export TABPFN_NO_BROWSER=1'
  echo "if [[ -f $(printf '%q' "$RESULTS_DIR/.tabpfn_token") ]]; then"
  echo "  export TABPFN_TOKEN=\"\$(tr -d '[:space:]' < $(printf '%q' "$RESULTS_DIR/.tabpfn_token"))\""
  echo 'fi'
  echo
  echo 'echo "Aufruf:"'
  printf 'printf "  %%s\\n" %q\n' "$(printf '%q ' "${CMD[@]}")"
  echo
  printf '%q ' "${CMD[@]}"
  printf '2>&1 | tee -a %q\n' "$RESULTS_DIR/${SESSION}.log"
  echo 'status=${PIPESTATUS[0]}'
  echo 'printf "\n=== beendet mit Status %s ===\n" "$status"'
  printf '%q %q --results-dir %q --csv || true\n' \
    "$PYTHON" "$HERE/summarize_runs.py" "$RESULTS_DIR"
  if [[ "$FOREGROUND" -eq 0 ]]; then
    echo 'printf "\nFenster bleibt offen. Schliessen: exit\n"'
    echo 'exec bash'
  fi
} >"$LAUNCHER"
chmod +x "$LAUNCHER"

cat >"$RESULTS_DIR/PROTOCOL.md" <<EOF
# TabPFN-Protokoll (reale Datensätze)

Dieses Verzeichnis gehört zum lokalen TabPFN-Lauf. Die Offline-Modelle
(lr / rf / lgbm / mlp) liegen unter \`results/offline/\`.

## Wiederholen

Voraussetzungen: die dreizehn CSVs aus \`download_data.py\`, CUDA-PyTorch,
\`tabpfn==8.0.4\`, lokale \`pysubgroup\` aus diesem Repo.

\`\`\`bash
cd experiments/experiments_real_world_data
bash run_tabpfn.sh
\`\`\`

Fertige Zellen werden übersprungen (\`--skip-existing\`). Derselbe Aufruf
nach einem Abbruch setzt also fort.

## Feste Einstellungen

- Datensätze: dieselben 13 wie \`run_batch.sh\` (Kandidaten, Baselines, Kontrollen)
- Modell: TabPFN lokal (\`tabpfn\`, nicht \`tabpfn_client\`)
- \`n_estimators=1\`, \`device=cuda\`, \`ignore_pretraining_limits=True\`
- Suche: Tiefe 2, Seed 42, α=0,7, β=0, Generalisierungsbewusstsein
- Nach dem 50/50-Split: höchstens 1024 Trainings- und 1024 Testzeilen
  (``--train-cap`` / ``--test-cap``). ``min_support`` bezieht sich auf diese
  gekappte Tabelle, nicht auf die vollen 100k.
- Algorithmus: dfs, ein Worker (eine GPU)
- \`PYTHONHASHSEED=0\`, BLAS-Threads = 1
- Einmalige Prior-Labs-Lizenz für den Gewichtsdownload (\`TABPFN_TOKEN\`).
  Die Suche selbst ist lokal.

Siehe \`environment.json\`, \`launch_${SESSION}.sh\` und je Zelle \`meta.json\`.
\`.tabpfn_token\` nicht weitergeben (nur lokal, gitignoriert).

## Auswertung

\`\`\`bash
python summarize_runs.py --results-dir results/tabpfn --csv
\`\`\`
EOF

printf 'Datensätze : %s\n' "$DATASETS"
printf 'Modell     : tabpfn (lokal, CUDA)\n'
printf 'Python     : %s\n' "$PYTHON"
printf 'Ergebnisse : %s\n' "$RESULTS_DIR"
printf 'Session    : %s\n\n' "$SESSION"

if [[ "$FOREGROUND" -eq 1 ]]; then
  echo "Starte im Vordergrund. Abbrechen: Ctrl-c"
  exec "$LAUNCHER"
fi

tmux new-session -d -s "$SESSION" -c "$HERE" "exec '$LAUNCHER'"

cat <<EOF
Gestartet. Nützliche Befehle:

  tmux attach -t $SESSION        # zuschauen (Ctrl-b d zum Loslösen)
  tail -f $RESULTS_DIR/run.log   # Log verfolgen
  tmux kill-session -t $SESSION  # abbrechen

Nach dem Lauf (das Skript macht das selbst):

  $PYTHON $HERE/summarize_runs.py --results-dir $RESULTS_DIR --csv
EOF
