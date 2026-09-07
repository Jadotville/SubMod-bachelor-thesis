#!/usr/bin/env bash
#
# Uncapped TabPFN on the ten datasets that finished on one GPU.
# Same search as run_tabpfn.sh (depth 2, seed 42, α=0.7, β=0, GA, 50/50)
# except no train/test cap and CUDA-safe spawn workers.
#
# Omitted (too large for one GPU in a thesis time budget):
#   covertype, Diabetes130US, road-safety
# Pass them with --datasets if you still want to try.
#
#   bash run_tabpfn_uncapped.sh
#   bash run_tabpfn_uncapped.sh --foreground
#   bash run_tabpfn_uncapped.sh --datasets mushroom adult
#
# Results: results/tabpfn_uncapped/  (gitignored, like results/tabpfn/)
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
RESULTS_DIR="$HERE/results/tabpfn_uncapped"
SESSION="sd_tabpfn_uncapped"
FOREGROUND=0
DATASETS_OVERRIDE=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) FOREGROUND=1; shift ;;
    --datasets)
      shift
      while [[ $# -gt 0 && "$1" != --* ]]; do
        DATASETS_OVERRIDE+=("$1")
        shift
      done
      ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unbekanntes Flag: $1" >&2; exit 1 ;;
  esac
done

# Recorded protocol. 16 workers OOM on large ACS covers (one L40, 46 GB);
# those three cells used 8. Phishing was recorded with 8 threads; spawn-8
# is the reproducible path.
DATASETS_DEFAULT="mushroom PhishingWebsites default-of-credit-card-clients bank-marketing electricity adult ACSIncome ACSTravelTime ACSPublicCoverage ACSMobility"
OMITTED="covertype Diabetes130US road-safety"

workers_for() {
  case "$1" in
    ACSTravelTime|ACSPublicCoverage|ACSMobility|PhishingWebsites) echo 8 ;;
    *) echo 16 ;;
  esac
}

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

  source $PROJECT_ROOT/.venv/bin/activate
  pip install torch --index-url https://download.pytorch.org/whl/cu130
  pip install -r $PROJECT_ROOT/requirements.txt

Oder: TABPFN_PYTHON=/pfad/zu/python bash run_tabpfn_uncapped.sh
EOF
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/pysubgroup/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TABPFN_NO_BROWSER=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ -z "${TABPFN_TOKEN:-}" ]]; then
  for candidate in \
    "$HOME/.cache/tabpfn/auth_token" \
    "$HOME/.tabpfn/token" \
    "$PROJECT_ROOT/tabpfn_api_token.txt"
  do
    if [[ -f "$candidate" ]]; then
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
  2. export TABPFN_TOKEN='...'
     oder Datei $PROJECT_ROOT/tabpfn_api_token.txt (nicht committen)
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
umask 077
printf '%s' "$TABPFN_TOKEN" >"$RESULTS_DIR/.tabpfn_token"
chmod 600 "$RESULTS_DIR/.tabpfn_token"

cat >"$RESULTS_DIR/PROTOCOL.md" <<EOF
# TabPFN ungekappt (volle 50/50-Hälften)

Dasselbe Suchprotokoll wie \`run_tabpfn.sh\`, ohne Train/Test-Cap.
Worker starten mit \`spawn\` (\`TabPFNSpawnProcessDFS\`), weil Fork nach
einem GPU-Fit im Parent CUDA in den Kindern nicht neu initialisieren kann.

## Wiederholen

\`\`\`bash
cd experiments/experiments_real_world_data
python download_data.py
bash run_tabpfn_uncapped.sh
\`\`\`

Fertige Zellen werden übersprungen (\`--skip-existing\`).

## Datensätze

Zehn von dreizehn: mushroom, PhishingWebsites, default-of-credit-card-clients,
bank-marketing, electricity, adult, ACSIncome, ACSTravelTime,
ACSPublicCoverage, ACSMobility.

Nicht im Default, weil eine GPU die großen Cover nicht in vertretbarer
Zeit schafft: ${OMITTED}.
Gezielt versuchen: \`bash run_tabpfn_uncapped.sh --datasets covertype\`

## Feste Einstellungen

- Tiefe 2, Seed 42, α=0,7, β=0, Generalisierungsbewusstsein
- kein \`--train-cap\` / \`--test-cap\`
- \`n_estimators=1\`, \`device=cuda\`, \`ignore_pretraining_limits=True\`
- Algorithmus: spawn; 16 Worker, außer ACSTravelTime / ACSPublicCoverage /
  ACSMobility / PhishingWebsites (8 Worker)
- \`PYTHONHASHSEED=0\`, BLAS-Threads = 1

Siehe \`tabpfn_uncapped_summary.csv\` im Experimentordner (erwartete
Kennzahlen der hier abgelegten Läufe) und je Zelle \`meta.json\`.
EOF

LAUNCHER="$RESULTS_DIR/launch_${SESSION}.sh"
{
  echo '#!/usr/bin/env bash'
  echo 'set -euo pipefail'
  echo "export PYTHONPATH=$(printf '%q' "$PYTHONPATH")"
  echo 'export PYTHONHASHSEED=0'
  echo 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1'
  echo 'export TABPFN_NO_BROWSER=1'
  echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
  echo "HERE=$(printf '%q' "$HERE")"
  echo "PYTHON=$(printf '%q' "$PYTHON")"
  echo "RESULTS_DIR=$(printf '%q' "$RESULTS_DIR")"
  echo "if [[ -f \"\$RESULTS_DIR/.tabpfn_token\" ]]; then"
  echo "  export TABPFN_TOKEN=\"\$(tr -d '[:space:]' < \"\$RESULTS_DIR/.tabpfn_token\")\""
  echo 'fi'
  if [[ ${#DATASETS_OVERRIDE[@]} -gt 0 ]]; then
    echo "DATASETS=( $(printf '%q ' "${DATASETS_OVERRIDE[@]}") )"
  else
    echo "DATASETS=( $(printf '%q ' $DATASETS_DEFAULT) )"
  fi
  cat <<'EOF'
workers_for() {
  case "$1" in
    ACSTravelTime|ACSPublicCoverage|ACSMobility|PhishingWebsites) echo 8 ;;
    *) echo 16 ;;
  esac
}
failed=0
for ds in "${DATASETS[@]}"; do
  w="$(workers_for "$ds")"
  echo "===== START $ds workers=$w ====="
  if ! "$PYTHON" "$HERE/run.py" \
      --datasets "$ds" \
      --models tabpfn \
      --depth 2 \
      --algorithm spawn \
      --max-workers "$w" \
      --tabpfn-n-estimators 1 \
      --results-dir "$RESULTS_DIR" \
      --skip-existing; then
    echo "===== FAIL $ds ====="
    failed=1
  else
    echo "===== DONE $ds ====="
  fi
done
"$PYTHON" "$HERE/summarize_runs.py" --results-dir "$RESULTS_DIR" --csv || true
echo "===== QUEUE FINISHED failed=$failed ====="
exit "$failed"
EOF
} >"$LAUNCHER"
chmod +x "$LAUNCHER"

printf 'Python     : %s\n' "$PYTHON"
printf 'Ergebnisse : %s\n' "$RESULTS_DIR"
if [[ ${#DATASETS_OVERRIDE[@]} -gt 0 ]]; then
  printf 'Datensätze : %s\n' "${DATASETS_OVERRIDE[*]}"
else
  printf 'Datensätze : %s\n' "$DATASETS_DEFAULT"
  printf 'Weggelassen: %s\n' "$OMITTED"
fi

if [[ "$FOREGROUND" -eq 1 ]]; then
  exec "$LAUNCHER"
fi
tmux new-session -d -s "$SESSION" -c "$HERE" "exec bash -lc 'exec \"$LAUNCHER\" 2>&1 | tee -a \"$RESULTS_DIR/${SESSION}.log\"'"
echo "Gestartet: tmux attach -t $SESSION"
echo "Log:       tail -f $RESULTS_DIR/${SESSION}.log"
