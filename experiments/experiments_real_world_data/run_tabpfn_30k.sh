#!/usr/bin/env bash
#
# TabPFN on covertype, Diabetes130US, road-safety with 30k sampled rows
# (15k/15k halves). Same protocol as the uncapped batch, spawn workers.
#
#   bash run_tabpfn_30k.sh
#   bash run_tabpfn_30k.sh --foreground
#
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
RESULTS_DIR="$HERE/results/tabpfn_30k"
SESSION="sd_tabpfn_30k"
FOREGROUND=0
DATASETS="covertype Diabetes130US road-safety"
ROW_LIMIT=30000
WORKERS=16

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) FOREGROUND=1; shift ;;
    --max-workers) WORKERS="$2"; shift 2 ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unbekanntes Flag: $1" >&2; exit 1 ;;
  esac
done

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
elif has_local_tabpfn "$PROJECT_ROOT/../bachelor thesis/.venv_bachelor_thesis/bin/python"; then
  PYTHON="$PROJECT_ROOT/../bachelor thesis/.venv_bachelor_thesis/bin/python"
else
  echo "TABPFN_PYTHON setzen oder CUDA-TabPFN im Projekt-venv installieren." >&2
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
    "$PROJECT_ROOT/tabpfn_api_token.txt"
  do
    if [[ -f "$candidate" ]]; then
      export TABPFN_TOKEN
      TABPFN_TOKEN="$(tr -d '[:space:]' < "$candidate")"
      break
    fi
  done
fi
if [[ -z "${TABPFN_TOKEN:-}" ]]; then
  echo "TABPFN_TOKEN fehlt." >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
umask 077
printf '%s' "$TABPFN_TOKEN" >"$RESULTS_DIR/.tabpfn_token"
chmod 600 "$RESULTS_DIR/.tabpfn_token"

if [[ "$FOREGROUND" -eq 0 ]]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux fehlt. --foreground nutzen." >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' läuft." >&2
    exit 1
  fi
fi

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
  echo "ROW_LIMIT=$(printf '%q' "$ROW_LIMIT")"
  echo "MAX_WORKERS=$(printf '%q' "$WORKERS")"
  echo "DATASETS=( $(printf '%q ' $DATASETS) )"
  echo "if [[ -f \"\$RESULTS_DIR/.tabpfn_token\" ]]; then"
  echo "  export TABPFN_TOKEN=\"\$(tr -d '[:space:]' < \"\$RESULTS_DIR/.tabpfn_token\")\""
  echo 'fi'
  cat <<'EOF'
failed=0
for ds in "${DATASETS[@]}"; do
  w="$MAX_WORKERS"
  ok=0
  while [[ "$w" -ge 8 ]]; do
    echo "===== START $ds row_limit=$ROW_LIMIT workers=$w $(date -Iseconds) ====="
    if "$PYTHON" "$HERE/run.py" \
        --datasets "$ds" \
        --models tabpfn \
        --depth 2 \
        --algorithm spawn \
        --max-workers "$w" \
        --row-limit "$ROW_LIMIT" \
        --tabpfn-n-estimators 1 \
        --results-dir "$RESULTS_DIR" \
        --skip-existing; then
      echo "===== DONE $ds ====="
      ok=1
      break
    fi
    echo "===== FAIL $ds workers=$w ====="
    w=$((w / 2))
    if [[ "$w" -lt 8 ]]; then
      break
    fi
    echo "===== RETRY $ds workers=$w ====="
  done
  if [[ "$ok" -eq 0 ]]; then
    failed=1
  fi
done
"$PYTHON" "$HERE/summarize_runs.py" --results-dir "$RESULTS_DIR" --csv || true
echo "===== QUEUE FINISHED failed=$failed ====="
exit "$failed"
EOF
} >"$LAUNCHER"
chmod +x "$LAUNCHER"

printf 'Python     : %s\n' "$PYTHON"
printf 'Datensätze : %s (row_limit=%s, workers=%s)\n' "$DATASETS" "$ROW_LIMIT" "$WORKERS"
printf 'Ergebnisse : %s\n' "$RESULTS_DIR"

if [[ "$FOREGROUND" -eq 1 ]]; then
  exec "$LAUNCHER"
fi
tmux new-session -d -s "$SESSION" -c "$HERE" "exec bash -lc 'exec \"$LAUNCHER\" 2>&1 | tee -a \"$RESULTS_DIR/${SESSION}.log\"'"
echo "Gestartet: tmux attach -t $SESSION"
echo "Log:       tail -f $RESULTS_DIR/${SESSION}.log"
