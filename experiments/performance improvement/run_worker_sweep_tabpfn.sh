#!/usr/bin/env bash
#
# Local TabPFN worker sweep (CUDA, package tabpfn). Phishing Websites, depth 2,
# train/test cap 1024, one estimator — same protocol as run_tabpfn.sh.
# Sequential DFS once, then threads and processes at 1 2 4 8 16 workers.
#
#   bash run_worker_sweep_tabpfn.sh --estimate-only
#   bash run_worker_sweep_tabpfn.sh
#   bash run_worker_sweep_tabpfn.sh --foreground
#
# Watch:     tmux attach -t worker_sweep_tabpfn
# Log:       tail -f results/worker_sweep_tabpfn.log
# Stop:      tmux kill-session -t worker_sweep_tabpfn

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
RW="$PROJECT_ROOT/experiments/experiments_real_world_data"
RESULTS_DIR="$HERE/results"
SESSION="worker_sweep_tabpfn"
FOREGROUND=0
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --foreground) FOREGROUND=1; shift ;;
    --tag)     SESSION="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
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
else
  echo "Kein Python mit lokalem TabPFN + CUDA. CUDA-PyTorch, dann requirements.txt." >&2
  echo "Oder: TABPFN_PYTHON=/pfad/zu/python bash $0" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/pysubgroup/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TABPFN_NO_BROWSER=1

if [[ -z "${TABPFN_TOKEN:-}" ]]; then
  for candidate in \
    "$HOME/.cache/tabpfn/auth_token" \
    "$HOME/.tabpfn/token" \
    "$PROJECT_ROOT/tabpfn_api_token.txt"
  do
    if [[ -f "$candidate" ]]; then
      TABPFN_TOKEN="$(tr -d '[:space:]' < "$candidate")"
      export TABPFN_TOKEN
      break
    fi
  done
fi

if [[ " ${EXTRA[*]-} " == *" --estimate-only "* ]]; then
  exec "$PYTHON" "$HERE/worker_sweep_tabpfn.py" "${EXTRA[@]}"
fi

if ! command -v tmux >/dev/null 2>&1 && [[ "$FOREGROUND" -eq 0 ]]; then
  echo "tmux fehlt. Mit --foreground starten." >&2
  exit 1
fi
if [[ "$FOREGROUND" -eq 0 ]] && tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' läuft. tmux attach -t $SESSION" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"
TOKEN_FILE="$RESULTS_DIR/.tabpfn_token"
if [[ -n "${TABPFN_TOKEN:-}" ]]; then
  umask 077
  printf '%s' "$TABPFN_TOKEN" >"$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
fi

CMD=(
  "$PYTHON" "$HERE/worker_sweep_tabpfn.py"
  --skip-existing
  --workers 1 2 4 8 16
  --depth 2
  --train-cap 1024
  --test-cap 1024
  --tabpfn-n-estimators 1
  "${EXTRA[@]+"${EXTRA[@]}"}"
)

LAUNCHER="$RESULTS_DIR/launch_${SESSION}.sh"
{
  echo '#!/usr/bin/env bash'
  echo "# Generiert von run_worker_sweep_tabpfn.sh am $(date -Is)."
  echo 'set -uo pipefail'
  echo
  echo "export PYTHONPATH=$(printf '%q' "$PYTHONPATH")"
  echo 'export PYTHONHASHSEED=0'
  echo 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1'
  echo 'export TABPFN_NO_BROWSER=1'
  echo "if [[ -f $(printf '%q' "$TOKEN_FILE") ]]; then"
  echo "  export TABPFN_TOKEN=\"\$(tr -d '[:space:]' < $(printf '%q' "$TOKEN_FILE"))\""
  echo 'fi'
  echo
  echo 'echo "Aufruf:"'
  printf 'printf "  %%s\\n" %q\n' "$(printf '%q ' "${CMD[@]}")"
  echo
  printf '%q ' "${CMD[@]}"
  printf '2>&1 | tee -a %q\n' "$RESULTS_DIR/${SESSION}.log"
  echo 'status=${PIPESTATUS[0]}'
  echo 'printf "\n=== beendet mit Status %s ===\n" "$status"'
  if [[ "$FOREGROUND" -eq 0 ]]; then
    echo 'printf "\nFenster bleibt offen. Schliessen: exit\n"'
    echo 'exec bash'
  fi
} >"$LAUNCHER"
chmod +x "$LAUNCHER"

printf 'Python     : %s\n' "$PYTHON"
printf 'Ergebnisse : %s\n' "$RESULTS_DIR"
printf 'Session    : %s\n\n' "$SESSION"

if [[ "$FOREGROUND" -eq 1 ]]; then
  exec "$LAUNCHER"
fi

tmux new-session -d -s "$SESSION" -c "$HERE" "exec '$LAUNCHER'"

cat <<EOF
Gestartet.

  tmux attach -t $SESSION
  tail -f $RESULTS_DIR/${SESSION}.log
  tmux kill-session -t $SESSION
EOF
