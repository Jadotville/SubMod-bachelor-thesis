#!/usr/bin/env bash
#
# Worker-Sweep (Threads vs. Prozesse vs. sequenzielles DFS) in einer
# abgekoppelten tmux-Session. Überlebt SSH-Disconnect; --skip-existing setzt
# nach einem Abbruch an der nächsten fehlenden Zelle fort.
#
#   bash run_worker_sweep.sh
#   bash run_worker_sweep.sh --quick
#   bash run_worker_sweep.sh --models lr lgbm --repeats 2
#
# Watch:     tmux attach -t worker_sweep     (Loslösen: Ctrl-b d)
# Log:       tail -f results/worker_sweep.log
# Stop:      tmux kill-session -t worker_sweep
#
# Default: PhishingWebsites, Tiefe 2, voller Suchraum, Modelle lr rf lgbm mlp,
# Worker 1 2 3 4 6 8 12 16 24 32 48 64. Sequenzielle DFS einmal je Modell als
# Baseline; Threads und Prozesse bei jeder Worker-Zahl.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

TAG="worker_sweep"
SESSION="worker_sweep"
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)     TAG="$2"; SESSION="$2"; shift 2 ;;
    -h|--help) sed -n '2,18p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         EXTRA+=("$1"); shift ;;
  esac
done

SESSION="${SESSION//./_}"
RESULTS_DIR="$HERE/results"
LOG="$RESULTS_DIR/${SESSION}.log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux ist nicht installiert." >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' läuft bereits. Anhängen: tmux attach -t $SESSION" >&2
  echo "Oder anderen Namen: --tag <name>" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

CMD=(
  "$PYTHON" "$HERE/worker_sweep.py"
  --tag "$TAG"
  --skip-existing
  --workers 1 2 3 4 6 8 12 16 24 32 48 64
  "${EXTRA[@]+"${EXTRA[@]}"}"
)

LAUNCHER="$RESULTS_DIR/launch_${SESSION}.sh"
{
  echo '#!/usr/bin/env bash'
  echo "# Generiert von run_worker_sweep.sh am $(date -Is)."
  echo 'set -uo pipefail'
  echo
  echo 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1'
  echo
  echo 'echo "Aufruf:"'
  printf 'printf "  %%s\\n" %q\n' "$(printf '%q ' "${CMD[@]}")"
  echo
  printf '%q ' "${CMD[@]}"
  printf '2>&1 | tee -a %q\n' "$LOG"
  echo 'status=${PIPESTATUS[0]}'
  echo 'printf "\n=== beendet mit Status %s ===\n" "$status"'
  echo 'printf "\nFenster bleibt offen. Schliessen: exit\n"'
  echo 'exec bash'
} >"$LAUNCHER"
chmod +x "$LAUNCHER"

printf 'Ergebnisse : %s\n' "$RESULTS_DIR"
printf 'Log        : %s\n' "$LOG"
printf 'Session    : %s\n\n' "$SESSION"

tmux new-session -d -s "$SESSION" -c "$HERE" "exec '$LAUNCHER'"

cat <<EOF
Gestartet.

  tmux attach -t $SESSION        # zuschauen (Ctrl-b d zum Loslösen)
  tail -f $LOG                   # Log verfolgen
  tmux kill-session -t $SESSION  # abbrechen
EOF
