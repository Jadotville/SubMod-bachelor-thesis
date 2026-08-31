#!/usr/bin/env bash
#
# Launch a full real-world batch in a detached tmux session.
#
# The point of tmux here is that a full batch outlives the ssh connection: detach,
# log out, come back, reattach. The session keeps the live output, and everything
# is written to disk anyway, so nothing depends on the terminal surviving.
#
#   bash run_batch.sh                      # candidates + baselines + controls, offline models
#   bash run_batch.sh --profile quick      # depth 1, small search space — minutes, as a sanity check
#   bash run_batch.sh --depth 3 --tag deep # anything after the profile is passed to run.py
#
# TabPFN locally on one GPU: bash run_tabpfn.sh (not this script).
#
# (`chmod +x run_batch.sh` once if you prefer `./run_batch.sh`.)
#
# Watch it:      tmux attach -t <session>      (detach again with Ctrl-b d)
# Follow the log: tail -f results/<tag>/run.log
# Stop it:       tmux kill-session -t <session>
#
# Restarting the same command after a crash resumes: --skip-existing makes finished
# dataset/model cells be skipped rather than recomputed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/../.." && pwd)"
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

PROFILE="offline"
TAG=""
EXTRA=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    --tag)     TAG="$2";     shift 2 ;;
    -h|--help) sed -n '2,28p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)         EXTRA+=("$1"); shift ;;
  esac
done

# Datasets are named explicitly rather than taken from the default so that the log
# and the session name record exactly what was run.
CANDIDATES="covertype electricity Diabetes130US road-safety ACSPublicCoverage"
BASELINES="ACSIncome ACSMobility ACSTravelTime adult bank-marketing default-of-credit-card-clients"
CONTROLS="PhishingWebsites mushroom"

case "$PROFILE" in
  quick)
    DATASETS="$CANDIDATES $CONTROLS"
    MODELS="lr rf lgbm mlp"
    ARGS=(--depth 1 --max-search-selectors 25)
    ;;
  offline)
    DATASETS="$CANDIDATES $BASELINES $CONTROLS"
    MODELS="lr rf lgbm mlp"
    ARGS=(--depth 2)
    ;;
  tabpfn|full|compare)
    echo "TabPFN läuft lokal auf der GPU, nicht über dieses Skript." >&2
    echo "  bash $HERE/run_tabpfn.sh" >&2
    exit 1
    ;;
  *)
    echo "Unbekanntes Profil: $PROFILE (quick | offline)" >&2
    exit 1
    ;;
esac

[[ -n "$TAG" ]] || TAG="$PROFILE"
RESULTS_DIR="$HERE/results/$TAG"

# The session name carries a --datasets override so a second invocation can start.
SESSION="sd_${TAG}"
for ((i = 0; i < ${#EXTRA[@]}; i++)); do
  if [[ "${EXTRA[i]}" == "--datasets" ]]; then
    for ((j = i + 1; j < ${#EXTRA[@]}; j++)); do
      [[ "${EXTRA[j]}" == --* ]] && break
      SESSION="${SESSION}_${EXTRA[j]}"
    done
    break
  fi
done
# tmux treats a dot as a pane separator in target names.
SESSION="${SESSION//./_}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux ist nicht installiert." >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' läuft bereits. Anhängen: tmux attach -t $SESSION" >&2
  echo "Oder anderen Namen wählen: --tag <name>" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

CMD=(
  "$PYTHON" "$HERE/run.py"
  --datasets $DATASETS
  --models $MODELS
  --results-dir "$RESULTS_DIR"
  --skip-existing
  "${ARGS[@]}"
  "${EXTRA[@]+"${EXTRA[@]}"}"
)

# Written out as a script rather than passed to tmux as one long string: quoting a
# pipeline plus a trailing shell through `tmux new-session` is fragile, and a file
# on disk also documents exactly what the batch was invoked with.
LAUNCHER="$RESULTS_DIR/launch_${SESSION}.sh"
{
  echo '#!/usr/bin/env bash'
  echo "# Generiert von run_batch.sh am $(date -Is). Direkt ausführbar."
  echo 'set -uo pipefail'
  echo
  echo '# BLAS muss vor dem Interpreterstart gesetzt sein; run.py tut das ebenfalls,'
  echo '# aber hier gilt es auch für alles, was der Batch sonst startet.'
  echo 'export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1'
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
  echo 'printf "\nFenster bleibt offen. Schliessen: exit\n"'
  echo 'exec bash'
} >"$LAUNCHER"
chmod +x "$LAUNCHER"

DS_SHOWN="$DATASETS"
MD_SHOWN="$MODELS"
# A later --datasets/--models on the command line wins in argparse, so do not print
# the profile's list as if it were what runs.
[[ " ${EXTRA[*]-} " == *" --datasets "* ]] && DS_SHOWN="(per Argument überschrieben)"
[[ " ${EXTRA[*]-} " == *" --models "* ]] && MD_SHOWN="(per Argument überschrieben)"

printf 'Profil     : %s\n' "$PROFILE"
printf 'Datensätze : %s\n' "$DS_SHOWN"
printf 'Modelle    : %s\n' "$MD_SHOWN"
printf 'Ergebnisse : %s\n' "$RESULTS_DIR"
printf 'Session    : %s\n\n' "$SESSION"

# The command string is re-parsed by a shell inside tmux, so the path has to be
# quoted *within* it — the project directory contains a space, and an unquoted path
# makes the session start, fail on the first word and vanish before it logs anything.
tmux new-session -d -s "$SESSION" -c "$HERE" "exec '$LAUNCHER'"

cat <<EOF
Gestartet. Nützliche Befehle:

  tmux attach -t $SESSION        # zuschauen (Ctrl-b d zum Loslösen)
  tail -f $RESULTS_DIR/run.log   # Log verfolgen
  tmux kill-session -t $SESSION  # abbrechen

Nach dem Lauf:

  $PYTHON $HERE/summarize_runs.py --results-dir $RESULTS_DIR
EOF
