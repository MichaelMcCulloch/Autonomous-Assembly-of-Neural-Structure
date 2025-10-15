#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Fast smoke test for all TPE drivers (fail fast).
# Models: rnn, lstm, transformer, esn
# Task:   narma  (keeps dependencies minimal; avoids SciPy)
#
# Env overrides:
#   SMOKE_EVALS   (default: 2)        -> tiny run for speed
#   SMOKE_SEED    (default: 0)
#   SMOKE_DEVICE  (default: auto)     -> 'auto' | 'cpu' | 'cuda'
#   SMOKE_TIMEOUT (default: 180)      -> per-run timeout in seconds (uses `timeout` if available)
#   SMOKE_OUT     (default: ./smoke_tpe_results/<timestamp>)
#
# Exits non-zero on first failure.
# ------------------------------------------------------------------------------

MAX_EVALS="${SMOKE_EVALS:-2}"
SEED="${SMOKE_SEED:-0}"
DEVICE="${SMOKE_DEVICE:-auto}"
TIMEOUT_SEC="${SMOKE_TIMEOUT:-180}"

TS="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="${SMOKE_OUT:-./smoke_tpe_results/$TS}"

# Prefer 'uv' if present, fall back to 'python'
if command -v uv >/dev/null 2>&1; then
    RUNNER=("uv" "run" "python" "-m")
else
    RUNNER=("python" "-m")
fi

# Optional timeout wrapper, if available
if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD=(timeout "${TIMEOUT_SEC}")
else
    TIMEOUT_CMD=()
fi

MODELS=("rnn" "lstm" "transformer" "esn")
TASK="narma"

tpe_module() {
    local model="$1"
    case "$model" in
        rnn)          echo "src.comparisons.rnn.tpe_search" ;;
        lstm)         echo "src.comparisons.lstm.tpe_search" ;;
        transformer)  echo "src.comparisons.transformer.tpe_search" ;;
        esn)          echo "src.comparisons.esn.tpe_search" ;;
        *) echo "[SMOKE] Unknown model: $model" >&2; exit 1 ;;
    esac
}

line() { printf '%*s\n' "80" '' | tr ' ' -; }

echo "[SMOKE] TPE quick-check"
echo "  Models : ${MODELS[*]}"
echo "  Task   : ${TASK}"
echo "  Evals  : ${MAX_EVALS}"
echo "  Seed   : ${SEED}"
echo "  Device : ${DEVICE}"
echo "  Output : ${OUT_ROOT}"
echo "  Timeout: ${TIMEOUT_SEC}s per run (if 'timeout' available)"
line

mkdir -p "$OUT_ROOT"

# Optional: fix threads for reproducibility on CPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

for model in "${MODELS[@]}"; do
    MODULE="$(tpe_module "$model")"
    OUT_DIR="$OUT_ROOT/${model}"
    BEST_DIR="$OUT_DIR/best"
    YAML_PATH="$OUT_DIR/${model}_${TASK}_best.yaml"
    LOG_PATH="$OUT_DIR/smoke_log.txt"
    
    mkdir -p "$BEST_DIR"
    
    line
    echo "[SMOKE] Model: $model | Module: $MODULE"
    echo "[SMOKE] OutDir: $OUT_DIR"
    line
    
    {
        echo "[START] $(date -Iseconds) model=$model task=$TASK evals=$MAX_EVALS seed=$SEED device=$DEVICE"
        "${TIMEOUT_CMD[@]}" "${RUNNER[@]}" "$MODULE" \
        --max-evals "$MAX_EVALS" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --tasks "$TASK" \
        --out-yaml "$YAML_PATH" \
        --best-out-dir "$BEST_DIR"
        echo "[DONE ] $(date -Iseconds) model=$model task=$TASK"
    } | tee "$LOG_PATH"
    
    # Basic assertions: YAML exists and is non-empty
    if [[ ! -f "$YAML_PATH" ]] || [[ ! -s "$YAML_PATH" ]]; then
        echo "[SMOKE][FAIL] Best YAML missing or empty: $YAML_PATH"
        exit 1
    fi
    
    # Optional: ensure some files were produced in the best-out-dir
    # (plot generation; harmless if empty but it helps catch silent failures)
    if [[ -d "$BEST_DIR" ]]; then
        # If directory exists but is empty, warn (don’t fail hard)
        if [ -z "$(ls -A "$BEST_DIR")" ]; then
            echo "[SMOKE][WARN] No plots found in $BEST_DIR (proceeding)"
        fi
    else
        echo "[SMOKE][WARN] Best output dir not found: $BEST_DIR (proceeding)"
    fi
    
    echo "[SMOKE][PASS] $model"
done

line
echo "[SMOKE] All models passed quick-check."
echo "[SMOKE] Results root: $OUT_ROOT"
line
