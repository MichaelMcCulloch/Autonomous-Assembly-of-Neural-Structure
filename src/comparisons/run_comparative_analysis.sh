#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Comparative TPE Search over:
#   Models:       RNN (rnn), LSTM (lstm), Transformer (transformer), ESN (esn)
#   Tasks:        narma, mackey, lorenz
# ------------------------------------------------------------------------------

MAX_EVALS="${TPE_EVALS:-50}"
SEED="${TPE_SEED:-0}"
DEVICE="${TPE_DEVICE:-auto}"

TS="$(date +%Y%m%d-%H%M%S)"
OUT_ROOT="${OUT_ROOT:-./comparative_tpe_results/$TS}"

# Prefer 'uv' if present, fall back to 'python'
if command -v uv >/dev/null 2>&1; then
    RUNNER=("uv" "run" "python" "-m")
else
    RUNNER=("python" "-m")
fi

MODELS=("esn" "rnn" "lstm" "transformer")
TASKS=("narma" "mackey" "lorenz")

tpe_module() {
    local model="$1"
    case "$model" in
        rnn)          echo "src.comparisons.rnn.tpe_search" ;;
        lstm)         echo "src.comparisons.lstm.tpe_search" ;;
        transformer)  echo "src.comparisons.transformer.tpe_search" ;;
        esn)          echo "src.comparisons.esn.tpe_search" ;;
        *) echo "Unknown model: $model" >&2; exit 1 ;;
    esac
}

line() { printf '%*s\n' "80" '' | tr ' ' -; }

echo "Comparative TPE search"
echo "  Models : ${MODELS[*]}"
echo "  Tasks  : ${TASKS[*]}"
echo "  Evals  : ${MAX_EVALS}"
echo "  Seed   : ${SEED}"
echo "  Device : ${DEVICE}"
echo "  Output : ${OUT_ROOT}"
line

mkdir -p "$OUT_ROOT"

# Optional: fix threads for reproducibility on CPU
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

for model in "${MODELS[@]}"; do
    for task in "${TASKS[@]}"; do
        MODULE="$(tpe_module "$model")"
        OUT_DIR="$OUT_ROOT/${model}/${task}"
        BEST_DIR="$OUT_DIR/best"
        YAML_PATH="$OUT_DIR/${model}_${task}_best.yaml"
        LOG_PATH="$OUT_DIR/tpe_log.txt"
        
        mkdir -p "$BEST_DIR"
        
        echo
        line
        echo "Model: $model | Task: $task"
        echo "Module: $MODULE"
        echo "OutDir: $OUT_DIR"
        line
        
        {
            echo "[START] $(date -Iseconds) model=$model task=$task evals=$MAX_EVALS seed=$SEED device=$DEVICE"
            "${RUNNER[@]}" "$MODULE" \
            --max-evals "$MAX_EVALS" \
            --seed "$SEED" \
            --device "$DEVICE" \
            --tasks "$task" \
            --out-yaml "$YAML_PATH" \
            --best-out-dir "$BEST_DIR"
            echo "[DONE ] $(date -Iseconds) model=$model task=$task"
        } | tee "$LOG_PATH"
        
        if [ -f "$YAML_PATH" ]; then
            echo "[SUMMARY] Best YAML: $YAML_PATH"
        else
            echo "[WARNING] YAML not found for $model/$task at $YAML_PATH"
        fi
    done
done

line
echo "All TPE runs complete."
echo "Results root: $OUT_ROOT"
echo "Best YAMLs and plots stored per model/task."
line
