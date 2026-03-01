#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# 🚀 run_experiments.sh - Automated Training Pipeline for RTX 5090 (32GB VRAM)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Runs the full data generation → SFT → GRPO → Quantize pipeline with
# multiple hyperparameter sweeps, tracked by Weights & Biases.
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh              # Full pipeline (data gen + all experiments)
#   ./run_experiments.sh --skip-data  # Skip data generation (reuse existing data)
#   ./run_experiments.sh --quick      # Quick mode: fewer samples, 1 epoch, 1 config
#   ./run_experiments.sh --deploy-only # Skip training, just deploy best existing model
#   ./run_experiments.sh --no-train   # Download pre-quantized GGUFs → deploy (no training)
#   ./run_experiments.sh --edge-stack  # Download Ministral + Voxtral GGUFs for full edge stack
#
# Unattended (leave running while away):
#   nohup ./run_experiments.sh > pipeline_stdout.log 2>&1 &
#   disown
#   # Check later: tail -f logs/pipeline_*.log
#   # Or check W&B dashboard - you'll get an alert when it finishes
#
# Hardware requirements:
#   - RTX 5090 (32GB VRAM) - Blackwell architecture (sm_100, CUDA 12.8+)
#   - OR Orin Nano Super (8GB) - JetPack 6.2 (sm_87, CUDA 12.6)
#
# Timeline: ~2-3 hours unattended on RTX 5090
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail
IFS=$'\n\t'

# ─── Color Output ─────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()    { echo -e "${BLUE}[INFO]${NC}  $(date '+%H:%M:%S') $*"; }
log_success() { echo -e "${GREEN}[✅]${NC}    $(date '+%H:%M:%S') $*"; }
log_warn()    { echo -e "${YELLOW}[⚠️]${NC}    $(date '+%H:%M:%S') $*"; }
log_error()   { echo -e "${RED}[❌]${NC}    $(date '+%H:%M:%S') $*"; }
log_step()    { echo -e "\n${CYAN}═══════════════════════════════════════════════════════════${NC}"; \
                echo -e "${CYAN}  $*${NC}"; \
                echo -e "${CYAN}═══════════════════════════════════════════════════════════${NC}\n"; }

# ─── Configuration ────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
    log_info "Loaded .env file"
else
    log_error ".env file not found! Copy .env.example to .env and configure."
    exit 1
fi

# Parse arguments
SKIP_DATA=false
QUICK_MODE=false
PARALLEL=false
DEPLOY_ONLY=false
NO_TRAIN=false
EDGE_STACK=false
for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --quick) QUICK_MODE=true ;;
        --parallel) PARALLEL=true ;;
        --deploy-only) DEPLOY_ONLY=true ;;
        --no-train) NO_TRAIN=true ;;
        --edge-stack) EDGE_STACK=true; NO_TRAIN=true ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Training modes:"
            echo "  (no flags)      Full pipeline: data gen + SFT sweep + GRPO + quantize + deploy"
            echo "  --skip-data     Reuse existing training data (skip generation)"
            echo "  --quick         Quick mode: 50 samples, 1 epoch, 1 config"
            echo "  --parallel      Run SFT experiments in parallel (needs multi-GPU)"
            echo ""
            echo "Deployment shortcuts (no 5090 needed for inference):"
            echo "  --no-train      Download pre-quantized Ministral 3B GGUF → deploy directly"
            echo "  --edge-stack    Download Ministral + Voxtral GGUFs for full edge AI stack"
            echo "  --deploy-only   Skip training, deploy best existing model from models/"
            echo ""
            echo "The value proposition: use the 5090 to TRAIN specialized models,"
            echo "then deploy to Orin Nano (8GB) for fully offline edge inference."
            echo "No 5090 needed at inference time."
            exit 0 ;;
    esac
done

# ─── Directories ──────────────────────────────────────────────────────────────
DATA_DIR="$SCRIPT_DIR/data"
MODEL_DIR="$SCRIPT_DIR/models"
LOG_DIR="$SCRIPT_DIR/logs"
RESULTS_FILE="$LOG_DIR/experiment_results.json"

mkdir -p "$DATA_DIR" "$MODEL_DIR" "$LOG_DIR"

# ─── Timestamps ───────────────────────────────────────────────────────────────
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
PIPELINE_LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"
PIPELINE_START_EPOCH=$(date +%s)

# Tee all output to log file
exec > >(tee -a "$PIPELINE_LOG") 2>&1

# ─── Notification on Completion ───────────────────────────────────────────────
# Sends a W&B alert + optional desktop notification when pipeline finishes
notify_completion() {
    local status="$1"
    local message="$2"
    local elapsed=$(( ($(date +%s) - PIPELINE_START_EPOCH) / 60 ))

    # W&B alert (visible on phone/laptop) - skip in offline mode
    if [ "${WANDB_MODE:-}" != "offline" ]; then
        timeout 30 python3 -c "
import wandb
try:
    wandb.init(project='${WANDB_PROJECT:-reachy-copilot}', name='pipeline-alert', reinit=True,
              settings=wandb.Settings(init_timeout=20))
    wandb.alert(
        title='Pipeline $status',
        text='$message (${elapsed}min elapsed)',
        level=wandb.AlertLevel.INFO if '$status' == 'COMPLETE' else wandb.AlertLevel.ERROR
    )
    wandb.finish(quiet=True)
except Exception as e:
    print(f'W&B alert failed: {e}')
" 2>/dev/null || true
    else
        log_info "W&B offline mode - skipping alert notification"
    fi

    log_info "Pipeline $status after ${elapsed} minutes"
}

# Trap for unexpected exits - get notified even if something crashes
trap 'notify_completion "FAILED" "Pipeline crashed at $(date). Check $PIPELINE_LOG"' ERR

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0: Environment Verification
# ═══════════════════════════════════════════════════════════════════════════════
log_step "PHASE 0: Environment Verification"

# Check Python
if ! command -v python3 &>/dev/null; then
    log_error "python3 not found! Install Python 3.10+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1)
log_info "Python: $PYTHON_VERSION"

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    log_info "Activated .venv"
elif [ -d "venv" ]; then
    source venv/bin/activate
    log_info "Activated venv"
fi

# Check CUDA + GPU
log_info "Checking CUDA and GPU..."
python3 -c "
import torch
import sys

if not torch.cuda.is_available():
    print('❌ CUDA not available!')
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
cuda_version = torch.version.cuda
arch = torch.cuda.get_device_capability(0)

print(f'✅ GPU: {gpu_name}')
print(f'   VRAM: {gpu_mem:.1f} GB')
print(f'   CUDA: {cuda_version}')
print(f'   Compute capability: sm_{arch[0]}{arch[1]}')

# Verify architecture compatibility
major, minor = arch
if major >= 10:
    print(f'   ✅ Blackwell architecture (sm_{major}{minor}) - RTX 5090 supported')
elif major == 8 and minor == 7:
    print(f'   ✅ Orin Nano (sm_87) - JetPack 6.x supported')
elif major >= 8:
    print(f'   ✅ Ampere/Ada/Hopper architecture - supported')
else:
    print(f'   ⚠️  Older GPU architecture - may have compatibility issues')

# Check bf16 support (RTX 5090 Blackwell supports bf16)
if torch.cuda.is_bf16_supported():
    print('   ✅ BF16 supported - optimal for training')
else:
    print('   ⚠️  BF16 not supported - will use FP16')
" || { log_error "GPU check failed"; exit 1; }

# Check required packages
log_info "Checking Python packages..."
python3 -c "
import importlib
packages = {
    'torch': 'torch',
    'transformers': 'transformers',
    'trl': 'trl',
    'peft': 'peft',
    'bitsandbytes': 'bitsandbytes',
    'datasets': 'datasets',
    'accelerate': 'accelerate',
    'wandb': 'wandb',
    'openai': 'openai',
    'dotenv': 'python-dotenv',
}
missing = []
for name, pip_name in packages.items():
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(pip_name)

if missing:
    print(f'❌ Missing packages: {\" \".join(missing)}')
    print(f'   Run: pip install {\" \".join(missing)}')
    exit(1)
else:
    print('✅ All required packages installed')
"

# Check W&B
if [ -n "${WANDB_API_KEY:-}" ]; then
    if [ "${WANDB_MODE:-}" = "offline" ]; then
        log_success "W&B configured in OFFLINE mode - logs saved locally, sync later with: wandb sync wandb/latest-run"
    else
        log_success "W&B API key configured - experiments will be tracked online"
        timeout 15 python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}', relogin=True)" 2>/dev/null || \
            log_warn "W&B login timed out - continuing anyway (offline logs will still work)"
    fi
else
    log_warn "WANDB_API_KEY not set - experiments will NOT be tracked"
fi

# Check HuggingFace token
if [ -n "${HF_TOKEN:-}" ]; then
    log_success "HuggingFace token configured"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
else
    log_warn "HF_TOKEN not set - may not be able to download gated models"
fi

log_success "Environment verification complete!"

# ─── Resilient Mode ───────────────────────────────────────────────────────────
# Disable exit-on-error for training phases so individual experiment failures
# don't kill the entire pipeline. Each phase checks its own prerequisites.
set +e
PIPELINE_FAILURES=0

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 0.5: Model Download & Cache Check
# ═══════════════════════════════════════════════════════════════════════════════
log_step "PHASE 0.5: Model Download & Cache Check"

if [ "$NO_TRAIN" = true ]; then
    # ─── NO-TRAIN MODE ────────────────────────────────────────────────────
    # Download pre-quantized GGUFs and deploy directly.
    # No 5090 GPU needed for this path - models run on Orin Nano (8GB).
    log_info "🚀 NO-TRAIN MODE - Downloading pre-quantized GGUFs for edge deployment"
    log_info "   No fine-tuning. Using off-the-shelf Ministral 3B GGUF."
    log_info "   Value prop: works today, fine-tuned model is an upgrade later."

    DOWNLOAD_FLAGS="--no-train --output $MODEL_DIR"
    if [ "$EDGE_STACK" = true ]; then
        DOWNLOAD_FLAGS="--edge-stack --output $MODEL_DIR"
        log_info "   Also downloading Voxtral Mini 3B for audio understanding on edge"
    fi

    python3 scripts/07_download_models.py $DOWNLOAD_FLAGS \
        2>&1 | tee "$LOG_DIR/model_download_${TIMESTAMP}.log"

    # Find the downloaded GGUF
    GGUF_FILE=$(find "$MODEL_DIR" -name "*Q4_K_M.gguf" -not -name "*Voxtral*" | head -1)
    if [ -z "$GGUF_FILE" ]; then
        log_error "No GGUF file found after download!"
        exit 1
    fi
    log_success "GGUF model ready: $GGUF_FILE"

    # Create Modelfile if not exists
    MODELFILE="$MODEL_DIR/Modelfile"
    if [ ! -f "$MODELFILE" ]; then
        GGUF_BASENAME=$(basename "$GGUF_FILE")
        python3 scripts/07_download_models.py --no-train --output "$MODEL_DIR" --check-only 2>/dev/null || true
    fi

    # Skip to deployment
    DEPLOY_BEST="$MODEL_DIR"
    BEST_SFT_NAME="ministral-3b-pretrained"
    BEST_GRPO_NAME=""

    # Jump to Phase 5 (Edge Deployment)
    log_step "PHASE 5: Deploy Pre-Quantized Model to Orin Nano"

    ORIN_IP="${ORIN_IP:-192.168.1.50}"
    ORIN_USER="${ORIN_USER:-orin}"

    if ping -c 1 -W 2 "$ORIN_IP" &>/dev/null; then
        log_info "Orin Nano reachable at $ORIN_IP - deploying..."
        ssh "$ORIN_USER@$ORIN_IP" "mkdir -p ~/reachy-model" 2>/dev/null || true
        scp "$GGUF_FILE" "$ORIN_USER@$ORIN_IP:~/reachy-model/" 2>&1 || log_warn "SCP failed"
        [ -f "$MODELFILE" ] && scp "$MODELFILE" "$ORIN_USER@$ORIN_IP:~/reachy-model/" 2>&1 || true

        # Deploy Voxtral too if edge-stack
        if [ "$EDGE_STACK" = true ]; then
            VOXTRAL_GGUF=$(find "$MODEL_DIR" -name "*Voxtral*Q4_K_M.gguf" | head -1)
            if [ -n "$VOXTRAL_GGUF" ]; then
                scp "$VOXTRAL_GGUF" "$ORIN_USER@$ORIN_IP:~/reachy-model/" 2>&1 || true
                log_success "Voxtral audio model also deployed"
            fi
        fi

        # Create Ollama models on Orin
        ssh "$ORIN_USER@$ORIN_IP" "cd ~/reachy-model && ollama create reachy-copilot -f Modelfile" 2>&1 || \
            log_warn "Ollama model creation failed - do it manually on the Orin"

        # Copy bridge + memory scripts
        ssh "$ORIN_USER@$ORIN_IP" "mkdir -p ~/reachy-bridge" 2>/dev/null || true
        scp scripts/05_memory_manager.py scripts/06_openclaw_bridge.py \
            "$ORIN_USER@$ORIN_IP:~/reachy-bridge/" 2>&1 || true

        log_success "Edge deployment complete!"
    else
        log_warn "Orin Nano not reachable at $ORIN_IP"
        log_info "Deploy manually: scp $GGUF_FILE orin@<IP>:~/reachy-model/"
    fi

    # Print summary and exit
    PIPELINE_END=$(date +%s)
    PIPELINE_DURATION=$(( (PIPELINE_END - PIPELINE_START_EPOCH) / 60 ))
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  ⚡ NO-TRAIN DEPLOYMENT COMPLETE"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "  📦 GGUF model:      $GGUF_FILE"
    if [ "$EDGE_STACK" = true ]; then
        echo "  🎤 Voxtral model:   $(find $MODEL_DIR -name '*Voxtral*Q4_K_M.gguf' | head -1)"
    fi
    echo "  🎯 Deploy target:   $ORIN_USER@$ORIN_IP"
    echo ""
    echo "  💡 This uses the off-the-shelf Ministral 3B model."
    echo "     To specialize it for Reachy tool-calling, run the full"
    echo "     training pipeline later on the 5090:"
    echo "       ./run_experiments.sh          # Full SFT + GRPO"
    echo "       ./run_experiments.sh --quick   # Quick 1-config run"
    echo ""
    echo "  Pipeline completed in ${PIPELINE_DURATION} minutes"
    echo "═══════════════════════════════════════════════════════════"

    notify_completion "COMPLETE" "No-train deployment done. Model: $GGUF_FILE"
    exit 0
fi

if [ "$DEPLOY_ONLY" = true ]; then
    # ─── DEPLOY-ONLY MODE ─────────────────────────────────────────────────
    log_info "🚀 DEPLOY-ONLY MODE - Looking for existing trained models..."

    BEST_GGUF=$(find "$MODEL_DIR" -name "model-q4_k_m.gguf" -o -name "*Q4_K_M.gguf" 2>/dev/null | head -1)
    if [ -z "$BEST_GGUF" ]; then
        log_error "No quantized model found in $MODEL_DIR!"
        log_info "Options:"
        log_info "  1. Run full pipeline first: ./run_experiments.sh"
        log_info "  2. Use no-train mode:       ./run_experiments.sh --no-train"
        exit 1
    fi

    DEPLOY_BEST=$(dirname "$BEST_GGUF")
    log_success "Found model: $BEST_GGUF"

    # Jump to Phase 5 (reuse same deployment logic as main pipeline)
    # ... falls through to Phase 5 below
    BEST_SFT_NAME="existing-model"
    BEST_GRPO_NAME=""
    BEST_SFT_MODEL="$DEPLOY_BEST"
    BEST_GRPO_MODEL=""
    SKIP_DATA=true
fi

# ─── Download base models for training ────────────────────────────────────────
if [ "$DEPLOY_ONLY" = false ]; then
    log_info "Checking base model cache (needed for SFT training)..."
    python3 scripts/07_download_models.py --check-only \
        2>&1 | tee -a "$LOG_DIR/model_download_${TIMESTAMP}.log"

    # Download if not cached
    python3 scripts/07_download_models.py \
        2>&1 | tee -a "$LOG_DIR/model_download_${TIMESTAMP}.log" || {
        log_error "Model download failed! Check HF_TOKEN and internet connection."
        log_info "If you want to skip training and use a pre-quantized model:"
        log_info "  ./run_experiments.sh --no-train"
        exit 1
    }
    log_success "Base models ready for training"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASES 1-4: Training Pipeline (skipped in --deploy-only mode)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$DEPLOY_ONLY" = true ]; then
    log_info "⏭️  Skipping training phases (--deploy-only mode)"
    log_info "   Using existing model: ${BEST_SFT_MODEL:-$MODEL_DIR}"
    TRAINING_DATA=""
else
# ─── Begin training pipeline ─────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Data Generation
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_DATA" = false ]; then
    log_step "PHASE 1: Data Generation (Teacher → Student Distillation)"

    if [ "$QUICK_MODE" = true ]; then
        NUM_SAMPLES=50
    else
        NUM_SAMPLES=100
    fi

    TRAINING_DATA="$DATA_DIR/training_data.jsonl"

    # Generate data using Mistral Large as teacher
    log_info "Generating $NUM_SAMPLES training samples..."
    python3 scripts/01_generate_training_data.py \
        --output "$TRAINING_DATA" \
        --num-samples "$NUM_SAMPLES" \
        --direct-only \
        2>&1 | tee "$LOG_DIR/data_gen_${TIMESTAMP}.log"

    if [ -f "$TRAINING_DATA" ]; then
        SAMPLE_COUNT=$(wc -l < "$TRAINING_DATA" | tr -d ' ')
        log_success "Generated $SAMPLE_COUNT training samples → $TRAINING_DATA"
    else
        log_error "Data generation failed - no output file created"
        log_warn "⏭️  SFT and GRPO phases will be skipped (no training data)"
        PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
        TRAINING_DATA=""
    fi

    # Also try HuggingFace datasets if not quick mode
    if [ "$QUICK_MODE" = false ]; then
        log_info "Augmenting with HuggingFace datasets..."
        AUGMENTED_DATA="$DATA_DIR/training_data_augmented.jsonl"
        python3 scripts/01_generate_training_data.py \
            --output "$AUGMENTED_DATA" \
            --num-samples "$NUM_SAMPLES" \
            --include-hf \
            2>&1 | tee "$LOG_DIR/data_gen_hf_${TIMESTAMP}.log" || {
            log_warn "HuggingFace augmentation failed - continuing with direct data only"
            AUGMENTED_DATA="$TRAINING_DATA"
        }
        if [ -f "$AUGMENTED_DATA" ]; then
            AUG_COUNT=$(wc -l < "$AUGMENTED_DATA" | tr -d ' ')
            log_success "Augmented dataset: $AUG_COUNT samples → $AUGMENTED_DATA"
            TRAINING_DATA="$AUGMENTED_DATA"
        fi
    fi
else
    log_info "Skipping data generation (--skip-data)"
    # Prefer augmented data (more samples) over base data
    if [ -f "$DATA_DIR/training_data_augmented.jsonl" ]; then
        TRAINING_DATA="$DATA_DIR/training_data_augmented.jsonl"
    elif [ -f "$DATA_DIR/training_data.jsonl" ]; then
        TRAINING_DATA="$DATA_DIR/training_data.jsonl"
    else
        log_error "No training data found! Run without --skip-data first."
        log_warn "⏭️  SFT and GRPO phases will be skipped (no training data)"
        PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
        TRAINING_DATA=""
    fi
    if [ -n "$TRAINING_DATA" ] && [ -f "$TRAINING_DATA" ]; then
    SAMPLE_COUNT=$(wc -l < "$TRAINING_DATA" | tr -d ' ')
    log_info "Using existing data: $SAMPLE_COUNT samples from $TRAINING_DATA"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: SFT Hyperparameter Sweep
# ═══════════════════════════════════════════════════════════════════════════════
log_step "PHASE 2: SFT with QLoRA - Hyperparameter Sweep"

# Initialize results tracker
echo '{"experiments": [], "best_sft": null, "best_grpo": null}' > "$RESULTS_FILE"

# ─── Define experiments ───────────────────────────────────────────────────────
# Format: "LORA_R EPOCHS LR BATCH_SIZE MAX_SEQ_LEN RUN_NAME"
if [ "$QUICK_MODE" = true ]; then
    SFT_EXPERIMENTS=(
        "32 1 2e-4 4 2048 sft-r32-ep1-quick"
    )
else
    SFT_EXPERIMENTS=(
        "16 3 2e-4 2 1024 sft-r16-lr2e4"
        "32 3 2e-4 2 1024 sft-r32-lr2e4"
        "64 3 2e-4 2 1024 sft-r64-lr2e4"
        "32 3 1e-4 2 1024 sft-r32-lr1e4"
        "32 5 2e-4 2 1024 sft-r32-ep5-lr2e4"
    )
fi

BEST_SFT_MODEL=""
BEST_SFT_NAME=""
EXP_NUM=0

# Skip SFT if no training data available
if [ -z "${TRAINING_DATA:-}" ] || [ ! -f "${TRAINING_DATA:-/nonexistent}" ]; then
    log_warn "⏭️  Skipping all SFT experiments - no training data available"
    PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
    SFT_EXPERIMENTS=()
fi

run_sft_experiment() {
    local exp="$1"
    local LORA_R EPOCHS LR BATCH_SIZE MAX_SEQ RUN_NAME
    IFS=' ' read -r LORA_R EPOCHS LR BATCH_SIZE MAX_SEQ RUN_NAME <<< "$exp"

    local OUTPUT="$MODEL_DIR/$RUN_NAME"

    log_info "▶ SFT Experiment: $RUN_NAME"
    log_info "  LoRA rank=$LORA_R, epochs=$EPOCHS, lr=$LR, batch=$BATCH_SIZE, seq=$MAX_SEQ"

    python3 scripts/02_sft_qlora.py \
        --data "$TRAINING_DATA" \
        --output "$OUTPUT" \
        --lora-r "$LORA_R" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --max-seq-len "$MAX_SEQ" \
        --grad-accum 4 \
        --wandb-run-name "$RUN_NAME" \
        2>&1 | tee "$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log"

    if [ -d "$OUTPUT" ] && [ -f "$OUTPUT/adapter_config.json" ]; then
        log_success "SFT experiment $RUN_NAME completed → $OUTPUT"
        return 0
    else
        log_error "SFT experiment $RUN_NAME failed"
        return 1
    fi
}

for exp in "${SFT_EXPERIMENTS[@]}"; do
    EXP_NUM=$((EXP_NUM + 1))
    IFS=' ' read -r _ _ _ _ _ RUN_NAME <<< "$exp"

    log_info "━━━ SFT Experiment $EXP_NUM/${#SFT_EXPERIMENTS[@]}: $RUN_NAME ━━━"

    if run_sft_experiment "$exp"; then
        # Compare eval_loss to find the actual best model
        CANDIDATE_DIR="$MODEL_DIR/$RUN_NAME"
        CANDIDATE_LOSS=$(python3 -c "
import json, sys
try:
    info = json.load(open('$CANDIDATE_DIR/training_info.json'))
    loss = info.get('best_eval_loss') or info.get('final_train_loss')
    print(loss if loss is not None else 'none')
except: print('none')
" 2>/dev/null)

        log_info "  📊 $RUN_NAME eval_loss: $CANDIDATE_LOSS"

        if [ -z "$BEST_SFT_MODEL" ]; then
            BEST_SFT_MODEL="$CANDIDATE_DIR"
            BEST_SFT_NAME="$RUN_NAME"
            BEST_SFT_LOSS="$CANDIDATE_LOSS"
        elif [ "$CANDIDATE_LOSS" != "none" ] && [ "$BEST_SFT_LOSS" != "none" ]; then
            IS_BETTER=$(python3 -c "print('yes' if float('$CANDIDATE_LOSS') < float('$BEST_SFT_LOSS') else 'no')" 2>/dev/null)
            if [ "$IS_BETTER" = "yes" ]; then
                log_success "  🏆 New best SFT! $RUN_NAME ($CANDIDATE_LOSS) beats $BEST_SFT_NAME ($BEST_SFT_LOSS)"
                BEST_SFT_MODEL="$CANDIDATE_DIR"
                BEST_SFT_NAME="$RUN_NAME"
                BEST_SFT_LOSS="$CANDIDATE_LOSS"
            fi
        elif [ -z "$BEST_SFT_LOSS" ] || [ "$BEST_SFT_LOSS" = "none" ]; then
            BEST_SFT_MODEL="$CANDIDATE_DIR"
            BEST_SFT_NAME="$RUN_NAME"
            BEST_SFT_LOSS="$CANDIDATE_LOSS"
        fi
    else
        log_warn "SFT experiment $RUN_NAME failed - continuing with next..."
        PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
    fi
done

if [ -z "$BEST_SFT_MODEL" ]; then
    log_error "All SFT experiments failed!"
    log_warn "⏭️  GRPO phase will be skipped (no SFT model to build on)"
else
    log_success "SFT sweep complete. Best: $BEST_SFT_NAME (eval_loss=${BEST_SFT_LOSS:-N/A})"
    log_info "Check W&B dashboard to compare runs and pick the best model."
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: GRPO Reinforcement Learning Sweep
# ═══════════════════════════════════════════════════════════════════════════════
log_step "PHASE 3: GRPO Agent Training - RL Sweep"

# Use the best SFT model (or the default r32 one)
# Prefer r32 as it's a good balance
GRPO_BASE="$MODEL_DIR/sft-r32-lr2e4"
if [ ! -d "$GRPO_BASE" ]; then
    GRPO_BASE="$BEST_SFT_MODEL"
fi

# Format: "NUM_GENERATIONS EPOCHS LR RUN_NAME"
if [ "$QUICK_MODE" = true ]; then
    GRPO_EXPERIMENTS=(
        "4 1 5e-6 grpo-g4-quick"
    )
else
    GRPO_EXPERIMENTS=(
        "4 1 5e-6 grpo-g4-lr5e6"
        "8 1 5e-6 grpo-g8-lr5e6"
        "4 2 1e-5 grpo-g4-ep2-lr1e5"
    )
fi

BEST_GRPO_MODEL=""
BEST_GRPO_NAME=""
EXP_NUM=0

# Skip GRPO if no valid base model available
if [ -z "${GRPO_BASE:-}" ] || [ ! -d "${GRPO_BASE:-/nonexistent}" ]; then
    log_warn "⏭️  Skipping all GRPO experiments - no valid SFT base model"
    PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
    GRPO_EXPERIMENTS=()
fi

for exp in "${GRPO_EXPERIMENTS[@]}"; do
    EXP_NUM=$((EXP_NUM + 1))
    IFS=' ' read -r NUM_GEN EPOCHS LR RUN_NAME <<< "$exp"
    OUTPUT="$MODEL_DIR/$RUN_NAME"

    log_info "━━━ GRPO Experiment $EXP_NUM/${#GRPO_EXPERIMENTS[@]}: $RUN_NAME ━━━"
    log_info "  generations=$NUM_GEN, epochs=$EPOCHS, lr=$LR, base=$GRPO_BASE"

    python3 scripts/03_grpo_agent.py \
        --model "$GRPO_BASE" \
        --output "$OUTPUT" \
        --num-generations "$NUM_GEN" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --batch-size 2 \
        --wandb-run-name "$RUN_NAME" \
        2>&1 | tee "$LOG_DIR/${RUN_NAME}_${TIMESTAMP}.log" || {
        log_warn "GRPO experiment $RUN_NAME crashed - continuing with next..."
        PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
    }

    if [ -d "$OUTPUT" ]; then
        log_success "GRPO experiment $RUN_NAME completed → $OUTPUT"

        # Compare reward scores to find the actual best GRPO model
        CANDIDATE_REWARD=$(python3 -c "
import json
try:
    info = json.load(open('$OUTPUT/training_info.json'))
    reward = info.get('best_reward')
    print(reward if reward is not None else 'none')
except: print('none')
" 2>/dev/null)

        log_info "  📊 $RUN_NAME best_reward: $CANDIDATE_REWARD"

        if [ -z "$BEST_GRPO_MODEL" ]; then
            BEST_GRPO_MODEL="$OUTPUT"
            BEST_GRPO_NAME="$RUN_NAME"
            BEST_GRPO_REWARD="$CANDIDATE_REWARD"
        elif [ "$CANDIDATE_REWARD" != "none" ] && [ "$BEST_GRPO_REWARD" != "none" ]; then
            IS_BETTER=$(python3 -c "print('yes' if float('$CANDIDATE_REWARD') > float('$BEST_GRPO_REWARD') else 'no')" 2>/dev/null)
            if [ "$IS_BETTER" = "yes" ]; then
                log_success "  🏆 New best GRPO! $RUN_NAME ($CANDIDATE_REWARD) beats $BEST_GRPO_NAME ($BEST_GRPO_REWARD)"
                BEST_GRPO_MODEL="$OUTPUT"
                BEST_GRPO_NAME="$RUN_NAME"
                BEST_GRPO_REWARD="$CANDIDATE_REWARD"
            fi
        elif [ -z "$BEST_GRPO_REWARD" ] || [ "$BEST_GRPO_REWARD" = "none" ]; then
            BEST_GRPO_MODEL="$OUTPUT"
            BEST_GRPO_NAME="$RUN_NAME"
            BEST_GRPO_REWARD="$CANDIDATE_REWARD"
        fi
    else
        log_error "GRPO experiment $RUN_NAME failed - continuing..."
    fi
done

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Quantization (Best Models)
# ═══════════════════════════════════════════════════════════════════════════════
log_step "PHASE 4: Quantize Best Models for Edge Deployment"

# Check if llama.cpp is available
LLAMA_CPP_PATH="$SCRIPT_DIR/llama.cpp"
if [ ! -d "$LLAMA_CPP_PATH" ]; then
    log_info "Cloning llama.cpp..."
    if git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_PATH" 2>&1; then
        cd "$LLAMA_CPP_PATH"

        # Build with CUDA (auto-detect architecture)
        log_info "Building llama.cpp with CUDA..."
        if cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc); then
            log_success "llama.cpp built successfully"
        else
            log_warn "llama.cpp build failed - quantization may not work"
            PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
        fi
        cd "$SCRIPT_DIR"
    else
        log_warn "Failed to clone llama.cpp - quantization will be skipped"
        PIPELINE_FAILURES=$((PIPELINE_FAILURES + 1))
    fi
fi

# Quantize the best SFT model
QUANT_TYPES=("Q4_K_M" "Q4_K_S")

for QUANT in "${QUANT_TYPES[@]}"; do
    # Quantize best SFT model
    if [ -n "$BEST_SFT_MODEL" ]; then
        log_info "Quantizing SFT model ($BEST_SFT_NAME) → $QUANT..."
        python3 scripts/04_quantize_deploy.py \
            --model "$BEST_SFT_MODEL" \
            --output "$MODEL_DIR/${BEST_SFT_NAME}-gguf-${QUANT}" \
            --llama-cpp "$LLAMA_CPP_PATH" \
            --quant "$QUANT" \
            2>&1 | tee "$LOG_DIR/quantize_sft_${QUANT}_${TIMESTAMP}.log" || {
            log_warn "SFT quantization ($QUANT) failed - continuing..."
        }
    fi

    # Quantize best GRPO model (if exists)
    if [ -n "$BEST_GRPO_MODEL" ]; then
        log_info "Quantizing GRPO model ($BEST_GRPO_NAME) → $QUANT..."
        python3 scripts/04_quantize_deploy.py \
            --model "$BEST_GRPO_MODEL" \
            --output "$MODEL_DIR/${BEST_GRPO_NAME}-gguf-${QUANT}" \
            --llama-cpp "$LLAMA_CPP_PATH" \
            --quant "$QUANT" \
            2>&1 | tee "$LOG_DIR/quantize_grpo_${QUANT}_${TIMESTAMP}.log" || {
            log_warn "GRPO quantization ($QUANT) failed - continuing..."
        }
    fi

    # Only do Q4_K_M in quick mode
    if [ "$QUICK_MODE" = true ]; then
        break
    fi
done

# ─── End training pipeline ───────────────────────────────────────────────────
fi  # end of DEPLOY_ONLY check

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: Edge Deployment Preparation
# ═══════════════════════════════════════════════════════════════════════════════
log_step "PHASE 5: Prepare Edge Deployment (Orin Nano)"

ORIN_IP="${ORIN_IP:-192.168.1.50}"
ORIN_USER="${ORIN_USER:-orin}"
DEPLOY_BEST=""

# Pick the best quantized model
if [ -n "$BEST_GRPO_MODEL" ]; then
    DEPLOY_CANDIDATE="$MODEL_DIR/${BEST_GRPO_NAME}-gguf-Q4_K_M"
    if [ -d "$DEPLOY_CANDIDATE" ]; then
        DEPLOY_BEST="$DEPLOY_CANDIDATE"
        log_info "Best model for deployment: GRPO ($BEST_GRPO_NAME)"
    fi
fi
if [ -z "$DEPLOY_BEST" ] && [ -n "$BEST_SFT_MODEL" ]; then
    DEPLOY_CANDIDATE="$MODEL_DIR/${BEST_SFT_NAME}-gguf-Q4_K_M"
    if [ -d "$DEPLOY_CANDIDATE" ]; then
        DEPLOY_BEST="$DEPLOY_CANDIDATE"
        log_info "Best model for deployment: SFT ($BEST_SFT_NAME)"
    fi
fi

if [ -n "$DEPLOY_BEST" ]; then
    log_info "Deployment candidate: $DEPLOY_BEST"

    # Try to deploy to Orin if reachable
    if ping -c 1 -W 2 "$ORIN_IP" &>/dev/null; then
        log_info "Orin Nano reachable at $ORIN_IP - deploying model..."

        # Copy model files
        ssh "$ORIN_USER@$ORIN_IP" "mkdir -p ~/reachy-model" 2>/dev/null || true
        scp -r "$DEPLOY_BEST"/* "$ORIN_USER@$ORIN_IP:~/reachy-model/" 2>&1 || {
            log_warn "SCP to Orin failed - model saved locally at $DEPLOY_BEST"
        }

        # Create Ollama model on Orin
        ssh "$ORIN_USER@$ORIN_IP" "cd ~/reachy-model && ollama create reachy-copilot -f Modelfile" 2>&1 || {
            log_warn "Ollama model creation on Orin failed - do it manually"
        }

        # Copy bridge + memory scripts to Orin
        ssh "$ORIN_USER@$ORIN_IP" "mkdir -p ~/reachy-bridge" 2>/dev/null || true
        scp scripts/05_memory_manager.py "$ORIN_USER@$ORIN_IP:~/reachy-bridge/" 2>&1 || true
        scp scripts/06_openclaw_bridge.py "$ORIN_USER@$ORIN_IP:~/reachy-bridge/" 2>&1 || true

        log_success "Model deployed to Orin Nano!"
    else
        log_warn "Orin Nano not reachable at $ORIN_IP - skipping auto-deploy"
        log_info "Deploy manually later:"
        log_info "  scp -r $DEPLOY_BEST/* $ORIN_USER@$ORIN_IP:~/reachy-model/"
    fi
else
    log_warn "No quantized model found for deployment"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Memory Manager Test
# ═══════════════════════════════════════════════════════════════════════════════
log_step "PHASE 6: Memory Manager Self-Test"

python3 scripts/05_memory_manager.py --test 2>&1 || {
    log_warn "Memory manager test failed - check scripts/05_memory_manager.py"
}

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7: Summary & Next Steps
# ═══════════════════════════════════════════════════════════════════════════════
log_step "PHASE 7: Pipeline Complete - Summary"

PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$(( (PIPELINE_END - PIPELINE_START_EPOCH) / 60 ))

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  📊 EXPERIMENT RESULTS"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  📁 Training data:     $TRAINING_DATA"
echo "  🏆 Best SFT model:    ${BEST_SFT_MODEL:-N/A} (eval_loss=${BEST_SFT_LOSS:-N/A})"
echo "  🏆 Best GRPO model:   ${BEST_GRPO_MODEL:-N/A} (reward=${BEST_GRPO_REWARD:-N/A})"
echo "  📦 Deploy candidate:  ${DEPLOY_BEST:-N/A}"
echo "  📋 Logs:              $LOG_DIR/"
echo "  📊 W&B dashboard:     https://wandb.ai/${WANDB_ENTITY:-tinytimor}/${WANDB_PROJECT:-reachy-copilot}"
if [ "$PIPELINE_FAILURES" -gt 0 ]; then
echo "  ⚠️  Pipeline failures: $PIPELINE_FAILURES (check logs for details)"
else
echo "  ✅ Pipeline failures: 0"
fi
echo ""
echo "  📦 Quantized models:"
find "$MODEL_DIR" -name "model-*.gguf" -exec ls -lh {} \; 2>/dev/null | while read -r line; do
    echo "     $line"
done
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  🚀 NEXT STEPS"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  1. Check W&B dashboard to compare experiment metrics"
echo "     https://wandb.ai/${WANDB_ENTITY:-tinytimor}/${WANDB_PROJECT:-reachy-copilot}"
echo ""
echo "  2. Deploy best model to Orin Nano (if not auto-deployed):"
echo "     scp -r ${DEPLOY_BEST:-$MODEL_DIR/<best>-gguf-Q4_K_M}/ $ORIN_USER@$ORIN_IP:~/reachy-model/"
echo "     ssh $ORIN_USER@$ORIN_IP 'cd ~/reachy-model && ollama create reachy-copilot -f Modelfile'"
echo ""
echo "  3. Start services on Orin Nano:"
echo "     # Memory manager:"
echo "     ssh $ORIN_USER@$ORIN_IP 'cd ~/reachy-bridge && python3 05_memory_manager.py --serve &'"
echo "     # Bridge server:"
echo "     ssh $ORIN_USER@$ORIN_IP 'cd ~/reachy-bridge && python3 06_openclaw_bridge.py --standalone --memory-url http://localhost:8100'"
echo ""
echo "  4. Connect from Mac (VNC + SSH):"
echo "     See docs/REMOTE-ACCESS-MAC.md"
echo ""
echo "  5. Test end-to-end:"
echo "     curl -X POST http://$ORIN_IP:8000/chat -H 'Content-Type: application/json' -d '{\"message\": \"Hello Reachy!\"}'"
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Pipeline completed at $(date)"
echo "  Total duration: ${PIPELINE_DURATION} minutes"
echo "  Total log: $PIPELINE_LOG"
echo "═══════════════════════════════════════════════════════════"

# Send completion notification (visible on phone via W&B)
notify_completion "COMPLETE" "All experiments done. Best SFT: ${BEST_SFT_NAME:-N/A}, Best GRPO: ${BEST_GRPO_NAME:-N/A}, Deploy: ${DEPLOY_BEST:-N/A}"

log_success "🎉 All experiments complete! Check W&B to pick your best model."
