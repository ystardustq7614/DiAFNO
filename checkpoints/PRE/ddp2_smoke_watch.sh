#!/usr/bin/env bash
# DDP2 smoke watcher (runs inside tmux; survives SSH disconnects).
#
# Polls GPU memory every 5 min. When two GPUs each have >=19.5 GiB free,
# launches the MS5 DDP2 smoke on them (foreground within this script — the
# tmux session is the durable container). Max 3 attempts, 30 min cooldown
# after each failure, 5 h total budget. All state changes are appended to
# ddp2_smoke_watch.status so progress is checkable from anywhere.
#
# Per-rank peak memory is ~17.5-20 GiB (measured 2026-09-02), hence the
# threshold; sharing a card with a resident job >4.5 GiB WILL OOM.
set -u

ROOT=/data2/user/zyq/projects/DiAFNO
CKPT=/data2/user/zyq/checkpoints/PRE
STATUS=$CKPT/ddp2_smoke_watch.status
RUN_DIR=$CKPT/surface_smoke_BS4_EMD180_I4_E4_S4_C7_SD2_RES_MS5_SMOKE_DDP2
INIT_CKPT=$CKPT/surface_smoke_BS4_EMD180_I4_E4_S32_C7_SD2_RES/Ep10.pth
TOTAL_MIB=24564
THRESHOLD_MIB=19968          # required FREE MiB per GPU
MAX_ATTEMPTS=3
DEADLINE=$(( $(date +%s) + 5 * 3600 ))

log() { echo "$(date '+%F %T') $*" >> "$STATUS"; }

log "watcher started (need 2 GPUs with free >= ${THRESHOLD_MIB} MiB; max ${MAX_ATTEMPTS} attempts; 5h budget)"

attempt=0
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    # defensive stop: a previous attempt that wrote checkpoints would make the
    # trainer's pre-flight refuse; that needs manual review, not auto-retry
    if ls "$RUN_DIR"/Ep*.pth >/dev/null 2>&1; then
        log "watcher stopped: $RUN_DIR contains checkpoints from a previous attempt - manual review required"
        exit 2
    fi

    free_gpus=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', *' -v t="$THRESHOLD_MIB" -v total="$TOTAL_MIB" \
              '$2 + 0 <= total - t {printf "%s ", $1}')

    set -- $free_gpus
    if [ "$#" -ge 2 ]; then
        A="$1"; B="$2"
        sleep 30    # re-verify: guard against a card being grabbed in between
        still=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
            | awk -F', *' -v t="$THRESHOLD_MIB" -v total="$TOTAL_MIB" \
                  '$2 + 0 <= total - t {printf " %s ", $1}')
        if echo "$still" | grep -q " $A " && echo "$still" | grep -q " $B "; then
            attempt=$((attempt + 1))
            TRYLOG=$CKPT/train_ms5_smoke_ddp2_$(date +%Y%m%d_%H%M%S).log
            log "attempt${attempt}: launching DDP2 smoke on GPU ${A},${B} (log: ${TRYLOG})"
            cd "$ROOT" || { log "fatal: cannot cd $ROOT"; exit 3; }
            CUDA_VISIBLE_DEVICES=${A},${B} \
            DIAFNO_PRESET=surface_smoke \
            DIAFNO_OBJECTIVE=persistence_residual \
            DIAFNO_TRAIN_HORIZON=5 \
            DIAFNO_INIT_CHECKPOINT="$INIT_CKPT" \
                torchrun --standalone --nproc_per_node=2 pre_trainer.py \
                > "$TRYLOG" 2>&1
            rc=$?
            if grep -q "SMOKE PASS" "$TRYLOG"; then
                log "attempt${attempt}: SMOKE PASS (rc=$rc) on GPU ${A},${B}"
                log "run dir: $RUN_DIR"
                log "watcher finished: SUCCESS"
                exit 0
            fi
            reason=$(grep -m1 -oE "OutOfMemoryError[^\"]{0,120}|SMOKE FAIL[^\"]{0,120}|[A-Za-z]+Error[^\"]{0,120}" "$TRYLOG" | head -1)
            log "attempt${attempt}: FAILED (rc=$rc) ${reason:-see $TRYLOG}"
            if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
                log "watcher finished: GAVE UP after $attempt attempts"
                exit 1
            fi
            log "cooldown 30 min before next attempt"
            sleep 1800
        else
            log "GPUs ${A},${B} freed then re-occupied within 30 s; re-polling"
        fi
    fi
    sleep 300
done

log "watcher finished: TIMEOUT (5 h budget exhausted, attempts used: $attempt)"
exit 4
