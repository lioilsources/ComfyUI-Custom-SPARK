#!/usr/bin/env bash
# ComfyUI launcher pro DGX Spark (GB10, sdílená unified paměť s vLLM dev).
#
# --reserve-vram nechává místo pro LLM (ai-dev ~drží svoje váhy+KV) + OS.
# MUSÍ být < aktuálně volné GPU paměti (viz /system_stats vram_free), jinak
# ComfyUI offloaduje všechno na CPU ("0 MB usable"). 8 GB je bezpečný polštář.
#
# --use-sage-attention: SageAttention (sm_121 build v .venv) — bez něj jede
#   pytorch sdpa, na video difuzi řádově pomalejší.
# --cache-lru 2: drží poslední dva výsledky nodů, tedy prakticky dva
#   checkpointy (UGC továrna střídá Illustrious a Juggernaut). Dřív tu bylo
#   --cache-none, které šetřilo paměť, ale nutilo načíst model z NVMe při
#   KAŽDÉM běhu - to byl po nasazení SF3D hlavní zbytek času na jeden kus.
#   LRU 2 je kompromis: zrychlí opakované běhy a nemůže vyčerpat unified
#   paměť tak, jako by to udělala neomezená cache (FLUX 24 GB + TRELLIS 16 GB).
# expandable_segments: méně fragmentace CUDA alokátoru na dlouhých video bězích.
#
# Použití:  ./run.sh                 # default reserve-vram 8
#           RESERVE_VRAM=12 ./run.sh # přepiš polštář
#           SAGE=0 ./run.sh          # nouzově vypni SageAttention
#           ./run.sh --highvram      # extra flagy se přidají
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
FLAGS=(--listen --reserve-vram "${RESERVE_VRAM:-8}" --disable-async-offload --disable-pinned-memory --cache-lru 2)
[ "${SAGE:-1}" = 1 ] && FLAGS+=(--use-sage-attention)
exec python main.py "${FLAGS[@]}" "$@"
