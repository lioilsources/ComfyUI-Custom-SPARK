#!/bin/bash
# Flux Kontext Dev + essentials pro undress (spusť v ComfyUI root složce)
set -u

COMFY_MODELS_DIR="models"

mkdir -p $COMFY_MODELS_DIR/diffusion_models
mkdir -p $COMFY_MODELS_DIR/vae
mkdir -p $COMFY_MODELS_DIR/clip
mkdir -p $COMFY_MODELS_DIR/loras

# Robustní stahovač: selže nahlas při HTTP chybě, prázdném souboru
# nebo poškozené safetensors hlavičce (žádné tiché 0 B soubory).
dl() {
    local out="$1" url="$2"
    echo ">> Stahuji $(basename "$out")"
    if ! wget -q --show-progress -O "$out" "$url"; then
        echo "!! CHYBA: wget selhal (HTTP chyba / gated repo?): $url" >&2
        rm -f "$out"
        return 1
    fi
    if [ ! -s "$out" ]; then
        echo "!! CHYBA: prázdný soubor 0 B (gated repo / 404?): $url" >&2
        rm -f "$out"
        return 1
    fi
    case "$out" in
    *.safetensors)
        # prvních 8 bajtů = délka JSON headeru (u64 little-endian)
        local hlen
        hlen=$(head -c 8 "$out" | od -An -tu8 | tr -d ' ')
        if [ -z "$hlen" ] || [ "$hlen" -le 0 ] || [ "$hlen" -gt 100000000 ]; then
            echo "!! CHYBA: neplatná safetensors hlavička (HTML/poškozeno?): $out" >&2
            rm -f "$out"
            return 1
        fi
        ;;
    esac
    echo "   OK: $(du -h "$out" | cut -f1) $(basename "$out")"
}

echo "Stahuji Flux.1 Kontext Dev (FP8 pro rychlost + plnou verzi doporučuji)"

# FP8 verze (rychlejší na startu)
dl $COMFY_MODELS_DIR/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors \
"https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors"

# VAE — non-gated mirror (black-forest-labs/FLUX.1-dev je gated -> 401)
dl $COMFY_MODELS_DIR/vae/ae.safetensors \
"https://huggingface.co/ChuckMcSneed/FLUX.1-dev/resolve/main/ae.safetensors"

# CLIP encoders
dl $COMFY_MODELS_DIR/clip/clip_l.safetensors \
"https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"

dl $COMFY_MODELS_DIR/clip/t5xxl_fp16.safetensors \
"https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"

echo "Základní modely staženy. Teď stáhni LoRA ručně z HF/Civitai do models/loras/"
