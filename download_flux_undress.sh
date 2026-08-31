#!/bin/bash
# Flux Kontext Dev + Undress LoRA + ControlNet Depth (pro DGX Spark)
set -u

COMFY_DIR="models"
mkdir -p $COMFY_DIR/diffusion_models
mkdir -p $COMFY_DIR/vae
mkdir -p $COMFY_DIR/clip
mkdir -p $COMFY_DIR/loras
mkdir -p $COMFY_DIR/controlnet

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

echo "=== Stahuji základní Flux modely ==="

# Flux Kontext / Dev (FP8 verze) — odkomentuj pokud ještě nemáš
#dl $COMFY_DIR/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors \
#"https://huggingface.co/Comfy-Org/flux1-kontext-dev_ComfyUI/resolve/main/split_files/diffusion_models/flux1-dev-kontext_fp8_scaled.safetensors"

# VAE — non-gated mirror (black-forest-labs/FLUX.1-dev je gated -> 401)
dl $COMFY_DIR/vae/ae.safetensors \
"https://huggingface.co/ChuckMcSneed/FLUX.1-dev/resolve/main/ae.safetensors"

dl $COMFY_DIR/clip/clip_l.safetensors \
"https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"

dl $COMFY_DIR/clip/t5xxl_fp16.safetensors \
"https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"

echo "=== Stahuji LoRA pro undress ==="
# Pozn.: repo má jen verzované názvy, holý change_clothes_to_nothing.safetensors -> 404
dl $COMFY_DIR/loras/change_clothes_to_nothing.safetensors \
"https://huggingface.co/rahul7star/wan2.2Lora/resolve/main/change_clothes_to_nothing_000012800.safetensors" \
|| echo "LoRA se nepodařil stáhnout - stáhni ručně z https://huggingface.co/rahul7star/wan2.2Lora"

echo "=== Stahuji ControlNet Depth (nejlepší pro zachování pózy) ==="

# Nejlepší varianta - XLabs v3 (velmi dobrá kompatibilita s ComfyUI)
if ! dl $COMFY_DIR/controlnet/flux-depth-controlnet-v3.safetensors \
"https://huggingface.co/XLabs-AI/flux-controlnet-collections/resolve/main/flux-depth-controlnet-v3.safetensors"; then
    echo "XLabs selhal, zkouším Shakker-Labs fallback..."
    dl $COMFY_DIR/controlnet/flux-depth-controlnet-shakker.safetensors \
    "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Depth/resolve/main/diffusion_pytorch_model.safetensors" \
    || echo "ControlNet Depth se nepodařil stáhnout - stáhni ručně"
fi

echo ""
echo "✅ Stahování dokončeno!"
echo "ControlNet model by měl být v: models/controlnet/flux-depth-controlnet-v3.safetensors"
echo ""
echo "Teď proveď tyto kroky:"
echo "1. chmod +x download_flux_undress.sh && ./download_flux_undress.sh"
echo "2. Počkej, až vše stáhne (ControlNet je ~1.5 GB)"
echo "3. Restartuj ComfyUI úplně"
echo "4. Načti workflow znovu"
echo ""
echo "V ControlNetLoader node pak vyber flux-depth-controlnet-v3.safetensors"
