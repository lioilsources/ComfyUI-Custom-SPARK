# ComfyUI-Custom-SPARK

<!-- Max 80 řádků. Co se nevejde, jde do docs/ a sem jeden odkaz. Veřejné repo: žádné LAN IP, domovské cesty, ID. -->

## Účel
Vlastní kopie upstream ComfyUI (v0.19.3) pro SPARK (DGX Spark, GB10, sdílená unified paměť s LLM).
Nad upstreamem: launch config, patche pro SPARK a vlastní custom nodes.

## Build a test
Na SPARKu se spouští `./run.sh` (aktivuje `.venv`, flagy `--reserve-vram`, `--cache-lru 2`, SageAttention).
Přepínače: `RESERVE_VRAM=12 ./run.sh`, `SAGE=0 ./run.sh`, extra flagy se připojí.
Upstream testy: `.venv/bin/pytest tests-unit` (pro vlastní nodes testy nejsou).

## Mapa (vlastní změny nad upstreamem)
- `run.sh`, `comfyui.service` – launcher a systemd unit pro SPARK
- `comfy/model_patcher.py` – patch pro SPARK (unpin_weight)
- `custom_nodes/ComfyUI-Trellis2/` – Trellis2 s patchi kompatibility (viz `Prompts/01-FIX-TRELLIS2.md`)
- `custom_nodes/ComfyUI-PrintBase/` – podstava a uzavírání tunelů pro tisknutelné meshe
- `download_flux_kontext.sh` – stažení Flux Kontext Dev do `models/`
- `.gitignore` – `custom_nodes/*` je ignorováno, vlastní nodes mají výjimku `!custom_nodes/<název>/`

## Invarianty
- `--reserve-vram` musí být menší než volná GPU paměť, jinak ComfyUI offloaduje vše na CPU (viz komentář v `run.sh`).
- Vlastní nodes přidávej do `.gitignore` výjimkou, jinak se necommitnou.

## Kam dál
`run.sh` (komentáře k flagům), `Prompts/`, upstream README.md
