# Fix: ComfyUI-Trellis2 compatibility patches

Fixes for `custom_nodes/ComfyUI-Trellis2` — all errors were caused by incompatibilities
between the library versions installed in the venv and the plugin code.

---

## 1. `DINOv3ViTModel` has no attribute `layer`

**File:** `trellis2/modules/image_feature_extractor.py:86`

**Error:** `AttributeError: 'DINOv3ViTModel' object has no attribute 'layer'`

**Cause:** `DINOv3ViTModel` (HuggingFace transformers) stores its encoder as `self.model`
(a `DINOv3ViTEncoder` instance), which has `.layer`. The code was calling `self.model.layer`
instead of `self.model.model.layer`.

**Fix:**
```python
# before
for i, layer_module in enumerate(self.model.layer):
# after
for i, layer_module in enumerate(self.model.model.layer):
```

---

## 2. `flash_attn` not installed — full attention backend

**Files:**
- `trellis2/modules/attention/config.py` — hardcoded default `BACKEND = 'flash_attn'`
- `custom_nodes/ComfyUI-Trellis2/nodes.py:328` — UI default `"flash_attn"` for the `backend` parameter

**Error:** `ModuleNotFoundError: No module named 'flash_attn'`

**Cause:** Config defaulted to `flash_attn`, but only PyTorch's built-in `sdpa` is available
in the venv. Additionally, the node explicitly called `config.set_backend(backend)` in its
`process()` method, overriding any auto-detection.

**Fix `config.py`:** Replaced hardcoded default with a `_detect_backend()` function that
tries `flash_attn_3 → flash_attn → xformers → sdpa` and picks the first available.

**Fix `nodes.py`:** Changed order and default in the dropdown:
```python
# before
["flash_attn","xformers","sdpa","flash_attn_3"], default="flash_attn"
# after
["sdpa","flash_attn","xformers","flash_attn_3"], default="sdpa"
```

---

## 3. `flash_attn` not installed — sparse windowed attention

**Files:**
- `trellis2/modules/sparse/config.py` — hardcoded default `ATTN = 'flash_attn'`
- `trellis2/modules/sparse/attention/windowed_attn.py` — no `sdpa` branch
- `nodes.py:333` — UI default `"flash_attn"` for `sparse_backend`

**Error:** Same `ModuleNotFoundError: No module named 'flash_attn'`, but for sparse ops.

**Cause:** Sparse attention had its own config and UI parameter, both defaulting to
`flash_attn`. The `windowed_attn.py` functions had no `sdpa` branch.

**Fix `sparse/config.py`:** Same auto-detect pattern as for full attention
(`flash_attn → xformers → sdpa`).

**Fix `windowed_attn.py`:** Added `sdpa` branch to `calc_window_partition`,
`sparse_windowed_scaled_dot_product_self_attention`, and
`sparse_windowed_scaled_dot_product_cross_attention`. Because `sdpa` does not natively
support variable-length sequences, the implementation loops over windows individually:
```python
for seq_len in attn_func_args['seq_lens']:
    # process each window separately via torch.nn.functional.scaled_dot_product_attention
```

**Fix `nodes.py`:**
```python
# before
["xformers","flash_attn"], default="flash_attn"
# after
["sdpa","xformers","flash_attn"], default="sdpa"
```

---

## 4. `tiled_flexible_dual_grid_to_mesh` missing from `o_voxel`

**File:** `trellis2/models/sc_vaes/fdg_vae.py:21`

**Error:** `ImportError: cannot import name 'tiled_flexible_dual_grid_to_mesh' from 'o_voxel.convert'`

**Cause:** The installed version of `o_voxel` does not include `tiled_flexible_dual_grid_to_mesh`.
The function does the same thing as `flexible_dual_grid_to_mesh` but processes the mesh
in tiles to reduce VRAM usage.

**Fix:** Graceful fallback at import time — if the function is missing, a shim is defined:
```python
try:
    from o_voxel.convert import tiled_flexible_dual_grid_to_mesh
except ImportError:
    def tiled_flexible_dual_grid_to_mesh(coords, dual_vertices, intersected_flag,
                                          split_weight, aabb, grid_size,
                                          tile_size=128, train=False):
        return flexible_dual_grid_to_mesh(coords=coords, dual_vertices=dual_vertices,
            intersected_flag=intersected_flag, split_weight=split_weight,
            aabb=aabb, grid_size=grid_size, train=train)
```
Note: the shim ignores `tile_size` — the mesh is processed all at once, so VRAM usage is higher.

---

## 5. `reconstruct_mesh_dc_quad`, `reconstruct_mesh_dc`, `remesh_narrow_band_dc_quad` missing from `cumesh`

**File:** `nodes.py` (lines 2202, 2240, 1756, 3204)

**Error:** `AttributeError: module 'cumesh.remeshing' has no attribute 'reconstruct_mesh_dc_quad'`

**Cause:** The installed version of `cumesh.remeshing` only contains `remesh_narrow_band_dc`.
The newer plugin code calls three additional functions that are absent.

**Fix:** Shims added to `nodes.py` immediately after `import cumesh as CuMesh`:

- `reconstruct_mesh_dc(vertices, faces, resolution, verbose)` — auto-computes
  `center`/`scale` from the bounding box, delegates to `remesh_narrow_band_dc`
- `reconstruct_mesh_dc_quad(vertices, faces, resolution, verbose, remove_inner_faces)` —
  same; `remove_inner_faces` is ignored (not supported)
- `remesh_narrow_band_dc_quad(vertices, faces, center, scale, resolution, ...)` —
  delegates to `remesh_narrow_band_dc`; `remove_inner_faces` is ignored

All shims are guarded with `if not hasattr(...)` so they won't apply if a future
version of `cumesh` adds these functions.

---

## Dependency status (venv state)

| Library | Available | Missing |
|---|---|---|
| `flash_attn` | no | yes |
| `flash_attn_interface` (v3) | no | yes |
| `xformers` | no | yes |
| `torch.nn.functional.scaled_dot_product_attention` | **yes** | — |
| `o_voxel.convert.tiled_flexible_dual_grid_to_mesh` | no | yes |
| `cumesh.remeshing.remesh_narrow_band_dc` | **yes** | — |
| `cumesh.remeshing.reconstruct_mesh_dc` | no | yes |
| `cumesh.remeshing.reconstruct_mesh_dc_quad` | no | yes |
| `cumesh.remeshing.remesh_narrow_band_dc_quad` | no | yes |
