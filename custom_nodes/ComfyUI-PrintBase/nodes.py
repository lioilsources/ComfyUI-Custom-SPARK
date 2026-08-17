"""AddPrintBase — přidá tisknutelnou podstavu pod vygenerovaný mesh.

Trellis z jednoho pohledu podstavu často zahodí (plochý disk pod postavou je
z frontálního záběru nejednoznačný), a i když ji vygeneruje, nemá kontrolu nad
rozměry ani rovinou dosedu. Tenhle node ji přidá deterministicky:

  1. volitelně uřízne spodek v dané výšce (rovná dosedací plocha)
  2. vytvoří válec / kvádr o zadaném průměru a výšce
  3. spojí ho s meshem boolean unionem (manifold3d) — výsledek zůstává
     watertight, což je podmínka spolehlivého sliceru

Průměr se zadává relativně k půdorysu modelu (1.0 = přesně opsaný kruh), takže
jedna hodnota sedí na figurky různých velikostí.
"""

import numpy as np
import trimesh


class AddPrintBase:
    CATEGORY = "mesh/print"
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "shape": (["cylinder", "box", "none"], {"default": "cylinder"}),
                "height_mm": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 50.0, "step": 0.5}),
                "diameter_scale": ("FLOAT", {"default": 1.15, "min": 0.5, "max": 3.0, "step": 0.05}),
                "flatten_bottom_mm": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.5}),
                "sink_mm": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "segments": ("INT", {"default": 96, "min": 12, "max": 256, "step": 4}),
                # STL jde ven v Z-up (Prusa), GLB náhled v Y-up (<model-viewer>).
                # Node musí umět obojí, aby podstava seděla v obou větvích.
                "up_axis": (["z", "y"], {"default": "z"}),
            }
        }

    def process(self, trimesh_input=None, **kw):
        mesh = kw.pop("trimesh", trimesh_input)
        shape = kw["shape"]
        height = float(kw["height_mm"])
        dia_scale = float(kw["diameter_scale"])
        flatten = float(kw["flatten_bottom_mm"])
        sink = float(kw["sink_mm"])
        segments = int(kw["segments"])
        up = 2 if kw.get("up_axis", "z") == "z" else 1

        if shape == "none" or height <= 0:
            return (mesh,)

        m = mesh.copy()

        # 1) Rovná dosedací plocha: uřízne vše pod z_min + flatten. Trellis dělá
        #    nohy zaoblené, takže model jinak stojí na dvou bodech.
        normal = [0.0, 0.0, 0.0]
        normal[up] = 1.0
        if flatten > 0:
            origin = [0.0, 0.0, 0.0]
            origin[up] = m.bounds[0][up] + flatten
            m = m.slice_plane(
                plane_origin=origin, plane_normal=normal, cap=True
            )
            if m is None or m.is_empty:
                # Řez by model zlikvidoval (flatten > výška) — vrať original.
                m = mesh.copy()

        lo, hi = m.bounds
        bottom = lo[up]
        # Osy kolmé na "nahoru" určují půdorys i střed podstavy.
        flat = [i for i in (0, 1, 2) if i != up]
        center = {i: (lo[i] + hi[i]) / 2.0 for i in flat}
        footprint = max(hi[i] - lo[i] for i in flat)
        radius = max(footprint * dia_scale / 2.0, 1e-3)

        # 2) Podstava POD model: horní podstava leží sink_mm nad dosedací
        #    rovinou modelu, takže vzniká reálný průnik (bez něj by union
        #    spojoval dvě tělesa dotýkající se přesně v rovině = degenerace),
        #    a figurka na podstavci stojí místo aby jí ho model spolkl.
        if shape == "cylinder":
            base = trimesh.creation.cylinder(
                radius=radius, height=height, sections=segments
            )
        else:
            base = trimesh.creation.box(
                extents=[radius * 2, radius * 2, height]
            )
        if up == 1:
            # trimesh primitivy stojí podél Z — pro Y-up je otočíme.
            base.apply_transform(
                trimesh.transformations.rotation_matrix(
                    np.radians(90), [1, 0, 0]
                )
            )
        offset = [0.0, 0.0, 0.0]
        for i in flat:
            offset[i] = center[i]
        offset[up] = bottom - height / 2.0 + sink
        base.apply_translation(offset)

        # 3) Union — manifold3d drží watertight; fallback na prosté spojení,
        #    které slicer taky zvládne (jen přestane platit is_watertight).
        try:
            out = trimesh.boolean.union([m, base], engine="manifold")
        except Exception as e:
            print(f"[AddPrintBase] boolean union selhal ({e}); "
                  f"spojuji bez booleanu")
            out = trimesh.util.concatenate([m, base])

        if out is None or out.is_empty:
            print("[AddPrintBase] union vrátil prázdný mesh, vracím originál")
            return (mesh,)

        # Posadit dosedací rovinu na nulu, ať slicer nemusí nic dorovnávat.
        drop = [0.0, 0.0, 0.0]
        drop[up] = -out.bounds[0][up]
        out.apply_translation(drop)
        print(f"[AddPrintBase] podstava {shape} ⌀{radius*2:.1f} × {height:.1f} mm | "
              f"watertight={out.is_watertight} | faces={len(out.faces)}")
        return (out,)


NODE_CLASS_MAPPINGS = {"AddPrintBase": AddPrintBase}
NODE_DISPLAY_NAME_MAPPINGS = {"AddPrintBase": "Add Print Base (podstava)"}
