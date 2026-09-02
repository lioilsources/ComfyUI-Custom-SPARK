"""Mesh nody pro tisk: AddPrintBase (podstava) a SealMeshTunnels (tunely).

AddPrintBase — přidá tisknutelnou podstavu pod vygenerovaný mesh.

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


class SealMeshTunnels:
    """Zavře tunely v meshi CGAL alpha wrappingem.

    Trellis z jednoho pohledu domýšlí odvrácenou stranu a přitom model
    „prokopne" — vzniknou průchody skrz tělo. Mesh přitom zůstane watertight
    a bez otevřených hran, takže ho `fill_holes` ani kontrola watertight
    nezachytí: vada je topologická (rod), ne díra v povrchu. Naměřeno
    2. 9. 2026 na figurce: watertight True, 0 otevřených hran, ale **rod 23**.

    Alpha wrapping objede model zvenčí koulí o poloměru `alpha` a vytvoří
    nový obal. Co je užší než koule (tenký tunel), se zavře; tvar a detaily
    zůstanou. Na téže figurce: rod 23 → 9 při zachování 222k trojúhelníků,
    zatímco morfologické uzavření dalo sice rod 7, ale rozmazalo obličej.

    Nezavře velké průchody — ty jsou buď legitimní (mezera mezi pažemi), nebo
    je model zkrátka vymyslel a žádný post-processing chybějící část
    nedomyslí; tam pomůže jen lepší vstupní fotka (zepředu: rod 7 místo 23).

    Pouští se PŘED AddPrintBase, aby se neobalovala podstava.
    """

    CATEGORY = "mesh/print"
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                # % z úhlopříčky bboxu; níž = jemnější detail, ale víc tunelů přežije
                "alpha": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 5.0, "step": 0.05}),
                "offset": ("FLOAT", {"default": 0.3, "min": 0.01, "max": 2.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": True}),
            }
        }

    @staticmethod
    def _genus(m):
        try:
            return (2 - int(m.euler_number)) // 2
        except Exception:
            return None

    def process(self, trimesh_input=None, **kw):
        # `trimesh` je zároveň jméno modulu, takže stejná obezlička jako výš
        import tempfile, os
        mesh = kw.pop("trimesh", trimesh_input)
        alpha, offset, enabled = kw.get("alpha", 0.3), kw.get("offset", 0.3), kw.get("enabled", True)
        before = self._genus(mesh)
        if not enabled:
            print(f"[SealMeshTunnels] vypnuto (rod={before})")
            return (mesh,)
        try:
            import pymeshlab
        except ImportError:
            print("[SealMeshTunnels] pymeshlab chybí, vracím mesh beze změny")
            return (mesh,)

        tmp = tempfile.mkdtemp(prefix="sealtunnels_")
        src, dst = os.path.join(tmp, "in.ply"), os.path.join(tmp, "out.ply")
        try:
            mesh.export(src)
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(src)
            ms.generate_alpha_wrap(alpha=pymeshlab.PercentageValue(alpha),
                                   offset=pymeshlab.PercentageValue(offset))
            ms.save_current_mesh(dst)
            out = trimesh.load(dst, force="mesh")
        except Exception as e:                                  # noqa: BLE001
            print(f"[SealMeshTunnels] alpha wrap selhal ({e}); vracím originál")
            return (mesh,)
        finally:
            for f in (src, dst):
                try:
                    os.remove(f)
                except OSError:
                    pass
            try:
                os.rmdir(tmp)
            except OSError:
                pass

        if out is None or out.is_empty:
            print("[SealMeshTunnels] výsledek prázdný, vracím originál")
            return (mesh,)
        # Příliš velká alpha model „obalí“ do beztvarého pytle: koule se přestane
        # vejít do detailů a zůstane hrubá skořápka (alpha 5 % dalo z 297k ploch
        # 1398). Pod pětinu původní hustoty už to není opravený model.
        if len(out.faces) < 0.2 * len(mesh.faces):
            print(f"[SealMeshTunnels] alpha {alpha} je na tenhle mesh příliš hrubá "
                  f"({len(mesh.faces)} → {len(out.faces)} faces), vracím originál")
            return (mesh,)
        # Alpha wrap pracuje v jednotkách vstupu, takže měřítko musí sedět;
        # kdyby se rozešlo, radši originál než mikroskopická figurka.
        ratio = out.extents.max() / max(mesh.extents.max(), 1e-9)
        if not 0.8 <= ratio <= 1.25:
            print(f"[SealMeshTunnels] měřítko ujelo ({ratio:.2f}×), vracím originál")
            return (mesh,)
        print(f"[SealMeshTunnels] rod {before} → {self._genus(out)} | "
              f"faces {len(mesh.faces)} → {len(out.faces)} | "
              f"watertight={out.is_watertight}")
        return (out,)


NODE_CLASS_MAPPINGS = {"AddPrintBase": AddPrintBase, "SealMeshTunnels": SealMeshTunnels}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AddPrintBase": "Add Print Base (podstava)",
    "SealMeshTunnels": "Seal Mesh Tunnels (alpha wrap)",
}
