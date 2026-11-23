"""
04_preparar_cvat.py   (va dentro de Segmentation/)
-------------------
Empaqueta las imágenes 'gold' (review_images/) y sus pre-anotaciones automáticas
(review_labels/) en un .zip con el formato "Ultralytics YOLO Segmentation 1.0",
listo para importarlo en CVAT y corregirlo a mano.

Uso:
    python Segmentation/04_preparar_cvat.py --root Segmentation/dataset_mini

Genera:  <root>/cvat_import.zip
"""
import argparse
import zipfile
from pathlib import Path

SEG = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset_mini",
                    help="carpeta del dataset (la misma que usaste en 01/02)")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.is_dir() and (SEG / a.root).is_dir():
        root = SEG / a.root
    imgs_dir, lbls_dir = root / "review_images", root / "review_labels"
    if not imgs_dir.is_dir():
        raise SystemExit(f"No existe {imgs_dir}. ¿Has ejecutado 01_select_leaf.py con este --root?")

    imagenes = sorted(imgs_dir.glob("*.jpg"))
    if not imagenes:
        raise SystemExit(f"{imgs_dir} está vacía.")

    destino = root / "cvat_import.zip"
    con_pre, sin_pre = 0, []
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        # nombre de la clase = nombre de la etiqueta que crearás en CVAT
        z.writestr("data.yaml", "path: ./\ntrain: train.txt\nnames:\n  0: leaf\n")
        z.writestr("train.txt", "".join(f"images/train/{p.name}\n" for p in imagenes))
        for p in imagenes:
            z.write(p, f"images/train/{p.name}")
            lbl = lbls_dir / f"{p.stem}.txt"
            if lbl.exists() and lbl.stat().st_size > 0:
                z.write(lbl, f"labels/train/{p.stem}.txt")
                con_pre += 1
            else:
                z.writestr(f"labels/train/{p.stem}.txt", "")   # sin pre-anotación: dibujar a mano
                sin_pre.append(p.name)

    print(f"Zip creado: {destino}")
    print(f"Imágenes: {len(imagenes)} | con pre-anotación: {con_pre} | "
          f"sin pre-anotación (habrá que dibujarlas): {len(sin_pre)}")


if __name__ == "__main__":
    main()