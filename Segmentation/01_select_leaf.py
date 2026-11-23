"""
01_select_leaf.py
Selecciona imágenes de HOJAS de Pl@ntNet-300k (organ == "leaf"), limita el
número por especie y monta la estructura de dataset que espera YOLO:

    dataset/
      images/{train,val,test}/<PN_hash>.jpg
      labels/{train,val,test}/          (vacío; lo rellena 02_autolabel.py)
      manifest.csv
      gold_val.txt, gold_test.txt       (imágenes que corregirás a mano)
      review_images/<split>__<PN_hash>.jpg   (copia de las 'gold' para CVAT/Roboflow)

Estructura de entrada esperada:  images/<split>/<species_id>/<PN_hash>.jpg

Uso:
    python 01_select_leaf.py --metadata plantnet300K_metadata.csv --images /ruta/a/images
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test")
SUFFIXES = (".jpg", ".jpeg", ".JPG", ".JPEG")


FOLDER_FORMATS = ("{}", "{:04d}", "{:03d}", "{:05d}")


def detect_folder_format(images_root: Path) -> str:
    """Las carpetas de especie pueden llamarse '0' o '0000' (con ceros a la izquierda)."""
    for fmt in FOLDER_FORMATS:
        for split in SPLITS:
            if (images_root / split / fmt.format(0)).is_dir():
                return fmt
    return "{}"


def find_file(images_root: Path, split: str, species_id: int, pn_hash: str, fmt: str = "{}"):
    folder = images_root / split / fmt.format(species_id)
    for suf in SUFFIXES:
        p = folder / f"{pn_hash}{suf}"
        if p.exists():
            return p
    return None


def describe_layout(images_root: Path, row, fmt: str = "{}") -> str:
    """Texto de diagnóstico cuando no se encuentra ninguna imagen del CSV."""
    folder = images_root / row.split / fmt.format(row.species_id)
    lines = [f"Para la primera fila del CSV esperaba:\n  {folder / (row.PN_hash + '.jpg')}"]
    if folder.is_dir():
        names = sorted(p.name for p in folder.iterdir())[:5]
        lines.append(f"La carpeta existe, pero contiene (ejemplos): {names}")
        lines.append("-> Los nombres de archivo no coinciden con la columna PN_hash del CSV.")
    else:
        split_dir = images_root / row.split
        if split_dir.is_dir():
            names = sorted(p.name for p in split_dir.iterdir())[:5]
            lines.append(f"La carpeta no existe. '{split_dir}' contiene (ejemplos): {names}")
        else:
            names = sorted(p.name for p in images_root.iterdir())[:5]
            lines.append(f"'{split_dir}' no existe. '{images_root}' contiene: {names}")
    return "\n".join(lines)


def prepare_output(out: Path, force: bool):
    """Evita mezclar imágenes de ejecuciones anteriores con otra semilla o límites."""
    if out.exists() and any(out.iterdir()):
        if not force:
            raise SystemExit(
                f"'{out}' ya existe y no está vacío. Usa --force para regenerarlo "
                "(se borrará, incluidas etiquetas y correcciones manuales)."
            )
        if not (out / "manifest.csv").exists():
            raise SystemExit(
                f"'{out}' no parece un dataset creado por este script (falta manifest.csv). "
                "Por seguridad no lo borro: elige otra carpeta con --out."
            )
        shutil.rmtree(out)
    for split in SPLITS:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "review_images").mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="plantnet300K_metadata.csv")
    ap.add_argument("--images", required=True, help="carpeta 'images' con train/val/test")
    ap.add_argument("--out", default="dataset", help="carpeta de salida")
    ap.add_argument("--organ", default="leaf")
    ap.add_argument("--max-train", type=int, default=6, help="máx. imágenes por especie en train")
    ap.add_argument("--max-val", type=int, default=2)
    ap.add_argument("--max-test", type=int, default=2)
    ap.add_argument("--gold-per-split", type=int, default=100,
                    help="imágenes de val y de test que corregirás a mano")
    ap.add_argument("--link", action="store_true",
                    help="symlinks en vez de copiar (Linux/Mac; en Windows suele fallar)")
    ap.add_argument("--force", action="store_true", help="regenerar --out si ya existe")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    images_root = Path(args.images)
    if not images_root.is_dir():
        raise SystemExit(f"No existe la carpeta de imágenes: {images_root}")
    # Tolera el caso típico de descomprimir y quedar images/images/train
    for cand in (images_root, images_root / "images"):
        if (cand / "train").is_dir():
            if cand != images_root:
                print(f"Nota: usando '{cand}' (carpeta anidada).")
            images_root = cand
            break
    else:
        found = sorted(p.name for p in images_root.iterdir())[:10]
        raise SystemExit(
            f"No encuentro la carpeta 'train' dentro de '{args.images}'. Contiene: {found}"
        )
    out = Path(args.out).resolve()
    caps = {"train": args.max_train, "val": args.max_val, "test": args.max_test}

    meta = pd.read_csv(args.metadata)
    sel = meta[(meta.organ == args.organ) & (meta.split.isin(SPLITS))]
    print(f"Imágenes con organ == '{args.organ}': {len(sel)}")

    # Localizar en disco ANTES de muestrear, para no perder cupo por archivos ausentes.
    fmt = detect_folder_format(images_root)
    print(f"Carpetas de especie con formato '{fmt.format(7)}' (p. ej. especie 7).")
    sel = sel.assign(src=[find_file(images_root, r.split, r.species_id, r.PN_hash, fmt)
                          for r in sel.itertuples()])
    missing = int(sel["src"].isna().sum())
    if missing == len(sel):
        raise SystemExit(
            "No se encontró ninguna imagen del CSV en disco.\n"
            + describe_layout(images_root, next(sel.itertuples()), fmt)
        )
    if missing:
        print(f"AVISO: {missing} imágenes del CSV no están en disco; se ignoran.")
    sel = sel.dropna(subset=["src"])

    # Muestreo: barajar y quedarse con las primeras N por especie, dentro de cada split.
    parts = []
    for split, g in sel.groupby("split"):
        g = g.sample(frac=1, random_state=args.seed)
        parts.append(g.groupby("species_id").head(caps[split]))
    sub = pd.concat(parts, ignore_index=True)

    # Conjunto 'gold' en val y test.
    sub["gold"] = False
    for split in ("val", "test"):
        pool = sub[sub.split == split]
        idx = pool.sample(n=min(args.gold_per_split, len(pool)), random_state=args.seed).index
        sub.loc[idx, "gold"] = True

    prepare_output(out, args.force)

    for r in sub.itertuples():
        dst = out / "images" / r.split / f"{r.PN_hash}.jpg"
        if args.link:
            dst.symlink_to(Path(r.src).resolve())
        else:
            shutil.copy2(r.src, dst)
        if r.gold:
            shutil.copy2(r.src, out / "review_images" / f"{r.split}__{r.PN_hash}.jpg")

    for split in ("val", "test"):
        lines = [str(out / "images" / split / f"{h}.jpg")
                 for h in sub[(sub.split == split) & sub.gold].PN_hash]
        (out / f"gold_{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sub.drop(columns=["src"]).to_csv(out / "manifest.csv", index=False)

    print("\nResumen:")
    print(sub.split.value_counts().reindex(SPLITS).to_string())
    print(f"Especies distintas: {sub.species_id.nunique()}")
    print(f"Imágenes gold: {int(sub.gold.sum())} (en {out / 'review_images'})")
    print(f"Dataset en: {out}")


if __name__ == "__main__":
    main()