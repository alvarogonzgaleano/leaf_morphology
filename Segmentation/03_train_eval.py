"""
03_train_eval.py
Tres modos:

1) Importar las correcciones manuales del conjunto gold (export YOLO-seg de CVAT/Roboflow):
    python 03_train_eval.py --root dataset --import-gold ruta/a/export
   Busca recursivamente .txt cuyo nombre contenga <split>__<PN_hash> (el nombre de
   review_images/), valida el formato y sobrescribe dataset/labels/<split>/<PN_hash>.txt.

2) Entrenar (solo con imágenes que tienen etiqueta):
    python 03_train_eval.py --root dataset --model yolo11s-seg.pt --epochs 100 --device 0

3) Evaluar contra el gold de test YA CORREGIDO:
    python 03_train_eval.py --root dataset --eval runs/segment/leaf/weights/best.pt
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")
GOLD_RE = re.compile(r"(train|val|test)__([0-9a-f]{40})")


# ---------------------------------------------------------------- utilidades
def labelled_images(root: Path, split: str) -> list[str]:
    """Imágenes con etiqueta no vacía. Sin etiqueta, YOLO las trataría como
    'fondo sin hojas', lo que es falso en fotos de hojas."""
    out = []
    for p in sorted((root / "images" / split).glob("*.jpg")):
        lbl = root / "labels" / split / f"{p.stem}.txt"
        if lbl.exists() and lbl.stat().st_size > 0:
            out.append(str(p.resolve()))
    return out


def write_list(path: Path, items: list[str]):
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")


def write_yaml(path: Path, train: Path, val: Path):
    path.write_text(
        f"train: {train.resolve().as_posix()}\n"
        f"val: {val.resolve().as_posix()}\n"
        "names:\n  0: leaf\n",
        encoding="utf-8",
    )


def clear_label_caches(root: Path):
    for c in (root / "labels").glob("*.cache"):
        c.unlink()


def read_imported(root: Path) -> set[str]:
    f = root / "gold_imported.txt"
    return set(f.read_text().split()) if f.exists() else set()


def validate_label(text: str) -> tuple[list[str], str | None]:
    """Devuelve (líneas normalizadas a clase 0, error o None)."""
    lines = []
    for n, raw in enumerate(text.splitlines(), 1):
        parts = raw.split()
        if not parts:
            continue
        try:
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            return [], f"línea {n}: valores no numéricos"
        if len(vals) < 6 or len(vals) % 2:
            return [], f"línea {n}: no es un polígono (¿exportaste cajas en vez de segmentación?)"
        if any(v < -1e-6 or v > 1 + 1e-6 for v in vals):
            return [], f"línea {n}: coordenadas fuera de [0, 1] (¿sin normalizar?)"
        lines.append("0 " + " ".join(f"{min(max(v, 0.0), 1.0):.5f}" for v in vals))
    return lines, None


def import_gold(root: Path, src_dir: Path):
    imported = read_imported(root)
    ok, errors, skipped, remapped = 0, [], 0, 0
    for f in sorted(src_dir.rglob("*.txt")):
        m = GOLD_RE.search(f.stem)  # tolera sufijos de Roboflow tipo _jpg.rf.xxxx
        if not m:
            skipped += 1
            continue
        split, h = m.groups()
        if not (root / "images" / split / f"{h}.jpg").exists():
            errors.append(f"{f.name}: la imagen no está en el dataset")
            continue
        text = f.read_text(encoding="utf-8")
        if any(l.split()[0] != "0" for l in text.splitlines() if l.strip()):
            remapped += 1
        lines, err = validate_label(text)
        if err:
            errors.append(f"{f.name}: {err}")
            continue
        if not lines:
            errors.append(f"{f.name}: vacío (¿imagen sin hojas? revísala)")
            continue
        (root / "labels" / split / f"{h}.txt").write_text("\n".join(lines) + "\n")
        imported.add(f"{split}__{h}")
        ok += 1
    write_list(root / "gold_imported.txt", sorted(imported))
    clear_label_caches(root)
    print(f"Importadas: {ok} | ignoradas (nombre no reconocido): {skipped} | errores: {len(errors)}")
    if remapped:
        print(f"AVISO: {remapped} archivos tenían otra clase; se ha puesto clase 0 (leaf).")
    for e in errors[:20]:
        print("  -", e)
    print(f"Total gold corregido acumulado: {len(imported)}")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--import-gold", default=None, help="carpeta con el export corregido")
    ap.add_argument("--eval", default=None, help="ruta a best.pt para evaluar en el gold de test")
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = automático según la GPU")
    ap.add_argument("--workers", type=int, default=4, help="bájalo a 0-2 si en Windows da errores")
    ap.add_argument("--device", default=None)
    ap.add_argument("--name", default="leaf")
    args = ap.parse_args()

    root = Path(args.root).resolve()

    if args.import_gold:
        import_gold(root, Path(args.import_gold))
        return

    kw = {"device": args.device} if args.device is not None else {}
    train_txt, val_txt = root / "train.txt", root / "val.txt"
    write_list(train_txt, labelled_images(root, "train"))
    write_list(val_txt, labelled_images(root, "val"))

    if args.eval:
        imported = read_imported(root)
        gold = [str((root / "images" / "test" / f"{k.split('__', 1)[1]}.jpg").resolve())
                for k in sorted(imported) if k.startswith("test__")]
        if not gold:
            raise SystemExit("No hay gold de test corregido. Importa antes con --import-gold.")
        expected = len((root / "gold_test.txt").read_text().split())
        if len(gold) < expected:
            print(f"AVISO: solo {len(gold)} de {expected} imágenes gold de test están corregidas.")
        gold_txt = root / "gold_test_corrected.txt"
        write_list(gold_txt, gold)
        gold_yaml = root / "gold.yaml"
        write_yaml(gold_yaml, train_txt, gold_txt)
        clear_label_caches(root)

        from ultralytics import YOLO  # import tardío
        m = YOLO(args.eval).val(data=str(gold_yaml), split="val", imgsz=args.imgsz, **kw)
        print(f"\nGold test ({len(gold)} imágenes)")
        print(f"Cajas    mAP50: {m.box.map50:.3f}  mAP50-95: {m.box.map:.3f}")
        print(f"Máscaras mAP50: {m.seg.map50:.3f}  mAP50-95: {m.seg.map:.3f}")
        return

    n_train = len(train_txt.read_text().split())
    n_val = len(val_txt.read_text().split())
    print(f"Train: {n_train} imágenes etiquetadas | Val: {n_val}")
    if n_train == 0 or n_val == 0:
        raise SystemExit("No hay etiquetas. Ejecuta antes 02_autolabel.py.")

    data_yaml = root / "data.yaml"
    write_yaml(data_yaml, train_txt, val_txt)

    from ultralytics import YOLO  # import tardío
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=20,
        flipud=0.5,      # las hojas no tienen "arriba" fijo en estas fotos
        name=args.name,
        **kw,
    )
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    print(f"\nModelo: {best}")
    print(f"Evalúa con: python 03_train_eval.py --root {args.root} --eval \"{best}\"")


if __name__ == "__main__":
    main()
