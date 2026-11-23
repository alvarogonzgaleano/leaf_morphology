"""
00_pipeline_mini.py   (va dentro de Segmentation/)
-------------------
Ejecuta el pipeline completo sobre una versión REDUCIDA de Pl@ntNet-300k.

Estructura esperada:
    proyecto/
      Segmentation/   01_select_leaf.py, 02_autolabel.py, 03_train_eval.py, 00_pipeline_mini.py
      Descriptors/    descriptores_hojas.py

Pasos:
    1) Segmentation/01_select_leaf.py   -> selecciona fotos de hojas (1 por especie y split)
    2) Segmentation/02_autolabel.py     -> etiqueta solo unas pocas (Grounding DINO + SAM 2)
    3) Segmentation/03_train_eval.py    -> entrena YOLO-seg pocas épocas
    4) Descriptors/descriptores_hojas.py -> descriptores de cada hoja de test

Dónde queda cada cosa:
    Segmentation/dataset_mini/              dataset, etiquetas y previews
    Segmentation/dataset_mini/ultimo_best.txt   ruta del modelo entrenado
    Descriptors/resultados_mini/            CSV, contornos e imágenes de debug

Uso (desde cualquier carpeta):
    python Segmentation/00_pipeline_mini.py --metadata plantnet300K_metadata.csv --images ruta/a/images
    ... --desde 3        retomar desde el paso 3
    ... --hasta 3        solo segmentación (sin descriptores)
    ... --dry-run        mostrar los comandos sin ejecutar nada
"""
import argparse
import subprocess
import sys
from pathlib import Path

SEG = Path(__file__).resolve().parent              # .../Segmentation
DESC = SEG.parent / "Descriptors"                  # .../Descriptors
SCRIPT_DESC = DESC / "descriptores_hojas.py"


def ejecutar(paso, titulo, script, argumentos, cwd, a):
    print(f"\n{'=' * 70}\nPASO {paso}: {titulo}\n{'=' * 70}")
    cmd = [sys.executable, str(script)] + [str(x) for x in argumentos]
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    if a.dry_run:
        return []
    # salida en directo y guardada (para leer la ruta del modelo en el paso 3)
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    salida = []
    for linea in proc.stdout:
        print(linea, end="", flush=True)
        salida.append(linea)
    if proc.wait() != 0:
        sys.exit(f"\nEl paso {paso} ha fallado (código {proc.returncode}). "
                 f"Corrige el error y retoma con:  --desde {paso}")
    return salida


def buscar_best(nombre, out):
    """Primero la ruta que imprimió 03_train_eval.py; si no, el best.pt más reciente
    (Ultralytics añade un número si la carpeta ya existe: leaf_mini2, leaf_mini3...)."""
    guardado = out / "ultimo_best.txt"
    if guardado.exists() and Path(guardado.read_text().strip()).exists():
        return Path(guardado.read_text().strip())
    candidatos = list((SEG / "runs" / "segment").glob(f"{nombre}*/weights/best.pt"))
    return max(candidatos, key=lambda p: p.stat().st_mtime) if candidatos else None


def comprobar_scripts(a):
    necesarios = []
    if a.desde <= 1 <= a.hasta: necesarios.append(SEG / "01_select_leaf.py")
    if a.desde <= 2 <= a.hasta: necesarios.append(SEG / "02_autolabel.py")
    if a.desde <= 3 <= a.hasta: necesarios.append(SEG / "03_train_eval.py")
    if a.desde <= 4 <= a.hasta: necesarios.append(SCRIPT_DESC)
    faltan = [p for p in necesarios if not p.exists()]
    if faltan:
        sys.exit("No encuentro estos scripts:\n  " + "\n  ".join(map(str, faltan)) +
                 "\nComprueba que Segmentation/ y Descriptors/ están al mismo nivel.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="plantnet300K_metadata.csv")
    ap.add_argument("--images", required=True, help="carpeta 'images' de Pl@ntNet-300k")
    ap.add_argument("--out", default="dataset_mini", help="carpeta del dataset (dentro de Segmentation)")
    ap.add_argument("--n-train", type=int, default=40, help="imágenes de train a etiquetar")
    ap.add_argument("--n-valtest", type=int, default=10, help="imágenes de val y de test")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--model", default="yolo11n-seg.pt", help="n = el más pequeño y rápido")
    ap.add_argument("--device", default=None, help="0 para GPU, cpu para CPU")
    ap.add_argument("--refine", action="store_true", help="pasa --refine a 02 (bordes más finos)")
    ap.add_argument("--desde", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--hasta", type=int, default=4, choices=[1, 2, 3, 4])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    comprobar_scripts(a)

    # rutas absolutas: los pasos se ejecutan en carpetas distintas
    cwd_usuario = Path.cwd()
    metadata = (cwd_usuario / a.metadata).resolve()
    images = (cwd_usuario / a.images).resolve()
    out = Path(a.out) if Path(a.out).is_absolute() else SEG / a.out
    salida_desc = DESC / "resultados_mini"
    dev = ["--device", a.device] if a.device else []
    nombre_run = "leaf_mini"
    hacer = lambda n: a.desde <= n <= a.hasta

    if hacer(1):
        if (out / "manifest.csv").exists():
            print(f"\n'{out}' ya existe: me salto el paso 1 (bórralo si quieres otra selección).")
        else:
            ejecutar(1, "Seleccionar imágenes de hojas", SEG / "01_select_leaf.py", [
                "--metadata", metadata, "--images", images, "--out", out,
                "--max-train", 1, "--max-val", 1, "--max-test", 1, "--gold-per-split", 10], SEG, a)

    if hacer(2):
        refine = ["--refine"] if a.refine else []
        ejecutar(2, f"Autoetiquetar {a.n_train} imágenes de train", SEG / "02_autolabel.py", [
            "--root", out, "--splits", "train", "--limit", a.n_train, *refine, *dev], SEG, a)
        ejecutar(2, f"Autoetiquetar {a.n_valtest} imágenes de val y de test", SEG / "02_autolabel.py", [
            "--root", out, "--splits", "val", "test", "--limit", a.n_valtest,
            "--preview", 10, *refine, *dev], SEG, a)

    if hacer(3):
        salida = ejecutar(3, f"Entrenar YOLO-seg ({a.epochs} épocas)", SEG / "03_train_eval.py", [
            "--root", out, "--model", a.model, "--epochs", a.epochs,
            "--batch", 8, "--workers", 2, "--name", nombre_run, *dev], SEG, a)
        for linea in salida:
            if linea.startswith("Modelo:"):
                (out / "ultimo_best.txt").write_text(linea.split(":", 1)[1].strip())

    if hacer(4):
        best = buscar_best(nombre_run, out)
        if best is None and not a.dry_run:
            sys.exit("No encuentro el best.pt del paso 3. ¿Terminó el entrenamiento?")
        ejecutar(4, "Extraer descriptores de las hojas de test", SCRIPT_DESC, [
            "--modelo", best or SEG / "runs/segment/leaf_mini/weights/best.pt",
            "--imagenes", out / "images" / "test", "--salida", salida_desc,
            "--conf", 0.3, "--debug"], DESC, a)

    if not a.dry_run:
        print("\nTodo listo. Revisa:")
        if a.hasta >= 2: print(f"  {out / 'preview'}   (etiquetas automáticas)")
        if a.hasta >= 4:
            print(f"  {salida_desc / 'debug'}   (contornos y lóbulos detectados)")
            print(f"  {salida_desc / 'descriptores.csv'}")


if __name__ == "__main__":
    main()