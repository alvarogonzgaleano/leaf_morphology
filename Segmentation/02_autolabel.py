"""
02_autolabel.py
Genera etiquetas de segmentación automáticas ("silver"):
  1) Un detector de vocabulario abierto busca cajas de "leaf". Dos opciones (--backend):
       gdino      Grounding DINO (transformers). Recomendado: detecta mucho mejor.
       yoloworld  YOLO-World (ultralytics). Más ligero, pero detecta poco en estas fotos.
  2) Se descartan cajas dudosas (se prioriza precisión: mejor pocas hojas pero bien):
       - las que el detector asocia a "tree", "shrub", etc. en vez de a "leaf" (--neg)
       - las que ocupan casi toda la imagen (--max-box-area)
       - las que engloban varias hojas (p. ej. "la planta entera")
  3) SAM 2 convierte cada caja en una máscara. Con --refine, SAM trabaja sobre un
     recorte de cada hoja, lo que conserva mucho mejor el borde (dientes, lóbulos).
  4) Se eliminan máscaras duplicadas o diminutas y se extrae el contorno principal.
  5) Se guardan polígonos normalizados en formato YOLO-seg (clase 0).

Además copia las etiquetas de las imágenes 'gold' a review_labels/ con el mismo
nombre que review_images/, para importarlas como pre-anotación en CVAT/Roboflow
y corregir en vez de dibujar desde cero.

Uso:
    python 02_autolabel.py --root dataset --preview 30 --device 0

Los nombres de pesos y parámetros pueden cambiar entre versiones de Ultralytics.
La primera ejecución de YOLO-World instala CLIP automáticamente (necesita internet).
"""
from __future__ import annotations

import argparse
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------- utilidades puras
def drop_container_boxes(boxes: np.ndarray, inside_thr=0.9, min_children=2) -> list[int]:
    """Índices de cajas a conservar: se quitan las que contienen >= min_children cajas
    más pequeñas (suelen ser ramas o plantas enteras etiquetadas como 'leaf')."""
    n = len(boxes)
    if n < min_children + 1:
        return list(range(n))
    b = boxes.astype(np.float32)
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep = []
    for i in range(n):
        children = 0
        for j in range(n):
            if i == j or areas[j] <= 0 or areas[j] >= areas[i]:
                continue
            iw = min(b[i, 2], b[j, 2]) - max(b[i, 0], b[j, 0])
            ih = min(b[i, 3], b[j, 3]) - max(b[i, 1], b[j, 1])
            if iw > 0 and ih > 0 and (iw * ih) / areas[j] >= inside_thr:
                children += 1
        if children < min_children:
            keep.append(i)
    return keep


def masks_to_polygons(masks: np.ndarray, scores: np.ndarray, min_area=0.01,
                      iou_thr=0.8, eps_px=1.5) -> list[np.ndarray]:
    """masks: (N, H, W) binarias. Devuelve polígonos normalizados (k, 2) en [0, 1].
    - descarta máscaras con área < min_area (fracción de la imagen)
    - descarta duplicados (IoU > iou_thr con una máscara de mayor score)
    - usa solo el contorno externo más grande de cada máscara"""
    if len(masks) == 0:
        return []
    _, H, W = masks.shape
    kept, polys = [], []
    for i in np.argsort(-scores):
        m = masks[i] > 0
        if m.sum() < min_area * H * W:
            continue
        if any((m & k).sum() / max((m | k).sum(), 1) > iou_thr for k in kept):
            continue
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        c = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2)
        if len(c) < 3:
            continue
        kept.append(m)
        poly = c.astype(np.float32) / np.array([W, H], dtype=np.float32)
        polys.append(np.clip(poly, 0.0, 1.0))
    return polys


def polygons_to_lines(polys: list[np.ndarray]) -> list[str]:
    return ["0 " + " ".join(f"{v:.5f}" for v in p.reshape(-1)) for p in polys]


def label_matches(label: str, prompt: str) -> bool:
    """La caja es válida si el detector la asoció SOLO a palabras del prompt
    ("leaf" sí; "tree", "leaf tree" o etiqueta vacía no: ante la duda, se descarta)."""
    tokens = set(label.lower().replace(".", " ").split())
    return bool(tokens) and tokens <= set(prompt.lower().replace(".", " ").split())


def crop_box(box, W: int, H: int, pad: float):
    """Recorte alrededor de la caja con un margen relativo; devuelve (x1, y1, x2, y2) enteros."""
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return (int(max(0, x1 - px)), int(max(0, y1 - py)),
            int(min(W, np.ceil(x2 + px))), int(min(H, np.ceil(y2 + py))))


def fit_mask(m: np.ndarray, h: int, w: int) -> np.ndarray:
    m = m > 0
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    return m


def draw_preview(img_path: Path, label_path: Path, out_path: Path):
    img = cv2.imread(str(img_path))
    if img is None or not label_path.exists():
        return
    h, w = img.shape[:2]
    overlay = img.copy()
    for line in label_path.read_text().splitlines():
        vals = list(map(float, line.split()[1:]))
        pts = (np.array(vals).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.polylines(img, [pts], True, (0, 0, 255), 2)
    cv2.imwrite(str(out_path), cv2.addWeighted(overlay, 0.35, img, 0.65, 0))


# ---------------------------------------------------------------- detectores
def torch_device(device):
    """Convierte '0' en 'cuda:0'; None en cuda si hay GPU y si no cpu."""
    import torch
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return f"cuda:{device}" if str(device).isdigit() else str(device)


def build_detector(args, conf: float):
    """Devuelve detect(ruta) -> (boxes (N,4) xyxy en píxeles, scores (N,), etiquetas), por score."""
    empty = (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), [])

    if args.backend == "gdino":
        try:
            import torch
            from PIL import Image
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as e:
            raise SystemExit(f"Falta una librería: {e}\nInstala con:  pip install transformers pillow")
        dev = torch_device(args.device)
        print(f"Cargando {args.gdino} en {dev} (la primera vez descarga el modelo)...")
        processor = AutoProcessor.from_pretrained(args.gdino)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino).to(dev).eval()
        # Grounding DINO: clases en minúsculas separadas por punto. Las negativas sirven
        # para que una copa de árbol se etiquete "tree" y no "leaf", y poder descartarla.
        classes = [args.prompt.strip().lower().rstrip(".")]
        classes += [c.strip().lower() for c in args.neg.split(".") if c.strip()]
        text = ". ".join(classes) + "."
        print(f"Texto para el detector: '{text}'")

        def detect(path):
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, text=text, return_tensors="pt").to(dev)
            with torch.no_grad():
                outputs = model(**inputs)
            kw = dict(text_threshold=args.text_threshold, target_sizes=[image.size[::-1]])
            try:   # transformers recientes
                res = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=conf, **kw)[0]
            except TypeError:   # versiones antiguas
                res = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=conf, **kw)[0]
            boxes = res["boxes"].cpu().numpy().astype(np.float32)
            scores = res["scores"].cpu().numpy().astype(np.float32)
            labels = list(res["text_labels"] if "text_labels" in res else res["labels"])
            if len(boxes) == 0:
                return empty
            order = np.argsort(-scores)[:args.max_det]
            return boxes[order], scores[order], [str(labels[i]) for i in order]

        return detect

    # --- yoloworld
    from ultralytics import YOLOWorld
    det = YOLOWorld(args.det)
    try:
        det.set_classes([args.prompt])  # descarga CLIP (~340 MB) la primera vez
    except (OSError, RuntimeError) as e:
        raise SystemExit(
            f"No se pudo cargar el modelo de texto CLIP: {e}\n"
            "Suele ser un problema de red. Vuelve a ejecutar el comando."
        )
    kw = {"verbose": False}
    if args.device is not None:
        kw["device"] = args.device

    def detect(path):
        r = det.predict(path, conf=conf, max_det=args.max_det, **kw)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return empty
        n = len(r.boxes)
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), [args.prompt.lower()] * n

    return detect


# ---------------------------------------------------------------- pipeline
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--backend", choices=["gdino", "yoloworld"], default="gdino",
                    help="detector de texto (por defecto: Grounding DINO)")
    ap.add_argument("--gdino", default="IDEA-Research/grounding-dino-tiny",
                    help="modelo de Hugging Face para --backend gdino")
    ap.add_argument("--det", default="yolov8s-worldv2.pt", help="pesos para --backend yoloworld")
    ap.add_argument("--sam", default="sam2.1_b.pt")
    ap.add_argument("--prompt", default="leaf")
    ap.add_argument("--conf", type=float, default=None,
                    help="umbral de confianza (por defecto 0.30 con gdino, 0.25 con yoloworld)")
    ap.add_argument("--text-threshold", type=float, default=0.25,
                    help="solo gdino: umbral de coincidencia con el texto")
    ap.add_argument("--neg", default="tree. shrub. bush. branch. flower. fruit.",
                    help="solo gdino: clases 'trampa'; las cajas etiquetadas así se descartan "
                         "('' para desactivar)")
    ap.add_argument("--max-box-area", type=float, default=0.9,
                    help="descartar cajas que ocupen más de esta fracción de la imagen")
    ap.add_argument("--refine", action="store_true",
                    help="SAM sobre un recorte de cada hoja: bordes más finos, más lento")
    ap.add_argument("--refine-pad", type=float, default=0.15,
                    help="margen del recorte en --refine, relativo al tamaño de la caja")
    ap.add_argument("--max-det", type=int, default=15)
    ap.add_argument("--min-area", type=float, default=0.01,
                    help="área mínima de la máscara como fracción de la imagen")
    ap.add_argument("--iou-dup", type=float, default=0.8, help="IoU para considerar duplicado")
    ap.add_argument("--eps-px", type=float, default=1.0, help="simplificación del contorno (px)")
    ap.add_argument("--device", default=None, help="p. ej. 0 o cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--preview", type=int, default=0,
                    help="guardar N imágenes con la máscara dibujada para revisar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="procesar solo las N primeras imágenes de cada split (para probar)")
    ap.add_argument("--retry-empty", action="store_true",
                    help="reintentar las imágenes listadas en no_detections.txt "
                         "(úsalo tras cambiar --conf o --prompt)")
    ap.add_argument("--debug", action="store_true",
                    help="imprimir por imagen cuántas cajas y máscaras salen en cada fase")
    args = ap.parse_args()

    root = Path(args.root)
    if not (root / "images").is_dir():
        raise SystemExit(
            f"No existe '{root / 'images'}'. Ejecuta primero 01_select_leaf.py y comprueba "
            "que --root es la misma carpeta que usaste como --out (por defecto, 'dataset')."
        )

    from ultralytics import SAM  # import tardío: --help no carga torch

    kw = {"verbose": False}
    if args.device is not None:
        kw["device"] = args.device

    conf = args.conf if args.conf is not None else (0.30 if args.backend == "gdino" else 0.25)
    detect = build_detector(args, conf)
    sam = SAM(args.sam)

    nodet_file = root / "no_detections.txt"
    errors_file = root / "autolabel_errors.txt"
    no_det = set(nodet_file.read_text().split()) if nodet_file.exists() else set()
    n_done, n_skipped_empty, n_err, consecutive_err = 0, 0, 0, 0
    errors: list[str] = []
    done_now: list[tuple[str, str]] = []   # (split, stem) etiquetadas en esta ejecución

    def dbg(img: Path, msg: str):
        if args.debug:
            print(f"  [debug] {img.stem[:8]}: {msg}", flush=True)

    def sam_masks(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Máscaras (N, H, W) booleanas a resolución de la imagen."""
        H, W = img.shape[:2]
        if not args.refine:
            s = sam(img, bboxes=boxes.tolist(), **kw)[0]
            if s.masks is None:
                return np.zeros((0, H, W), bool)
            return np.stack([fit_mask(m, H, W) for m in s.masks.data.cpu().numpy()])
        out = []
        for b in boxes:   # un recorte por hoja: SAM ve la hoja más grande -> borde más fino
            cx1, cy1, cx2, cy2 = crop_box(b, W, H, args.refine_pad)
            crop = np.ascontiguousarray(img[cy1:cy2, cx1:cx2])
            local = [[b[0] - cx1, b[1] - cy1, b[2] - cx1, b[3] - cy1]]
            s = sam(crop, bboxes=local, **kw)[0]
            full = np.zeros((H, W), bool)
            if s.masks is not None and len(s.masks.data):
                full[cy1:cy2, cx1:cx2] = fit_mask(s.masks.data.cpu().numpy()[0], cy2 - cy1, cx2 - cx1)
            out.append(full)
        return np.stack(out) if out else np.zeros((0, H, W), bool)

    def process(p: Path) -> list[str]:
        """Devuelve las líneas YOLO-seg de la imagen (lista vacía si no hay hojas válidas)."""
        img = cv2.imread(str(p))
        if img is None:
            raise ValueError("no se pudo leer la imagen")
        H, W = img.shape[:2]
        boxes, confs, labels = detect(str(p))
        if len(boxes) == 0:
            dbg(p, f"detector: 0 cajas con conf >= {conf}")
            return []
        dbg(p, f"detector: {len(boxes)} cajas, conf máx = {confs.max():.2f}")

        # 1) solo cajas que el detector asocia a la hoja (no a árbol, arbusto...)
        ok = [i for i, l in enumerate(labels) if label_matches(l, args.prompt)]
        dropped = Counter(labels[i] or "(sin etiqueta)" for i in range(len(labels)) if i not in ok)
        boxes, confs = boxes[ok], confs[ok]
        if dropped:
            dbg(p, f"descartadas por etiqueta: {dict(dropped)}")
        # 2) cajas que ocupan casi toda la imagen
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / float(W * H)
        ok = areas <= args.max_box_area
        if (~ok).any():
            dbg(p, f"descartadas por ocupar > {args.max_box_area:.0%} de la imagen: {int((~ok).sum())}")
        boxes, confs = boxes[ok], confs[ok]
        # 3) cajas que engloban varias hojas
        keep = drop_container_boxes(boxes)
        if len(keep) < len(boxes):
            dbg(p, f"descartadas por englobar varias hojas: {len(boxes) - len(keep)}")
        boxes, confs = boxes[keep], confs[keep]
        if len(boxes) == 0:
            dbg(p, "no queda ninguna caja tras los filtros")
            return []

        masks = sam_masks(img, boxes)
        if len(masks) == 0:
            dbg(p, "SAM: sin máscaras")
            return []
        scores = confs[:len(masks)] if len(confs) >= len(masks) else np.ones(len(masks))
        polys = masks_to_polygons(masks, scores, args.min_area, args.iou_dup, args.eps_px)
        dbg(p, f"SAM: {len(masks)} máscaras -> {len(polys)} polígonos válidos")
        return polygons_to_lines(polys)

    try:
        for split in args.splits:
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            imgs = sorted((root / "images" / split).glob("*.jpg"))
            if args.limit:
                imgs = imgs[:args.limit]
            print(f"[{split}] {len(imgs)} imágenes")
            t0, n_proc, last_print = time.time(), 0, 0
            for i, p in enumerate(imgs, 1):
                if n_proc and n_proc % 25 == 0 and n_proc != last_print:
                    last_print = n_proc
                    el = time.time() - t0
                    print(f"[{split}] {i}/{len(imgs)} | {n_proc} procesadas en {el / 60:.1f} min "
                          f"({el / n_proc:.1f} s/imagen)", flush=True)
                lbl = root / "labels" / split / f"{p.stem}.txt"
                if not args.overwrite:
                    if lbl.exists():
                        continue
                    if str(p) in no_det and not args.retry_empty:
                        n_skipped_empty += 1
                        continue
                n_proc += 1

                try:
                    lines = process(p)
                except Exception as e:   # una imagen problemática no debe parar todo
                    n_err += 1
                    consecutive_err += 1
                    errors.append(f"{p}\t{type(e).__name__}: {e}")
                    if n_err <= 3:
                        print(f"  ERROR en {p.name}: {type(e).__name__}: {e}", flush=True)
                    if consecutive_err >= 10:
                        raise SystemExit(
                            f"10 errores seguidos; paro para no perder tiempo. "
                            f"El primero fue:\n  {errors[0]}\nDetalles en {errors_file}"
                        )
                    continue
                consecutive_err = 0

                if lines:
                    lbl.write_text("\n".join(lines) + "\n")
                    no_det.discard(str(p))
                    done_now.append((split, p.stem))
                    n_done += 1
                else:
                    no_det.add(str(p))
                    lbl.unlink(missing_ok=True)   # con --overwrite: quitar la etiqueta antigua
    finally:
        # se guarda también si paras con Ctrl+C o si hay un error grave
        nodet_file.write_text("\n".join(sorted(no_det)))
        if errors:
            errors_file.write_text("\n".join(errors) + "\n")

    print(f"\nEtiquetadas en esta ejecución: {n_done} | sin detección válida (acumulado): "
          f"{len(no_det)} (ver {nodet_file})")
    if n_err:
        print(f"AVISO: {n_err} imágenes dieron error y se saltaron (ver {errors_file}).")
    if n_skipped_empty:
        print(f"AVISO: {n_skipped_empty} imágenes se omitieron por estar ya en no_detections.txt. "
              "Si cambiaste --conf o --prompt, añade --retry-empty para reintentarlas.")

    # Pre-anotaciones para las imágenes gold, con el mismo nombre que review_images/
    review_imgs = root / "review_images"
    review_lbls = root / "review_labels"
    if review_imgs.exists():
        review_lbls.mkdir(exist_ok=True)
        n_rev = 0
        for img in review_imgs.glob("*.jpg"):
            split, h = img.stem.split("__", 1)
            src = root / "labels" / split / f"{h}.txt"
            dst = review_lbls / f"{img.stem}.txt"
            if src.exists():
                shutil.copy2(src, dst)
                n_rev += 1
            else:
                dst.unlink(missing_ok=True)   # no dejar pre-anotaciones obsoletas
        print(f"Pre-anotaciones gold: {n_rev} en {review_lbls}")

    if args.preview:
        prev_dir = root / "preview"
        prev_dir.mkdir(exist_ok=True)
        for old in prev_dir.glob("*.jpg"):   # no mezclar con previews de ejecuciones anteriores
            old.unlink()
        # Prioriza lo etiquetado en ESTA ejecución; si no hay nada nuevo, usa lo existente.
        pool = done_now or [(f.parent.name, f.stem) for f in sorted((root / "labels").glob("*/*.txt"))]
        random.seed(args.seed)
        chosen = random.sample(pool, min(args.preview, len(pool)))
        for split, stem in chosen:
            draw_preview(root / "images" / split / f"{stem}.jpg",
                         root / "labels" / split / f"{stem}.txt",
                         prev_dir / f"{split}__{stem}.jpg")
        print(f"Previews: {len(chosen)} en {prev_dir}")


if __name__ == "__main__":
    main()