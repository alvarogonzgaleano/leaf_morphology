"""
descriptores_hojas.py
---------------------
Extrae descriptores morfológicos de cada hoja segmentada por un modelo YOLO-seg
(Ultralytics). Cada máscara = una hoja = una fila del CSV.

Uso:
    python descriptores_hojas.py --modelo runs/segment/train/weights/best.pt \
                                 --imagenes ruta/a/imagenes \
                                 --salida resultados --debug

Salida:
    resultados/descriptores.csv   -> una fila por hoja con todos los descriptores
    resultados/contornos.npz      -> contorno normalizado de cada hoja (256 puntos)
    resultados/debug/*.jpg        -> (opcional) imágenes con contorno, ejes y senos dibujados
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

N_PUNTOS_CONTORNO = 256   # puntos del contorno remuestreado (para Fourier y para guardar)
N_ARMONICOS = 10          # coeficientes de Fourier que se guardan como columnas
ARMONICO_CORTE_ALTO = 20  # a partir de este armónico se considera "detalle fino" del borde (dientes)


# ---------------------------------------------------------------------------
# 1. Limpieza de la máscara: nos quedamos con UNA hoja por máscara
# ---------------------------------------------------------------------------
def limpiar_mascara(mascara):
    """Devuelve (máscara limpia, contorno) quedándose solo con la mayor región
    conectada y rellenando agujeros. Si no hay región, devuelve (None, None)."""
    m = (mascara > 0).astype(np.uint8)
    contornos, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return None, None
    c = max(contornos, key=cv2.contourArea)
    limpia = np.zeros_like(m)
    cv2.drawContours(limpia, [c], -1, 1, thickness=cv2.FILLED)  # rellena agujeros
    return limpia, c


# ---------------------------------------------------------------------------
# 2. Funciones auxiliares
# ---------------------------------------------------------------------------
def ejes_principales(mascara, contorno):
    """Longitud y anchura usando PCA: la longitud es la extensión de la hoja a lo
    largo de su eje principal; la anchura, la extensión perpendicular a él.
    No depende de cómo esté girada la hoja en la foto."""
    ys, xs = np.nonzero(mascara)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    centro = pts.mean(axis=0)
    cov = np.cov((pts - centro).T)
    valores, vectores = np.linalg.eigh(cov)
    eje_mayor = vectores[:, np.argmax(valores)]
    eje_menor = np.array([-eje_mayor[1], eje_mayor[0]])

    c = contorno.reshape(-1, 2).astype(np.float64) - centro
    proy_mayor = c @ eje_mayor
    proy_menor = c @ eje_menor
    longitud = proy_mayor.max() - proy_mayor.min()
    anchura = proy_menor.max() - proy_menor.min()
    return longitud, anchura, centro, eje_mayor, eje_menor


def suavizar_contorno(contorno, sigma=2.0):
    """Suaviza el contorno (filtro gaussiano circular) para eliminar el efecto
    'escalera' de los píxeles, que inflaría el perímetro en torno a un 5-10 %."""
    c = contorno.reshape(-1, 2).astype(np.float64)
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    pad = np.vstack([c[-r:], c, c[:r]])
    x = np.convolve(pad[:, 0], k, mode="valid")
    y = np.convolve(pad[:, 1], k, mode="valid")
    return np.column_stack([x, y]).astype(np.float32).reshape(-1, 1, 2)


def remuestrear_contorno(contorno, n=N_PUNTOS_CONTORNO):
    """Remuestrea el contorno a n puntos equiespaciados a lo largo del borde."""
    c = contorno.reshape(-1, 2).astype(np.float64)
    c = np.vstack([c, c[:1]])  # cerrar
    seg = np.sqrt((np.diff(c, axis=0) ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=False)
    x = np.interp(t, s, c[:, 0])
    y = np.interp(t, s, c[:, 1])
    return np.column_stack([x, y])


def contar_lobulos(contorno, longitud, umbral_relativo):
    """Cuenta lóbulos a partir de los 'senos' (entrantes) del borde.
    Un seno es un defecto de convexidad cuya profundidad supera
    umbral_relativo * longitud. Con k senos profundos el borde queda dividido
    en k lóbulos (k=0 -> hoja entera = 1 lóbulo)."""
    casco_idx = cv2.convexHull(contorno, returnPoints=False)
    if casco_idx is None or len(casco_idx) < 4:
        return 1, []
    try:
        defectos = cv2.convexityDefects(contorno, casco_idx)
    except cv2.error:
        return 1, []
    if defectos is None:
        return 1, []
    profundidad_min = umbral_relativo * longitud
    defectos = defectos.reshape(-1, 4)  # la forma varía entre versiones de OpenCV
    senos = [tuple(contorno[f][0]) for s, e, f, d in defectos
             if d / 256.0 > profundidad_min]
    return max(1, len(senos)), senos


def dimension_fractal(contorno, forma):
    """Dimensión fractal del borde por box-counting.
    ~1.0 = borde liso; valores más altos = borde más dentado/irregular."""
    lienzo = np.zeros(forma, dtype=np.uint8)
    cv2.drawContours(lienzo, [contorno], -1, 1, thickness=1)
    x, y, w, h = cv2.boundingRect(contorno)
    lienzo = lienzo[y:y + h, x:x + w]
    lado_max = max(2, min(w, h) // 4)
    tamanos = [2 ** k for k in range(1, int(np.log2(lado_max)) + 1)]
    if len(tamanos) < 3:
        return np.nan
    cuentas = []
    for t in tamanos:
        hh, ww = int(np.ceil(h / t)) * t, int(np.ceil(w / t)) * t
        pad = np.zeros((hh, ww), dtype=np.uint8)
        pad[:h, :w] = lienzo
        cajas = pad.reshape(hh // t, t, ww // t, t).max(axis=(1, 3))
        cuentas.append(cajas.sum())
    pendiente = np.polyfit(np.log(1 / np.array(tamanos)), np.log(cuentas), 1)[0]
    return pendiente


def descriptores_fourier(contorno_remuestreado, centroide):
    """Fourier sobre la 'firma de distancia al centroide':
    se recorre el borde y se mide la distancia al centro en cada punto.
    Esa señal se descompone en frecuencias (armónicos):
      - armónico 2: forma alargada (la distancia sube y baja 2 veces por vuelta)
      - armónico k: k "picos" en el borde (p. ej., k lóbulos)
      - armónicos altos: dientes y detalle fino del margen
    Se normaliza por el armónico 0 (distancia media) -> invariante a escala,
    y se usa la magnitud -> invariante a rotación y al punto de inicio."""
    r = np.linalg.norm(contorno_remuestreado - centroide, axis=1)
    F = np.abs(np.fft.rfft(r))
    F_norm = F / F[0]
    armonicos = F_norm[1:]                     # k = 1 .. N/2
    k = np.arange(1, len(armonicos) + 1)
    energia = armonicos ** 2
    energia_total = energia.sum()

    return {
        **{f"fourier_h{i}": armonicos[i - 1] for i in range(1, N_ARMONICOS + 1)},
        # "frecuencia media": centro de masas del espectro
        "fourier_frecuencia_media": (k * energia).sum() / max(energia_total, 1e-12),
        # armónico con más energía (ignorando el 1, que solo refleja asimetría)
        "fourier_armonico_dominante": int(k[1:][np.argmax(armonicos[1:])]),
        # energía del detalle fino del borde (dientes), relativa al tamaño medio
        "fourier_energia_alta": np.sqrt(energia[k > ARMONICO_CORTE_ALTO].sum()),
    }


# ---------------------------------------------------------------------------
# 3. Descriptores de una hoja
# ---------------------------------------------------------------------------
def descriptores_hoja(mascara, umbral_lobulo=0.08):
    """Recibe una máscara binaria (una hoja) y devuelve (dict, contorno_256, extra)
    o (None, None, None) si la máscara no es válida."""
    m, contorno = limpiar_mascara(mascara)
    if m is None or len(contorno) < 10:
        return None, None, None

    area = float(m.sum())                                   # (1)
    suave = suavizar_contorno(contorno)
    perimetro = cv2.arcLength(suave, closed=True)           # (2)
    longitud, anchura, centro, e1, e2 = ejes_principales(m, contorno)  # (3)(4)
    relacion_lw = longitud / anchura if anchura > 0 else np.nan        # (5)
    circularidad = 4 * np.pi * area / perimetro ** 2        # (6)

    # (7) Contorno: se guarda el contorno normalizado (en contornos.npz)
    # y se resume con dos medidas de forma respecto a su envolvente convexa
    casco = cv2.convexHull(contorno)
    area_casco = cv2.contourArea(casco)
    perimetro_casco = cv2.arcLength(casco, closed=True)
    solidez = area / area_casco if area_casco > 0 else np.nan
    c256 = remuestrear_contorno(suave)
    c_norm = (c256 - centro) / np.sqrt(area)   # centrado y sin escala

    lobulos, senos = contar_lobulos(contorno, longitud, umbral_lobulo)  # (8)

    # (9) Complejidad del borde
    rugosidad = perimetro / perimetro_casco if perimetro_casco > 0 else np.nan
    fractal = dimension_fractal(contorno, m.shape)

    d = {
        "area_px": area,
        "perimetro_px": perimetro,
        "longitud_px": longitud,
        "anchura_px": anchura,
        "relacion_longitud_anchura": relacion_lw,
        "circularidad": circularidad,
        "solidez": solidez,
        "n_puntos_contorno": len(contorno),
        "lobulos": lobulos,
        "rugosidad_borde": rugosidad,
        "dimension_fractal_borde": fractal,
        **descriptores_fourier(c256, c256.mean(axis=0)),
    }
    extra = {"contorno": contorno, "casco": casco, "senos": senos,
             "centro": centro, "e1": e1, "longitud": longitud}
    return d, c_norm, extra


def dibujar_debug(img, extra, idx):
    cv2.drawContours(img, [extra["contorno"]], -1, (0, 255, 0), 2)
    cv2.drawContours(img, [extra["casco"]], -1, (255, 128, 0), 1)
    for p in extra["senos"]:
        cv2.circle(img, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
    c, e1, L = extra["centro"], extra["e1"], extra["longitud"]
    p1 = tuple((c - e1 * L / 2).astype(int)); p2 = tuple((c + e1 * L / 2).astype(int))
    cv2.line(img, p1, p2, (255, 0, 255), 1)
    cv2.putText(img, str(idx), tuple(c.astype(int)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)


def toca_borde(mascara, margen=2):
    return bool(mascara[:margen].any() or mascara[-margen:].any()
                or mascara[:, :margen].any() or mascara[:, -margen:].any())


# ---------------------------------------------------------------------------
# 4. Pipeline con YOLO
# ---------------------------------------------------------------------------
def procesar(args):
    from ultralytics import YOLO

    salida = Path(args.salida); salida.mkdir(parents=True, exist_ok=True)
    if args.debug:
        (salida / "debug").mkdir(exist_ok=True)

    modelo = YOLO(args.modelo)
    filas, contornos = [], {}
    descartes = {"pequena": 0, "borde": 0, "invalida": 0}

    for r in modelo.predict(source=args.imagenes, conf=args.conf,
                            retina_masks=True, stream=True, verbose=False):
        nombre = Path(r.path).stem
        if r.masks is None:
            continue
        H, W = r.orig_shape
        mascaras = r.masks.data.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        img_debug = r.orig_img.copy() if args.debug else None

        for i, (mk, conf) in enumerate(zip(mascaras, confs)):
            if mk.shape != (H, W):
                mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)
            mk = (mk > 0.5).astype(np.uint8)

            if mk.sum() < args.area_min:
                descartes["pequena"] += 1; continue
            borde = toca_borde(mk)
            if borde and not args.incluir_borde:
                descartes["borde"] += 1; continue

            d, c_norm, extra = descriptores_hoja(mk, args.umbral_lobulo)
            if d is None:
                descartes["invalida"] += 1; continue

            id_hoja = f"{nombre}_{i}"
            filas.append({"id_hoja": id_hoja, "imagen": nombre,
                          "confianza": float(conf), "toca_borde": borde, **d})
            contornos[id_hoja] = c_norm
            if args.debug:
                dibujar_debug(img_debug, extra, i)

        if args.debug and img_debug is not None:
            cv2.imwrite(str(salida / "debug" / f"{nombre}.jpg"), img_debug)

    if not filas:
        print("No se ha extraído ninguna hoja. Revisa conf / area_min.")
        return
    with open(salida / "descriptores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    np.savez_compressed(salida / "contornos.npz", **contornos)
    print(f"Hojas procesadas: {len(filas)} | descartadas: {descartes}")
    print(f"CSV: {salida / 'descriptores.csv'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Descriptores morfológicos de hojas")
    p.add_argument("--modelo", required=True, help="pesos YOLO-seg (best.pt)")
    p.add_argument("--imagenes", required=True, help="carpeta o imagen de entrada")
    p.add_argument("--salida", default="resultados_descriptores")
    p.add_argument("--conf", type=float, default=0.5, help="confianza mínima")
    p.add_argument("--area_min", type=int, default=2000, help="área mínima en px")
    p.add_argument("--umbral_lobulo", type=float, default=0.08,
                   help="profundidad mínima de un seno, relativa a la longitud")
    p.add_argument("--incluir_borde", action="store_true",
                   help="no descartar hojas cortadas por el borde de la imagen")
    p.add_argument("--debug", action="store_true", help="guardar imágenes de control")
    procesar(p.parse_args())