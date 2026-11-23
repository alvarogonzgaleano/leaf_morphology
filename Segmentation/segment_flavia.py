"""
Batch leaf segmentation for the Flavia dataset (v2).

Pipeline per image:
  1. Estimate the background colour from a frame around the image border.
  2. Compute the per-pixel colour distance to that background (CIELAB, Delta E).
  3. Otsu threshold on the distance map  -> raw mask.
  4. Light morphological cleaning (small opening/closing, to keep serration).
  5. Keep the largest connected component and fill its holes.

Outputs: binary masks, contact sheets with the contour overlaid (aspect ratio
preserved), and metrics.csv with quality flags computed on the RAW mask, i.e.
before the largest-component / hole-filling steps hide the problems.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

FRAME_PX = 30      # width of the border frame used to estimate the background
IGNORE_PX = 4      # outermost pixels ignored (scanner edge artefacts)


def parse_args():
    """
    Parse command-line arguments.
    Returns:
        - input_dir (Path): folder containing Flavia JPEG images
        - output_dir (Path): folder where masks, contact sheets and metrics are saved
        - open_px (int): radius of the morphological opening (noise removal)
        - close_px (int): radius of the morphological closing (small gaps)
        - sample (int): if set, process only this many images (uniform sampling)
    """
    p = argparse.ArgumentParser(description="Batch segmentation - Flavia dataset")
    p.add_argument("input_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--open-px", type=int, default=1, help="Opening radius (px)")
    p.add_argument("--close-px", type=int, default=2, help="Closing radius (px)")
    p.add_argument("--sample", type=int, default=None, help="Process only N images")
    return p.parse_args()


def background_distance(bgr):
    """
    Colour distance of every pixel to the page background.

    The background colour is the per-channel median of a frame around the image
    border (the leaf never reaches it in Flavia). Working in CIELAB and with the
    distance to the background, instead of a grey-level threshold, keeps pale or
    glossy parts of the leaf (specular highlights, yellowish zones) in the mask,
    since they are dark/saturated *relative to the page* in colour even when their
    grey level is close to white.

    Parameters:
        - bgr (np.ndarray): image in BGR format

    Returns:
        - dist (np.ndarray): float32 map, Delta E to the background colour
        - bg_lab (np.ndarray): estimated background colour (L, a, b)
        - bg_std (float): mean per-channel std of the border frame (background uniformity)
    """
    lab = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    frame = np.zeros((h, w), bool)
    frame[:FRAME_PX, :] = frame[-FRAME_PX:, :] = True
    frame[:, :FRAME_PX] = frame[:, -FRAME_PX:] = True
    bg = np.median(lab[frame], axis=0)
    bg_std = float(lab[frame].std(axis=0).mean())
    return np.sqrt(((lab - bg) ** 2).sum(-1)), bg, bg_std


def segment_leaf(bgr, open_px=1, close_px=2):
    """
    Segment a single leaf image.

    Parameters:
        - bgr (np.ndarray): image in BGR format as loaded by cv2.imread
        - open_px (int): radius of the morphological opening
        - close_px (int): radius of the morphological closing

    Returns:
        - mask (np.ndarray): final binary mask (uint8, 0/255), single component, no holes
        - info (dict): intermediate quantities used by compute_metrics:
            - raw (np.ndarray): mask after Otsu + morphology, before component selection
            - thr (float): Otsu threshold on the Delta E map
            - bg_std (float): std of the border frame used as background reference
    """
    dist, _, bg_std = background_distance(bgr)
    scale = max(float(np.percentile(dist, 99.5)), 1.0)
    d8 = np.clip(dist / scale * 255, 0, 255).astype(np.uint8)
    t, raw = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = t / 255 * scale

    raw[:IGNORE_PX, :] = raw[-IGNORE_PX:, :] = 0
    raw[:, :IGNORE_PX] = raw[:, -IGNORE_PX:] = 0

    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_px+1, 2*open_px+1))
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    if close_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_px+1, 2*close_px+1))
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    mask = np.zeros_like(raw)
    if n > 1:
        mask = np.where(labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]), 255, 0).astype(np.uint8)
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros_like(raw)
        cv2.drawContours(mask, cs, -1, 255, cv2.FILLED)   # fills holes, keeps the outer boundary

    return mask, {"raw": raw, "thr": thr, "bg_std": bg_std}


def compute_metrics(mask, info, name):
    """
    Quality metrics and warning flags. They are computed on the raw mask where it
    matters, because the final mask is a single component without holes by construction.

    Parameters:
        - mask (np.ndarray): final mask returned by segment_leaf
        - info (dict): second value returned by segment_leaf
        - name (str): image filename

    Returns:
        - dict with: name, area_frac, n_raw_components, second_ratio (2nd/1st component area
          in the raw mask), hole_frac (area added by hole filling / final area), solidity,
          frame_frac (fraction of the background-reference frame that is leaf), touches_border,
          flags (pipe-separated), exclude (bool: flags that invalidate the measurements)
    """
    h, w = mask.shape
    area = float((mask > 0).sum())
    raw = info["raw"]

    n, _, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True) or [0]
    second_ratio = areas[1] / areas[0] if len(areas) > 1 and areas[0] else 0.0
    hole_frac = max(0.0, (area - float((raw > 0).sum() - sum(areas[1:]))) / area) if area else 0.0

    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solidity = 0.0
    if cs:
        hull_area = cv2.contourArea(cv2.convexHull(max(cs, key=cv2.contourArea)))
        solidity = area / hull_area if hull_area > 0 else 0.0

    frame = np.zeros((h, w), bool)
    frame[:FRAME_PX, :] = frame[-FRAME_PX:, :] = True
    frame[:, :FRAME_PX] = frame[:, -FRAME_PX:] = True
    frame_frac = float((raw[frame] > 0).mean())

    m = IGNORE_PX + 2
    border = bool(mask[:m].any() or mask[-m:].any() or mask[:, :m].any() or mask[:, -m:].any())

    f = []
    area_frac = area / (h * w)
    if area_frac < 0.01 or area_frac > 0.90:
        f.append("area_frac")
    if second_ratio > 0.02:
        f.append("second_component")
    if hole_frac > 0.02:
        f.append("holes_filled")      # typical of glare / pale zones: check the contour
    if frame_frac > 0.002:
        f.append("leaf_near_edge")    # leaf (or debris) inside the background-reference frame
    if border:
        f.append("touches_border")
    # "exclude": flags that really invalidate the measurements (truncated leaf, extra objects, absurd area).
    # leaf_near_edge and holes_filled are informational: on a 24-image check of flagged Flavia images
    # (all of them) the contour was correct, see README.md.
    exclude = border or second_ratio > 0.02 or (area_frac < 0.01 or area_frac > 0.90)
    return {"name": name, "area_frac": round(area_frac, 4), "n_raw_components": n - 1,
            "second_ratio": round(second_ratio, 4), "hole_frac": round(hole_frac, 4),
            "solidity": round(solidity, 4), "frame_frac": round(frame_frac, 4),
            "bg_std": round(info["bg_std"], 3), "touches_border": border, "flags": "|".join(f),
            "exclude": bool(exclude)}


def make_overlay(bgr, mask):
    """
    Original image with the leaf contour drawn in red. Thickness is proportional to the
    image size, so that it is still visible after the image is reduced for the contact sheet.

    Parameters:
        - bgr (np.ndarray): original image
        - mask (np.ndarray): binary mask

    Returns:
        - overlay (np.ndarray): BGR image with the contour
    """
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ov = bgr.copy()
    cv2.drawContours(ov, cs, -1, (0, 0, 255), max(3, bgr.shape[1] // 250))
    return ov


def save_contact_sheet(tiles, names, sheet_index, output_dir, cols=6, thumb=(400, 300)):
    """
    Build and save a contact sheet. The aspect ratio of the images is preserved and each
    tile is labelled with the image name.

    Parameters:
        - tiles (list of np.ndarray): overlay images (BGR)
        - names (list of str): image names, written on each tile
        - sheet_index (int): index of the sheet, used in the file name
        - output_dir (Path): root output folder; sheets go to output_dir/contact_sheets/
        - cols (int): number of columns
        - thumb (tuple): (width, height) of each tile in px
    """
    d = output_dir / "contact_sheets"
    d.mkdir(parents=True, exist_ok=True)
    tw, th = thumb
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * th, cols * tw, 3), 255, np.uint8)
    for i, (t, nm) in enumerate(zip(tiles, names)):
        t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(t, nm, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        r, c = divmod(i, cols)
        canvas[r*th:(r+1)*th, c*tw:(c+1)*tw] = t
        cv2.rectangle(canvas, (c*tw, r*th), ((c+1)*tw-1, (r+1)*th-1), (200, 200, 200), 1)
    cv2.imwrite(str(d / f"sheet_{sheet_index:03d}.jpg"), canvas)


def main():
    args = parse_args()
    out = args.output_dir
    (out / "masks").mkdir(parents=True, exist_ok=True)
    images = sorted(args.input_dir.glob("*.jpg"))
    if args.sample:
        images = images[::max(1, len(images) // args.sample)]

    rows, tiles, names, sheet = [], [], [], 0
    per_sheet = 30
    for p in images:
        bgr = cv2.imread(str(p))
        if bgr is None:
            print(f"Could not read {p.name}, skipping.")
            continue
        mask, info = segment_leaf(bgr, args.open_px, args.close_px)
        cv2.imwrite(str(out / "masks" / (p.stem + ".png")), mask)
        rows.append(compute_metrics(mask, info, p.name))
        tiles.append(make_overlay(bgr, mask))
        names.append(p.stem)
        if len(tiles) == per_sheet:
            save_contact_sheet(tiles, names, sheet, out)
            tiles, names, sheet = [], [], sheet + 1
    if tiles:
        save_contact_sheet(tiles, names, sheet, out)

    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    print(f"Done. Processed {len(rows)} images.")


if __name__ == "__main__":
    main()
