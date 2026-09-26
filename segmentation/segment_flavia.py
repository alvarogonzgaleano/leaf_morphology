import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    """
    Parse command-line arguments for the Flavia batch segmentation script.
    Returns:
        - input_dir   (Path): folder containing Flavia JPEG images
        - output_dir (Path): folder where masks, contact sheets and metrics will be saved
        - open_px (int): structuring element radius for morphological opening
        - close_px (int): structuring element radius for morphological closing
        - sample (int): if set, process only this many images (uniform sampling)
    """
    p = argparse.ArgumentParser(description="Batch segmentation - Flavia dataset")
    p.add_argument("input_dir", type=Path, help="Folder containing Flavia images")
    p.add_argument("output_dir", type=Path, help="Output folder")
    p.add_argument("--open-px", type=int, default=5, help="Morphological opening radius")
    p.add_argument("--close-px", type=int, default=9, help="Morphological closing radius")
    p.add_argument("--sample", type=int, default=None, help="Process only N images")
    return p.parse_args()

def segment_leaf(bgr, open_px=5, close_px=9):
    """
    Segment a single leaf image using Otsu thresholding and morphological operations.

    Parameters:
        - bgr (np.ndarray): input image in BGR format as loaded by cv2.imread
        - open_px (int): radius of the structuring element for morphological opening
        - close_px (int): radius of the structuring element for morphological closing
    
    Returns:
        - mask (np.ndarray): binary mask (uint8, 0/255) with the leaf as white and background as black
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*open_px+1, 2*open_px+1))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_px+1, 2*close_px+1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    flood = mask.copy()
    cv2.floodFill(flood, None, (0, 0), 255)
    mask = mask | ~flood

    return mask

def compute_metrics(mask, name):
    """
    Compute quality metrics and generate warning flags for a segmented leaf mask

    Parameters:
        - mask (np.ndarray): binary mask (uint8, 0/255) as returned by segmented_leaf
        - name (str): image filename, used to identify the row in the output CSV
    
    Returns:
        - dict (Dictionary with the following fields):
            - name (str): image filename
            - area_frac (float): fraction of image pixels classified as leaf
            - n_components (int): number of connected components in the mask
            - second_ratio (float): area of second largest component (close to 0 => probably noise, close to 1 => is the leaf divided? is there another object?)
            - solidity (float): mask area / convex hull area
            - touches_border (bool): whether the mask touches any image edge
            - flags (str): pipe-separated list of triggered warning flags
    """
    h, w = mask.shape
    area = np.sum(mask == 255)
    area_frac = area / (h * w)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    n_components = n_labels - 1 # exclude background

    areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True) if n_labels > 1 else [0]
    second_ratio = (areas[1] / areas[0]) if len(areas) > 1 else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
    else:
        solidity = 0.0

    border = (
        np.any(mask[0, :] == 255) or
        np.any(mask[-1, :] == 255) or
        np.any(mask[:, 0] == 255) or
        np.any(mask[:, -1] == 255)
    )

    flags = []
    if area_frac < 0.05 or area_frac > 0.95:
        flags.append("area_frac")
    if n_components > 1:
        flags.append("n_components")
    if second_ratio > 0.1:
        flags.append("second_ratio")
    if solidity < 0.7:
        flags.append("solidity")
    if border:
        flags.append("touches_border")

    return {
        "name": name,
        "area_frac": round(area_frac, 4),
        "n_components": n_components,
        "second_ratio": round(second_ratio, 4),
        "touches_border": border,
        "flags": "|".join(flags)
    }

def save_outputs(bgr, mask, name, output_dir):
    """
    Save the binary mask and the original image with the leaf contour overlaid.

    Parameters:
        - bgr (np.ndarray): original image in BGR format as loaded by cv2.imread
        - mask (np.ndarray): binary mask (uint8, 0/255) as returned by segment_leaf
        - name (str): image filename, used to name the output files
        - output_dir (path): root output folder. Masks are saved under output_dir/masks/
    
    Returns:
        overlay (np.ndarray): original image with leaf contour drawn in green, used later to build the contact sheets
    """
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(masks_dir / name.replace(".jpg", ".png")), mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    return overlay

def save_contact_sheet(overlays, sheet_index, output_dir, cols=9):
    """
    Build and save a contact sheet from a list of overlay images

    Parameters:
        - overlays (list of np.ndarray): list of BGR images with the leaf contour drawn, as returned by save_outputs
        - sheet_index (int): index of the contact sheet, used to name the output file
        - output_dir (Path): root output folder. Contact sheets are saved under output_dir/contact_sheets/
        - cols (int): number of columns in the grid. Rows are computed automatically
    """
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)

    thumb_h, thumb_w = 200, 150
    thumbs = [cv2.resize(img, (thumb_w, thumb_h)) for img in overlays]

    rows = (len(thumbs) + cols - 1) // cols
    canvas_h = rows * thumb_h
    canvas_w = cols * thumb_w
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        canvas[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb

    out_path = sheets_dir / f"sheet_{sheet_index:03d}.jpg"
    cv2.imwrite(str(out_path), canvas)

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(args.input_dir.glob("*.jpg"))
    if args.sample:
        step = max(1, len(images) // args.sample)
        images = images[::step]

    rows = []
    overlays = []
    sheet_index = 0

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"Could not read {img_path.name}, skipping.")
            continue

        mask = segment_leaf(bgr, args.open_px, args.close_px)
        metrics = compute_metrics(mask, img_path.name)
        overlay = save_outputs(bgr, mask, img_path.name, args.output_dir)

        rows.append(metrics)
        overlays.append(overlay)

        if len(overlays) == 90:
            save_contact_sheet(overlays, sheet_index, args.output_dir)
            overlays = []
            sheet_index += 1

    if overlays:
        save_contact_sheet(overlays, sheet_index, args.output_dir)

    pd.DataFrame(rows).to_csv(args.output_dir / "metrics.csv", index=False)
    print(f"Done. Processed {len(rows)} images.")

if __name__ == "__main__":
    main()
