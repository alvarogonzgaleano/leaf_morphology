#!/usr/bin/env python3
"""
Basic geometric descriptors of a leaf, computed from its binary mask.
Input masks are the PNG files produced by segment_flavia.py.
See README.md for the definition of every descriptor.
"""
import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    """
    Parse command-line arguments for the geometric descriptors script.
    Returns:
        - input_dir (Path): folder containing the binary PNG masks (leaf > 0)
        - output_dir (Path): folder where geometric_descriptors.csv will be saved
        - sigma (float): standard deviation (px) of the Gaussian used to smooth the contour
    """
    p = argparse.ArgumentParser(description="Geometric descriptors from leaf masks")
    p.add_argument("input_dir", type=Path, help="Folder containing PNG masks")
    p.add_argument("output_dir", type=Path, help="Output folder")
    p.add_argument("--sigma", type=float, default=2.0, help="Contour smoothing sigma (px)")
    return p.parse_args()

def smooth_contour(c, sigma=2.0):
    """
    Smooth the contour coordinates with a circular Gaussian filter. This removes the
    staircase effect of the pixel grid, which overestimates the perimeter by ~5-6 %
    (a perfect circle would otherwise give a circularity of about 0.90).

    Parameters:
        - c (np.ndarray): contour as returned by cv2.findContours, shape (N, 1, 2)
        - sigma (float): standard deviation of the Gaussian, in pixels

    Returns:
        - pts (np.ndarray): smoothed contour points (x, y), shape (N, 2), float64
    """
    pts = c.reshape(-1, 2).astype(np.float64)
    if len(pts) < 8:
        return pts

    half = int(math.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    k /= k.sum()
    ext = np.concatenate([pts[-half:], pts, pts[:half]])
    sx = np.convolve(ext[:, 0], k, mode="valid")
    sy = np.convolve(ext[:, 1], k, mode="valid")

    return np.stack([sx, sy], axis=1)

def closed_length(pts):
    """
    Compute the length of a closed polyline.

    Parameters:
        - pts (np.ndarray): vertices (x, y), shape (N, 2)

    Returns:
        - length (float): sum of the segment lengths, including the last-to-first segment
    """
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())

def pca_length_width(mask_pts, hull_pts):
    """
    Measure leaf length and width on the two axes given by PCA of the mask. The two
    extents are measured on the convex hull (+1 px, because contour points are pixel
    centres); the larger one is the length and the smaller one the width, so that
    length >= width always holds. The largest-variance axis is not always the one with
    the largest extent (e.g. wide or lobed leaves), hence this explicit ordering.

    Parameters:
        - mask_pts (np.ndarray): coordinates (x, y) of all mask pixels, shape (N, 2)
        - hull_pts (np.ndarray): convex hull of the contour as returned by cv2.convexHull

    Returns:
        - length (float): larger of the two extents (px)
        - width (float): smaller of the two extents (px)
        - angle (float): angle of the length direction, in degrees within [0, 180)
    """
    xy = mask_pts.astype(np.float64)
    cov = np.cov((xy - xy.mean(axis=0)).T)
    _, evecs = np.linalg.eigh(cov)
    u = evecs[:, 1]  # axis of largest variance
    v = evecs[:, 0]

    h = hull_pts.reshape(-1, 2).astype(np.float64)
    ext_u = float(np.ptp(h @ u) + 1.0)
    ext_v = float(np.ptp(h @ v) + 1.0)
    if ext_v > ext_u:
        u, ext_u, ext_v = v, ext_v, ext_u
    angle = float(np.rad2deg(math.atan2(u[1], u[0])) % 180)

    return ext_u, ext_v, angle

def feret_extremes(hull_pts, step_deg=0.5):
    """
    Compute the maximum and minimum Feret (caliper) diameters with an angular sweep.

    Parameters:
        - hull_pts (np.ndarray): convex hull of the contour as returned by cv2.convexHull
        - step_deg (float): angular step of the sweep, in degrees

    Returns:
        - max_feret (float): maximum caliper extent (px)
        - min_feret (float): minimum caliper extent (px)
    """
    pts = hull_pts.reshape(-1, 2).astype(np.float64)
    angles = np.deg2rad(np.arange(0, 180, step_deg))
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    extent = np.ptp(pts @ d.T, axis=0) + 1.0

    return float(extent.max()), float(extent.min())

def describe_mask(mask, sigma=2.0):
    """
    Compute the geometric descriptors of a single leaf mask.

    Parameters:
        - mask (np.ndarray): binary mask (uint8), leaf as >0 and background as 0
        - sigma (float): contour smoothing sigma (px), used for the perimeter

    Returns:
        - d (dict or None): None if the mask is empty. Otherwise:
            - area_px (float): number of pixels in the mask
            - perimeter_px (float): length of the smoothed external contour
            - length_px (float): larger extent along the PCA axes
            - width_px (float): smaller extent along the PCA axes
            - lw_ratio (float): length_px / width_px
            - circularity (float): 4 * pi * area / perimeter^2
            - axis_angle_deg (float): angle of the length direction
            - max_feret_px, min_feret_px (float): Feret diameters
            - rect_long_px, rect_short_px (float): sides of the minimum-area rectangle
            - n_components (int): connected components in the mask (should be 1)
    """
    m = (mask > 0).astype(np.uint8)
    n_cc, _ = cv2.connectedComponents(m, connectivity=8)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)

    area = float(m.sum())
    perimeter = closed_length(smooth_contour(c, sigma))
    hull = cv2.convexHull(c)
    ys, xs = np.nonzero(m)
    length, width, angle = pca_length_width(np.stack([xs, ys], axis=1), hull)
    max_feret, min_feret = feret_extremes(hull)
    _, (rw, rh), _ = cv2.minAreaRect(c)

    return {
        "area_px": area,
        "perimeter_px": perimeter,
        "length_px": length,
        "width_px": width,
        "lw_ratio": length / width if width > 0 else float("nan"),
        "circularity": 4 * math.pi * area / perimeter ** 2 if perimeter > 0 else float("nan"),
        "axis_angle_deg": angle,
        "max_feret_px": max_feret,
        "min_feret_px": min_feret,
        "rect_long_px": float(max(rw, rh)),
        "rect_short_px": float(min(rw, rh)),
        "n_components": n_cc - 1,
    }

def main():
    args = parse_args()
    files = sorted(args.input_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"No PNG files found in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for f in files:
        mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        d = describe_mask(mask, args.sigma)
        if d is None:
            print(f"[warning] empty mask: {f.name}")
            continue
        rows.append({"image": f.stem, **d})

    out_path = args.output_dir / "geometric_descriptors.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} masks processed -> {out_path}")


if __name__ == "__main__":
    main()