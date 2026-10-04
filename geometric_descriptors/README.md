# Geometric descriptors

Computes basic geometric descriptors of a leaf from its binary segmentation
mask. It is the stage that follows `segmentation/`: that stage produces the
masks, this one reads them.

## Requirements

Python 3, OpenCV (`opencv-python`) and NumPy.

## Usage

```
python geometric_descriptors.py input_dir output_dir [--sigma 2.0]
```

- `input_dir`: folder with binary PNG masks (leaf = pixels > 0), such as the
  ones written by `segment_flavia.py`.
- `output_dir`: folder where `geometric_descriptors.csv` is saved (created if
  it does not exist).
- `--sigma`: standard deviation, in pixels, of the Gaussian used to smooth the
  contour before measuring the perimeter (default 2.0).

One row per mask. Empty masks are skipped with a warning.

## Descriptors

| Column | Definition | Unit |
|---|---|---|
| `area_px` | Number of pixels in the mask | px² |
| `perimeter_px` | Length of the external contour, after smoothing (see below) | px |
| `length_px` | Larger of the two extents of the leaf measured along the PCA axes of the mask | px |
| `width_px` | Smaller of those two extents | px |
| `lw_ratio` | `length_px / width_px` (always >= 1) | – |
| `circularity` | `4 * pi * area / perimeter^2` (1 for a circle, lower for elongated or serrated shapes) | – |

Auxiliary columns, useful to check how sensitive length and width are to
their definition:

| Column | Definition |
|---|---|
| `axis_angle_deg` | Angle of the length direction, in [0, 180) |
| `max_feret_px`, `min_feret_px` | Maximum and minimum Feret (caliper) diameters |
| `rect_long_px`, `rect_short_px` | Sides of the minimum-area bounding rectangle |
| `n_components` | Connected components in the mask (should be 1) |

## Design decisions

- **Perimeter is measured on a smoothed contour.** The raw pixel contour has
  a staircase effect that overestimates the perimeter by about 5-6 %, so a
  perfect circle would get a circularity of about 0.90 instead of 1.00. The
  contour coordinates are smoothed with a circular Gaussian filter
  (`--sigma`, default 2 px, `smooth_contour`). On the synthetic test shapes this
  brings the perimeter error below 1 %. Larger values of `--sigma` would start
  to erase the serration of the leaf margin, so it should be tuned on real
  data.
- **Length and width are measured on the PCA axes, not on the Feret
  diameters.** The maximum Feret diameter of an elongated, roughly
  rectangular shape is its diagonal, which makes the perpendicular width too
  small. Both Feret diameters are still exported for comparison.
- **Length is always the larger extent.** The direction of largest variance
  is not always the direction of largest extent (wide, lobed or compound
  shapes), so the two extents along the PCA axes are ordered explicitly and
  `lw_ratio >= 1` by construction.
- **Extents are measured on the convex hull** with a +1 px correction, since
  contour points are pixel centres.

## Limitations

- **Units.** Area, perimeter, length and width are in pixels. The datasets
  have no metric scale (see the dataset comparison), so these values are not
  comparable across datasets. Compare `lw_ratio` and `circularity`, or
  normalise by `sqrt(area)`.
- **Petiole.** It enters the mask and affects length, width, perimeter and
  circularity. In a synthetic test, adding a thin petiole to an ellipse with
  `lw_ratio = 2.0` raised it to 2.76. If the petiole is kept in some images
  and removed in others, the descriptors are not comparable between them.
- **Segmentation quality.** All values are computed on the mask. Errors in
  the mask (lost margin teeth, shadows, stray components) propagate directly
  to the descriptors.
- **Length/width are not botanical base-to-apex measurements.** They come
  from the silhouette only, so for wide, lobed or compound leaves the
  length direction is not necessarily the one a botanist would pick (e.g.
  image 3328 of Flavia: two wide lateral lobes plus a long tip). In near
  isotropic shapes `axis_angle_deg` is not meaningful. For such leaves
  `lw_ratio` stays close to 1 while the Feret ratio
  (`max_feret_px / min_feret_px`, computable from the CSV) is higher
  (1.03 vs 1.29 in 3328), so the choice of definition matters.
- **Curved leaves.** Width is measured perpendicular to a straight axis, so a
  strongly curved narrow leaf (e.g. 2623) gets a slightly inflated width
  and an `lw_ratio` lower than the one measured along the curve.

## Validation

Tested on synthetic masks with analytic values (circle, ellipses at several
orientations, rectangle):

| Shape | Quantity | Expected | Obtained |
|---|---|---|---|
| Circle, r = 200 | circularity | 1.000 | 1.004 |
| Ellipse 300 x 100 (semi-axes), rotated 0/30/77° | `lw_ratio` | 3.000 | 2.990 / 2.978 / 2.979 |
| Ellipse 300 x 100, rotated 0/30/77° | circularity | 0.663 | 0.667 |
| Rectangle 400 x 100 | length / width | 400 / 100 | 400.0 / 100.0 |

**Not yet validated on real Flavia masks.** Suggested checks on the output
CSV:

- `n_components` should be 1 for (almost) every row.
- `circularity` should lie in (0, 1]; values above ~1.05 indicate a contour
  problem.
- `length_px / max_feret_px` should be close to 1 for most leaves.
- Visually inspect the images at the extremes of `lw_ratio`, `circularity`
  and `area_px`, which is where segmentation failures show up.

## Planned

Contour descriptors (solidity, convexity), lobe counting and margin
complexity.