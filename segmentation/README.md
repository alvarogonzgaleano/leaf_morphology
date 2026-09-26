# segment_flavia.py

Batch leaf segmentation for the Flavia dataset. Produces binary masks, contact sheets
with the leaf contour overlaid, and a CSV with per-image quality metrics.

## Requirements

```bash
pip install opencv-python numpy pandas
```

## Usage

```bash
python segment_flavia.py <input_dir> <output_dir> [--open-px N] [--close-px N] [--sample N]
```

Example — process 50 images to check results before running the full dataset:

```bash
python segment_flavia.py data/Flavia/ output/ --sample 50
```

Example — full run with custom morphological parameters:

```bash
python segment_flavia.py data/Flavia/ output/ --open-px 7 --close-px 11
```

## Arguments

| Argument      | Default | Description                                              |
|---------------|---------|----------------------------------------------------------|
| `input_dir`   | —       | Folder containing Flavia JPEG images                     |
| `output_dir`  | —       | Folder where all outputs will be saved                   |
| `--open-px`   | 5       | Structuring element radius for morphological opening     |
| `--close-px`  | 9       | Structuring element radius for morphological closing     |
| `--sample`    | None    | If set, process only N images (uniform sampling)         |

## Output structure


## Quality control

After running the script, open `metrics.csv` and filter by the `flags` column to find
images that may have segmentation problems. Each flag signals a specific type of failure:

| Flag             | Meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| `area_frac`      | Mask covers less than 5% or more than 95% of the image                 |
| `n_components`   | More than one connected component in the mask                           |
| `second_ratio`   | Second largest component is more than 10% the size of the largest       |
| `solidity`       | Mask area / convex hull area below 0.7 — unusual shape                 |
| `touches_border` | Mask reaches the image edge — leaf may be cut off                       |

Inspect flagged images visually using the contact sheets in `contact_sheets/`.
The green contour shows exactly what the segmentation captured.

When adjusting `--open-px` and `--close-px`, record what values were tried and their
effect on the flags — this information is needed for the thesis write-up.