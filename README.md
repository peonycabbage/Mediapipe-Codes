# MediaPipe Keypoint Extraction for Sign Language Recognition

Python research scripts that turn sign language videos and frame images into landmark features using MediaPipe Holistic, OpenCV, and NumPy. This repository contains the feature extraction code associated with [*The Influence of Iconicity in Transfer Learning for Sign Language Recognition*](https://doi.org/10.1007/978-3-031-70239-6_16) (NLDB 2024).

## How the features are built

The scripts read frames with OpenCV, convert them from BGR to RGB for MediaPipe Holistic, and save the extracted coordinates as per-frame NumPy `.npy` files.

The Arabic and Chinese extractors encode each saved frame as a **138-value vector**. It contains 21 landmarks per hand, followed by the left shoulder, right shoulder, left wrist, and right wrist. Each landmark contributes `(x, y, z)` coordinates. If a hand is missing, its 63 values are filled with zeros. The PHOENIX extractor also appends selected face landmarks.

## Scripts

| Script | Input and purpose |
| --- | --- |
| [`convert2keypArabic-local.py`](convert2keypArabic-local.py) | Arabic Sign Language frame folders → NumPy keypoints |
| [`convert2keypArabic.py`](convert2keypArabic.py) | Arabic Sign Language frame folders → keypoints with optional pose overlays |
| [`convert2keypCSLvid.py`](convert2keypCSLvid.py) | Chinese Sign Language video files → keypoints with optional pose overlays |
| [`convert2keypChinese-local - Copy.py`](convert2keypChinese-local%20-%20Copy.py) | Chinese Sign Language frame folders → NumPy keypoints |
| [`convert2keypChinese-outlier60.py`](convert2keypChinese-outlier60.py), [`convert2keypChinese-outlier70.py`](convert2keypChinese-outlier70.py) | Chinese Sign Language frame folders with experimental outlier filtering |
| [`convert2keypPhoenix.py`](convert2keypPhoenix.py) | PHOENIX-2014-T frame folders → hand, upper-body, and selected face keypoints |

## Quick start

Use Python 3.10–3.12 on a platform supported by MediaPipe. Install the dependencies in a virtual environment.

```bash
python -m venv .venv
# Activate the environment for your shell before installing:
python -m pip install -r requirements.txt
python extract_keypoints.py --help
```

```bash
python extract_keypoints.py data/frames output/keypoints
python extract_keypoints.py data/videos output/keypoints --mode videos --overlays output/overlays
python extract_keypoints.py data/phoenix output/phoenix --layout phoenix
```

Pass a single file or a directory. Directories are scanned recursively, and output paths mirror the input tree. For example, an input image at `train/clip/frame.jpg` produces `train/clip/frame.jpg.npy`. A video at `train/clip.mp4` produces files such as `train/clip.mp4/frame_000001.npy`. Video filenames use the original frame number, so gaps indicate skipped frames.

The standard layout has 138 values. The PHOENIX layout has 171 values and appends 11 facial points from the pose model. Frames without a detected pose are skipped. The main command writes `float32` by default and accepts `--dtype float64`. Use a fresh output directory when changing filters or layouts to avoid mixing results across runs.

The original script names remain available as shortcuts with the same input, output, and optional overlay arguments. The Chinese outlier scripts use wrist thresholds of 0.60 and 0.70, respectively. They keep a frame when either wrist's normalised y coordinate is below the threshold. The main command exposes this filter as `--wrist-threshold`.

## Research context

The study examined whether iconic similarities between signs help transfer learning across sign languages. The experiments used a multilayer perceptron for spatial features and a gated recurrent unit for temporal information. The paper reports improvements of **7.02%** for Chinese-to-Arabic transfer and **1.07%** for Greek-to-Flemish transfer in the iconic transfer experiments. Use the paper for the training methodology and evaluation details.

## Citation

```bibtex
@inproceedings{artiaga2024iconicity,
  author    = {Artiaga, Keren and Lynch, Conor and Afli, Haithem and Hasanuzzaman, Mohammed},
  title     = {The Influence of Iconicity in Transfer Learning for Sign Language Recognition},
  booktitle = {Natural Language Processing and Information Systems},
  year      = {2024},
  pages     = {226--240},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-031-70239-6_16}
}
```
