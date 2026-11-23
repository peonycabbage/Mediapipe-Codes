# MediaPipe Keypoint Extraction for Sign Language Recognition

Python research scripts that turn sign language videos and frame images into landmark features using MediaPipe Holistic, OpenCV, and NumPy. This repository contains the feature extraction code associated with [*The Influence of Iconicity in Transfer Learning for Sign Language Recognition*](https://doi.org/10.1007/978-3-031-70239-6_16) (NLDB 2024).

## How the features are built

The scripts read frames with OpenCV, convert them from BGR to RGB for MediaPipe Holistic, and save the extracted coordinates as per-frame NumPy `.npy` files.

The Arabic and Chinese extractors encode each saved frame as a **138-value vector**. It contains 21 landmarks per hand, followed by the left shoulder, right shoulder, left wrist, and right wrist. Each landmark contributes `(x, y, z)` coordinates. If a hand is missing, its 63 values are filled with zeros. The PHOENIX extractor also appends selected face landmarks.

## Scripts

| Script | Input and purpose |
| --- | --- |
| [`convert2keypArabic-local.py`](convert2keypArabic-local.py) | Arabic Sign Language frame folders → NumPy keypoints |
| [`convert2keypArabic.py`](convert2keypArabic.py) | Arabic Sign Language frame folders → keypoints and pose overlays |
| [`convert2keypCSLvid.py`](convert2keypCSLvid.py) | Chinese Sign Language video files → keypoints and pose overlays |
| [`convert2keypChinese-local - Copy.py`](convert2keypChinese-local%20-%20Copy.py) | Chinese Sign Language frame folders → NumPy keypoints |
| [`convert2keypChinese-outlier60.py`](convert2keypChinese-outlier60.py), [`convert2keypChinese-outlier70.py`](convert2keypChinese-outlier70.py) | Chinese Sign Language frame folders with experimental outlier filtering |
| [`convert2keypPhoenix.py`](convert2keypPhoenix.py) | PHOENIX-2014-T frame folders → hand, upper-body, and selected face keypoints |

## Running the frame extractor

1. Use a Python environment with NumPy, OpenCV (`opencv-python`), and a MediaPipe release that provides the legacy `mp.solutions.holistic` API.
2. Arrange your frame images as `<dataset>/<split>/<sign-class>/<clip>/<frame-image>`.
3. In `convert2keypArabic-local.py`, set `path` to your dataset root and `newcoordspath` to your output root. Run:

   ```bash
   python convert2keypArabic-local.py
   ```

The extractor mirrors the input folder structure under the output root and writes one `.npy` file per processed frame. The original frame extractors assume a pose is detected and can stop on frames without one.

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
