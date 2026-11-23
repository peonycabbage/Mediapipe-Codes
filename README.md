# MediaPipe Keypoint Extraction for Sign Language Recognition

Extract hand and upper-body landmarks from sign language videos or frame images using MediaPipe Holistic. The extractor writes one NumPy feature vector per retained frame and can save pose overlays for visual inspection. This research utility accompanies [*The Influence of Iconicity in Transfer Learning for Sign Language Recognition*](https://doi.org/10.1007/978-3-031-70239-6_16) (NLDB 2024).

## Quick start

Use Python 3.10–3.12 on a platform supported by the pinned MediaPipe release. From the repository root, create a virtual environment.

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in Bash or Zsh. Then install the dependencies and try the extractor on one image.

```bash
python -m pip install -r requirements.txt
python extract_keypoints.py path/to/frame.jpg output/keypoints
```

Use an image with a visible upper body. A successful run prints the number of saved and skipped frames and writes `output/keypoints/frame.jpg.npy`.

## Batch extraction

```bash
python extract_keypoints.py data/frames output/frames
python extract_keypoints.py data/videos output/videos --mode videos --overlays output/overlays
python extract_keypoints.py data/phoenix output/phoenix --layout phoenix
python extract_keypoints.py --help
```

Pass a file or a directory. The extractor scans directories recursively and follows the input directory structure. Keep output directories outside the input directory. Use a fresh output directory when changing layouts or filters so results from different runs stay separate.

| Input relative to the dataset root | Keypoints relative to the output root |
| --- | --- |
| `train/clip/frame.jpg` | `train/clip/frame.jpg.npy` |
| `train/clip.mp4` | `train/clip.mp4/frame_000001.npy`, then one file per retained frame |

Video frames are numbered from 1 in decode order. Gaps indicate skipped frames. With `--overlays`, corresponding PNGs are saved under the overlay directory.

## Feature vectors

| Layout | Values | Feature order |
| --- | ---: | --- |
| `standard` | 138 | Left hand (21 landmarks), right hand (21), left shoulder, right shoulder, left wrist, right wrist |
| `phoenix` | 171 | Standard layout followed by 11 facial points from the pose model |

Every landmark contributes `(x, y, z)` in MediaPipe's image landmark coordinates. Missing hands are zero-filled. Frames without a detected pose or with unreadable images are skipped. Arrays use `float32` by default. Select `--dtype float64` for pipelines that expect double precision.

## Migrating from the original scripts

Run `python extract_keypoints.py INPUT OUTPUT` with the matching options below. Add `--overlays DIR` to save pose overlays.

| Previous script | Equivalent options |
| --- | --- |
| `convert2keypArabic-local.py`, `convert2keypChinese-local - Copy.py` | Default options |
| `convert2keypArabic.py` | `--dtype float64` |
| `convert2keypCSLvid.py` | `--mode videos --dtype float64` |
| `convert2keypChinese-outlier60.py` | `--wrist-threshold 0.60` |
| `convert2keypChinese-outlier70.py` | `--wrist-threshold 0.70` |
| `convert2keypPhoenix.py` | `--layout phoenix --dtype float64` |

The wrist filter keeps frames where either wrist's normalised y coordinate is below the given threshold.

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
