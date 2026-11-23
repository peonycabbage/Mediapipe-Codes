"""Extract landmarks from PHOENIX-2014-T frame images."""

from extract_keypoints import main


if __name__ == "__main__":
    raise SystemExit(main(layout="phoenix", dtype="float64"))
