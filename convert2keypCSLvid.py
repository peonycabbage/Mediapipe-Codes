"""Extract landmarks from Chinese Sign Language videos."""

from extract_keypoints import main


if __name__ == "__main__":
    raise SystemExit(main(mode="videos", dtype="float64"))
