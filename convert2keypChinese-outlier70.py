"""Extract landmarks from Chinese frames with a wrist threshold of 0.70."""

from extract_keypoints import main


if __name__ == "__main__":
    raise SystemExit(main(wrist_threshold=0.70))
