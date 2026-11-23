"""Extract MediaPipe Holistic landmarks from images and videos."""

import argparse
from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
BODY_POINTS = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_WRIST", "RIGHT_WRIST")
FACE_POINTS = (
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER",
    "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT",
)


def coordinates(landmarks):
    return [value for point in landmarks for value in (point.x, point.y, point.z)]


def extract_keypoints(result, landmark_ids, layout):
    """Return the landmark coordinates in the original feature order."""
    left = result.left_hand_landmarks
    right = result.right_hand_landmarks
    pose = result.pose_landmarks.landmark
    names = BODY_POINTS + FACE_POINTS if layout == "phoenix" else BODY_POINTS
    return (
        (coordinates(left.landmark) if left else [0.0] * 63)
        + (coordinates(right.landmark) if right else [0.0] * 63)
        + coordinates(pose[getattr(landmark_ids, name)] for name in names)
    )


def video_frames(cv2, source):
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise OSError(f"Cannot open video: {source}")
        frame_number = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_number += 1
            yield f"frame_{frame_number:06d}", frame
        if frame_number == 0:
            raise OSError(f"No readable frames in video: {source}")
    finally:
        capture.release()


def save_overlay(cv2, solutions, image, result, output):
    if result.segmentation_mask is not None:
        image[result.segmentation_mask <= 0.1] = 192
    solutions.drawing_utils.draw_landmarks(
        image, result.pose_landmarks, solutions.holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=solutions.drawing_styles.get_default_pose_landmarks_style(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise OSError(f"Cannot write overlay: {output}")


def process_frames(model, frames, relative_dir, args, cv2, np, solutions):
    saved = skipped = 0
    landmark_ids = solutions.pose.PoseLandmark
    for name, image in frames:
        if image is None:
            print(f"Skipping unreadable image: {relative_dir / name}")
            skipped += 1
            continue
        result = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if result.pose_landmarks is None:
            skipped += 1
            continue
        if args.wrist_threshold is not None:
            pose = result.pose_landmarks.landmark
            wrist_y = min(pose[landmark_ids.LEFT_WRIST].y, pose[landmark_ids.RIGHT_WRIST].y)
            if wrist_y >= args.wrist_threshold:
                skipped += 1
                continue

        values = extract_keypoints(result, landmark_ids, args.layout)
        output = args.output / relative_dir / f"{name}.npy"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, np.asarray(values, dtype=args.dtype))
        if args.overlays:
            overlay = args.overlays / relative_dir / f"{name}.png"
            save_overlay(cv2, solutions, image, result, overlay)
        saved += 1
    return saved, skipped


def run(args, sources, source_root, cv2, np, solutions):
    holistic = solutions.holistic.Holistic
    saved = skipped = 0
    if args.mode == "frames":
        with holistic(
            static_image_mode=True, model_complexity=2,
            enable_segmentation=bool(args.overlays),
        ) as model:
            for source in sources:
                relative = source.relative_to(source_root)
                frames = [(relative.name, cv2.imread(str(source)))]
                written, omitted = process_frames(
                    model, frames, relative.parent, args, cv2, np, solutions
                )
                saved += written
                skipped += omitted
    else:
        for source in sources:
            # Reset tracking state at the start of each clip.
            with holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as model:
                written, omitted = process_frames(
                    model, video_frames(cv2, source), source.relative_to(source_root),
                    args, cv2, np, solutions,
                )
                saved += written
                skipped += omitted
    return saved, skipped


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input", type=Path, help="Input file or dataset directory")
    parser.add_argument("output", type=Path, help="Directory for NumPy keypoints")
    parser.add_argument("--mode", choices=("frames", "videos"), default="frames",
                        help="Read frame images or decode video files")
    parser.add_argument("--layout", choices=("standard", "phoenix"), default="standard",
                        help="Landmark vector layout")
    parser.add_argument("--wrist-threshold", type=float,
                        help="Keep a frame if either wrist y coordinate is below this value")
    parser.add_argument("--overlays", type=Path, help="Optional directory for pose overlays")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32",
                        help="NumPy output precision")
    args = parser.parse_args()
    if args.wrist_threshold is not None and not 0 <= args.wrist_threshold <= 1:
        parser.error("--wrist-threshold must be between 0 and 1")

    args.input = args.input.resolve()
    args.output = args.output.resolve()
    if not args.input.exists():
        parser.error(f"Input does not exist: {args.input}")
    if args.overlays:
        args.overlays = args.overlays.resolve()
        if args.overlays == args.output:
            parser.error("Choose separate directories for keypoints and overlays")
    destinations = [path for path in (args.output, args.overlays) if path]
    if args.input.is_dir() and any(path.is_relative_to(args.input) for path in destinations):
        parser.error("Output and overlay directories must be outside the input directory")

    source_root = args.input.parent if args.input.is_file() else args.input
    extensions = IMAGE_EXTENSIONS if args.mode == "frames" else VIDEO_EXTENSIONS
    candidates = [args.input] if args.input.is_file() else args.input.rglob("*")
    sources = sorted(
        path for path in candidates if path.suffix.lower() in extensions and path.is_file()
    )
    if not sources:
        parser.error(f"No supported {args.mode} found in {args.input}")

    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        solutions = mp.solutions
    except (ImportError, AttributeError) as error:
        parser.exit(
            1, f"Cannot load extraction dependencies: {error}\n"
            "Install them with python -m pip install -r requirements.txt\n",
        )

    try:
        saved, skipped = run(args, sources, source_root, cv2, np, solutions)
    except OSError as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved {saved} frame(s), skipped {skipped} frame(s).")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
