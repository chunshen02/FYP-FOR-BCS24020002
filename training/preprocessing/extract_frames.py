"""
extract_frames.py

Usage:
    python extract_frames.py --src <video_dir> --dst <frames_dir> --interval <N>

Arguments:
    --src       Path to the source directory containing .mp4 video files
    --dst       Path to the output directory where extracted frames will be saved
    --interval  Extract one frame every N frames (default: 10)
"""

import os
import argparse
import cv2


def extract_frames(src_dir: str, dst_dir: str, interval: int = 10) -> None:
    os.makedirs(dst_dir, exist_ok=True)

    video_files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".mp4", ".avi", ".mov"))
    ]

    total_videos = len(video_files)
    total_frames = 0
    total_skipped = 0

    for video_file in video_files:
        video_id = os.path.splitext(video_file)[0]
        video_path = os.path.join(src_dir, video_file)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARNING] Cannot open video: {video_path}. Skipping.")
            total_skipped += 1
            continue

        frame_index = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % interval == 0:
                out_filename = f"{video_id}_{frame_index:05d}.jpg"
                out_path = os.path.join(dst_dir, out_filename)
                cv2.imwrite(out_path, frame)
                saved_count += 1
            frame_index += 1

        cap.release()
        total_frames += saved_count

    print(f"\n[DONE] Videos processed: {total_videos - total_skipped}/{total_videos}")
    print(f"[DONE] Total frames extracted: {total_frames}")
    print(f"[DONE] Videos skipped: {total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video files.")
    parser.add_argument("--src", required=True, help="Source directory of video files")
    parser.add_argument("--dst", required=True, help="Output directory for extracted frames")
    parser.add_argument("--interval", type=int, default=10, help="Extract every N frames")
    args = parser.parse_args()

    extract_frames(args.src, args.dst, args.interval)
