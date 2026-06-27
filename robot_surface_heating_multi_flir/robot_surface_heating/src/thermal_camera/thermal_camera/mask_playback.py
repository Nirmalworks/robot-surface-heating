#!/usr/bin/env python3
import os
import cv2
import time
import argparse
import glob

def play_frames(root_dir, camera_count=4, fps=5, save_video=False, video_path="masked_output.mp4"):
    print(f"Loading from: {root_dir}")

    # Load sorted timestamped file lists for each camera
    frame_lists = []
    for i in range(camera_count):
        cam_dir = os.path.join(root_dir, f"camera_{i}")
        files = sorted(glob.glob(os.path.join(cam_dir, "*.png")))
        if not files:
            print(f"No frames found in {cam_dir}")
            return
        frame_lists.append(files)

    frame_count = min(len(lst) for lst in frame_lists)
    print(f"Found {frame_count} synchronized frames.")

    # Prepare video writer (lazy init after first frame to get shape)
    out = None

    for idx in range(frame_count):
        imgs = [cv2.imread(frame_lists[i][idx]) for i in range(camera_count)]
        resized = [cv2.resize(img, (640, 480)) for img in imgs]

        # Arrange in 2x2 grid
        top_row = cv2.hconcat([resized[0], resized[1]])
        bottom_row = cv2.hconcat([resized[2], resized[3]])
        grid = cv2.vconcat([top_row, bottom_row])

        # Initialize video writer after first frame
        if save_video and out is None:
            height, width, _ = grid.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            print(f"Saving video to: {video_path}")

        if save_video:
            out.write(grid)

        cv2.imshow("Masked Thermal Frames", grid)
        key = cv2.waitKey(int(1000 / fps))
        if key == ord('q'):
            break

    if out:
        out.release()
        print("Video saved.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="/tmp/masked_frames", help="Root directory for saved frames")
    parser.add_argument("--cameras", type=int, default=4, help="Number of thermal cameras")
    parser.add_argument("--fps", type=int, default=9, help="Playback FPS")
    parser.add_argument("--save", action="store_true", help="Save video output")
    parser.add_argument("--video", type=str, default="masked_output.mp4", help="Path to output video file")
    args = parser.parse_args()

    play_frames(args.dir, args.cameras, args.fps, args.save, args.video)
