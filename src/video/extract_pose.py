import numpy as np
import cv2
import argparse
import mediapipe as mp
import csv
import pandas as pd
from pathlib import Path
import subprocess
from fractions import Fraction

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

POSE_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]

def parse_arg():
    parser = argparse.ArgumentParser(
        description="Analyze running video"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video"
    )

    parser.add_argument(
        "--keypoints",
        type=str,
        required=True,
        help="Path to output csv file"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output video"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model file"
    )
    parser.add_argument(
        "--output_meta",
        type=str,
        required=True,
        help="Path to outputh csv meta file "
    )
    parser.add_argument(
        "--normalized_video",
        type=str,
        required=True,
        help="Path to normalized CFR video"
    )

    return parser.parse_args()

def get_nominal_fps(video_path: Path) -> str:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    frame_rate = result.stdout.strip()
    fps = float(Fraction(frame_rate))

    if not 1 <= fps <= 240:
        raise ValueError(
            f"Подозрительная номинальная частота кадров: {frame_rate}"
        )

    return frame_rate

def normalize_video(
    input_path: Path,
    output_path: Path,
    target_fps: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-vf",
        f"fps=fps={target_fps}",
        "-fps_mode:v",
        "cfr",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(output_path),
    ]

    subprocess.run(command, check=True)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
    pose_connection_style = drawing_utils.DrawingSpec(
        color=(0, 255, 0),
        thickness=2
    )

    for pose_landmarks in pose_landmarks_list:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=pose_landmarks,
            connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=pose_landmark_style,
            connection_drawing_spec=pose_connection_style
        )

    return annotated_image


def write_landmark_rows(detection_result, frame_index, timestamp_ms, writer):    
    all_frame_rows = []

    if detection_result.pose_landmarks:
        normalized_landmarks = detection_result.pose_landmarks[0]

        for idx, landmark in enumerate(normalized_landmarks):
            landmark_name = POSE_LANDMARK_NAMES[idx]

            x = landmark.x
            y = landmark.y
            z = landmark.z
            visibility = landmark.visibility
            presence = landmark.presence

            row = [
                frame_index,
                timestamp_ms,
                1,
                0,
                idx,
                landmark_name,
                x,
                y,
                z,
                visibility,
                presence
            ]
            all_frame_rows.append(row)
        
        writer.writerows(all_frame_rows)
    else:
        row = [
            frame_index,
            timestamp_ms,
            0,
            -1,
            -1,
            '',
            '',
            '',
            '',
            '',
            '',
        ]
        all_frame_rows.append(row)

        writer.writerows(all_frame_rows)
          

def main():
    args = parse_arg()

    source_video_path = Path(args.video)
    normalized_video_path = Path(args.normalized_video)
    csv_path = Path(args.keypoints)
    output_video_path = Path(args.output)
    output_csv = Path(args.output_meta)
    model_path = Path(args.model)

    for output_path in (
        csv_path,
        output_video_path,
        output_csv,
        normalized_video_path,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    target_fps = get_nominal_fps(source_video_path)

    normalize_video(
        input_path=source_video_path,
        output_path=normalized_video_path,
        target_fps=target_fps,
    )

    data_column = ['frame_index','timestamp_ms', 'pose_detected', 'pose_id', 'landmark_id','landmark_name','x','y','z','visibility','presence']

    cap = None
    out = None
    detector = None

    try:
        cap = cv2.VideoCapture(str(normalized_video_path))

        if not cap.isOpened():
            raise RuntimeError("Не удалось открыть нормализованное видео")


        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    

        if fps <= 0:
            raise ValueError("Fps неизвестен")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))


        if not out.isOpened():
            raise RuntimeError("Не удалось открыть выходное видео")


        print(fps, width, height)

        base_options = python.BaseOptions(model_asset_path=str(model_path))

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses = 1,
            output_segmentation_masks=False,

            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        frame_index = 0

        with open(csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(data_column)

            while(True):
                ret, frame = cap.read()

                if not ret:
                    print("Видео закончилось или кадр не прочитан")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(frame_index * 1000 / fps)

                detection_result = detector.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                write_landmark_rows(detection_result, frame_index, timestamp_ms, writer)

                annotated_rgb = draw_landmarks_on_image(
                    rgb_frame,
                    detection_result
                )

                annotated_bgr = cv2.cvtColor(
                    annotated_rgb,
                    cv2.COLOR_RGB2BGR
                )

                out.write(annotated_bgr)

                frame_index += 1 

            df = pd.DataFrame({
                'frame_count': [frame_index],
                'fps': [fps],
                'width': [width],
                'height': [height]
            })
            df.to_csv(str(output_csv), index=False)

    finally:
        if cap is not None:
            cap.release()

        if out is not None:
            out.release()

        if detector is not None:
            detector.close()

if __name__ == "__main__":
    main()