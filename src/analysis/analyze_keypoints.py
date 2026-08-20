import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

LEGS_LIST = ["LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX", "LEFT_HIP", "RIGHT_HIP"]


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Analyze running metrics csv"
    )

    parser.add_argument(
        "--keypoints",
        type=str,
        required=True,
        help="Path to input csv file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Dir for output reports"
    )

    return parser.parse_args()


def calculate_detection_stats(df):
    total_frames = df["frame_index"].nunique(dropna=False)

    detected_frames = df.loc[df["pose_detected"] == 1, "frame_index"].nunique(dropna=False)

    missing_frames = total_frames - detected_frames

    detected_rate = detected_frames / total_frames

    return {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "missing_frames": missing_frames,
        "detected_rate": detected_rate,
    }
    

def mean_visibility(df):
    valid_poses = df[df["pose_detected"] == 1].dropna(subset=["landmark_name"])
    
    avg_visibility = valid_poses.groupby("landmark_name")["visibility"].mean()

    return avg_visibility


def plot_time(df, landmarks_name, y, title, file_name, inverse_bool=False, use_valid_data=True):
    if use_valid_data:
        df = df.copy()

        df.loc[
            (df["is_valid_frame"] == 0) |
            (df["is_valid_landmark"] == 0),
            y
        ] = np.nan
    else:
        df = df[df["pose_detected"] == 1]


    if len(landmarks_name) != 2:
        raise ValueError("В landmarks_name_y должно быть два имени")
    
    one = df[df["landmark_name"] == landmarks_name[0]].sort_values("timestamp_ms")
    two = df[df["landmark_name"] == landmarks_name[1]].sort_values("timestamp_ms")

    plt.figure(figsize=(14, 6))

    plt.plot(one["timestamp_ms"], one[y], label=landmarks_name[0], color="blue", alpha=0.8)
    plt.plot(two["timestamp_ms"], two[y], label=landmarks_name[1], color="red", alpha=0.8)

    if inverse_bool:
        plt.gca().invert_yaxis()

    plt.xlabel("Время (timestamp_ms)")
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.savefig(file_name, dpi=300)

    plt.close()

def main():
    args = parse_arg()

    csv_path = Path(args.keypoints)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(str(csv_path))
    
    stats = calculate_detection_stats(df)
    for name, value in stats.items():
        print(f"{name}: {value}")


    visibility = mean_visibility(df)

    print(visibility.reindex(LEGS_LIST).sort_values(ascending=False))

    plot_time(df, 
        ["LEFT_ANKLE", "RIGHT_ANKLE"],
        "y", 
        "Вертикальная координата голеностопов", 
        out_dir / "plot_ankle.png",
        True
    )
    plot_time(
        df,
        ["LEFT_KNEE", "RIGHT_KNEE"],
        "y",
        "Вертикальная координата колена",
        out_dir / "plot_knee.png",
        True
    )
    plot_time(
        df,
        ["LEFT_ANKLE", "RIGHT_ANKLE"],
        "visibility",
        "visibility лодыжки",
        out_dir / "plot_visibility_ankle.png",
        use_valid_data=False
    )
    plot_time(
        df,
        ["LEFT_KNEE", "RIGHT_KNEE"],
        "visibility",
        "visibility колена",
        out_dir / "plot_visibility_knee.png",
        use_valid_data=False
    )
    plot_time(
        df,
        ["LEFT_HIP", "RIGHT_HIP"],
        "visibility",
        "visibility бедра",
        out_dir / "plot_visibility_hip.png",
        use_valid_data=False
    )


if __name__ == "__main__":
    main()
