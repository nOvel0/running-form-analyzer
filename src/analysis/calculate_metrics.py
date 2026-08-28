import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

METRIC_LANDMARKS = [
    "LEFT_HIP",
    "LEFT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_HIP",
    "RIGHT_KNEE",
    "RIGHT_ANKLE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
]

def parse_arg():
    parser = argparse.ArgumentParser(
        description="Calculate metrics running video"
    )

    parser.add_argument(
        "--keypoints",
        type=str,
        required=True,
        help="Path to input csv file"
    )

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadate csv file"
    )

    parser.add_argument(
        "--frame_table",
        type=str,
        required=True,
        help="Path to output frame table"
    )
    parser.add_argument(
        "--plot",
        type=str,
        required=True,
        help="Dir plot graphics"
    )

    return parser.parse_args()

def prepare_frame_level_table(df: pd.DataFrame) -> pd.DataFrame:
    frames = (
        df[
            [
                "frame_index",
                "timestamp_ms",
                "pose_detected",
                "is_valid_frame"
            ]
        ]
        .drop_duplicates("frame_index")
        .copy()
    )
    metrics_df = df[
        df["landmark_name"].isin(METRIC_LANDMARKS)
    ].copy()

    points = metrics_df.pivot(
        index="frame_index",
        columns="landmark_name",
        values=["x_px", "y_px", "is_valid_landmark"],
    )

    points.columns = [
        f"{landmark}_{value}"
        for value, landmark in points.columns
    ]

    frame_df = frames.merge(
        points,
        on="frame_index",
        how="left",
    )

    valid_columns = [
        f"{landmark}_is_valid_landmark"
        for landmark in METRIC_LANDMARKS
    ]

    frame_df[valid_columns] = (
        frame_df[valid_columns]
        .fillna(0)
        .astype(bool)
    )

    frame_df["is_valid_frame"] = (
        frame_df["is_valid_frame"]
        .fillna(0)
        .astype(bool)
    )

    frame_df["left_knee_valid"] = (
        frame_df["is_valid_frame"]
        & frame_df["LEFT_HIP_is_valid_landmark"]
        & frame_df["LEFT_KNEE_is_valid_landmark"]
        & frame_df["LEFT_ANKLE_is_valid_landmark"]
    )
    frame_df["right_knee_valid"] = (
        frame_df["is_valid_frame"]
        & frame_df["RIGHT_HIP_is_valid_landmark"]
        & frame_df["RIGHT_KNEE_is_valid_landmark"]
        & frame_df["RIGHT_ANKLE_is_valid_landmark"]
    )

    return frame_df.sort_values("frame_index").reset_index(drop=True)


def calculate_angle(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)

    v1 = A - B
    v2 = C - B

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if (
        not np.isfinite(v1).all()
        or not np.isfinite(v2).all()
        or norm_v1 == 0
        or norm_v2 == 0
    ):
        return np.nan

    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)

    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return float(angle_deg)


def internal_angle_to_flexion(angle: float) -> float:
    if not np.isfinite(angle):
        return np.nan

    return 180.0 - angle


def new_column_angle(frame_df: pd.DataFrame) -> pd.DataFrame:
    frame_df["left_knee_internal_angle"] = frame_df.apply(
    lambda row: calculate_angle(
        [row["LEFT_HIP_x_px"], row["LEFT_HIP_y_px"]],
        [row["LEFT_KNEE_x_px"], row["LEFT_KNEE_y_px"]],
        [row["LEFT_ANKLE_x_px"], row["LEFT_ANKLE_y_px"]],
    ),
    axis=1,
    )

    frame_df["right_knee_internal_angle"] = frame_df.apply(
    lambda row: calculate_angle(
        [row["RIGHT_HIP_x_px"], row["RIGHT_HIP_y_px"]],
        [row["RIGHT_KNEE_x_px"], row["RIGHT_KNEE_y_px"]],
        [row["RIGHT_ANKLE_x_px"], row["RIGHT_ANKLE_y_px"]],
    ),
    axis=1,
    )

    frame_df.loc[frame_df["left_knee_valid"] == 0, "left_knee_internal_angle"] = np.nan
    frame_df.loc[frame_df["right_knee_valid"] == 0, "right_knee_internal_angle"] = np.nan

    frame_df["left_knee_flexion_raw"] = (
        frame_df["left_knee_internal_angle"]
        .apply(internal_angle_to_flexion)
    )

    frame_df["right_knee_flexion_raw"] = (
        frame_df["right_knee_internal_angle"]
        .apply(internal_angle_to_flexion)
    )

    return frame_df

def plot_hip(frame_df: pd.DataFrame, plot_path: Path):
    plt.figure(figsize=(12, 6))

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["left_hip_flexion_raw"],
        linewidth=1,
        label="Левое бедро raw",
    )

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["right_hip_flexion_raw"],
        linewidth=1,
        label="Правое бедро raw",
    )

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["left_hip_flexion_despiked"],
        linewidth=1,
        label="Левое бедро без шумов",
    )

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["right_hip_flexion_despiked"],
        linewidth=1,
        label="Правое бедро без шумов",
    )

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["left_hip_flexion_smooth"],
        linewidth=1,
        label="Левое бедро smooth",
    )

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["right_hip_flexion_smooth"],
        linewidth=1,
        label="Правое бедро smooth",
    )

    plt.xlabel("Время, мс")
    plt.ylabel("Угол бедра, градусы")

    plt.title("Угол бедер относительно туловища")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_path / "hip.png", dpi=300)

    plt.close()


def plot_trunk_lean(frame_df: pd.DataFrame, plot_path: Path):
    plt.figure(figsize=(12, 6))

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["trunk_lean_image_deg"],
        linewidth=1,
        label="Наклон туловища, градусы",
    )

    plt.xlabel("Время, мс")
    plt.ylabel("Наклон туловища, градусы")

    plt.title("Наклон туловища относительно вертикали")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_path / "trunk_lean.png", dpi=300)

    plt.close()

def plot_angle(frame_df: pd.DataFrame, plot_path: Path):
    plt.figure(figsize=(12, 6))

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["left_knee_flexion_raw"],
        linewidth=1,
        alpha=0.60,
        label="Левое колено — raw",
    )

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["right_knee_flexion_raw"],
        linewidth=1,
        alpha=0.60,
        label="Правое колено — raw",
    )

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["left_knee_flexion_smooth"],
        linewidth=2,
        label="Левое колено — smooth",
    )

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["right_knee_flexion_smooth"],
        linewidth=2,
        label="Правое колено - smooth",
    )

    mask_left = frame_df["interpolated_mask_left"]
    mask_right = frame_df["interpolated_mask_right"]

    plt.scatter(
        frame_df.loc[mask_left, "timestamp_ms"],
        frame_df.loc[mask_left, "left_knee_flexion_interpolated"],
        s=25,
        marker="o",
        label="Интерполированные точки left",
    )
    plt.scatter(
        frame_df.loc[mask_right, "timestamp_ms"],
        frame_df.loc[mask_right, "right_knee_flexion_interpolated"],
        s=25,
        marker="o",
        label="Интерполированные точки right",
    )

    plt.xlabel("Время, мс")
    plt.ylabel("Угол колена, градусы")
    plt.title("Изменение угла колена с течением времени")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(plot_path / "angle.png", dpi=300)

    plt.close()

def interpolate_short_gaps(series: pd.Series, fps, max_gap_ms=50):
    df = pd.DataFrame({
        "value": series
    })

    df["is_nan"] = df["value"].isna()
    df["group_id"] = df["is_nan"].ne(df["is_nan"].shift()).cumsum()
    df["interpolated_mask"] = False

    max_gap_frames = round(fps * max_gap_ms / 1000)

    for group_id, gap in df[df["is_nan"]].groupby("group_id"):
        gap_len = len(gap)

        if gap_len > max_gap_frames:
            continue

        start_pos = df.index.get_loc(gap.index[0])
        end_pos = df.index.get_loc(gap.index[-1])

        if start_pos == 0 or end_pos == len(df) - 1:
            continue

        left_value = df.iloc[start_pos - 1]["value"]
        right_value = df.iloc[end_pos + 1]["value"]

        if pd.isna(left_value) or pd.isna(right_value):
            continue

        values = np.linspace(
            left_value,
            right_value,
            gap_len + 2
        )[1:-1]

        value_col = df.columns.get_loc("value")
        mask_col = df.columns.get_loc("interpolated_mask")

        df.iloc[start_pos:end_pos + 1, value_col] = values
        df.iloc[start_pos:end_pos + 1, mask_col] = True

    return df["value"], df["interpolated_mask"]


def smooth_continuous_segments(series: pd.Series, fps, window_ms=120, poly=2):
    df = pd.DataFrame({
        "value": series
    })
    df["is_notna"] = df["value"].notna()
    df["group_id"] = df["is_notna"].ne(df["is_notna"].shift()).cumsum()

    window_frame = round(fps*window_ms/1000)
    if window_frame % 2 == 0:
        window_frame += 1

    if window_frame <= poly:
        raise ValueError("window_frame меньше чем poly")

    for group_id, gap in df[df["is_notna"] == True].groupby("group_id"):
        if len(gap) >= window_frame:
            start_pos = df.index.get_loc(gap.index[0])
            end_pos = df.index.get_loc(gap.index[-1])
            values_slice = df['value'].iloc[start_pos : end_pos + 1]

            smoothed_signal = savgol_filter(values_slice, window_frame, poly)

            value_col = df.columns.get_loc("value")
                    
            df.iloc[start_pos:end_pos + 1, value_col] = smoothed_signal  

    return df["value"]

def smoothing_hip(series: pd.Series, window_median_filter=3):
    df = pd.DataFrame({
        "value": series
    })
    df["is_notna"] = df["value"].notna()
    df["group_id"] = df["is_notna"].ne(df["is_notna"].shift()).cumsum()

    for _, gap in df[df["is_notna"] == True].groupby("group_id"):
        start_pos = df.index.get_loc(gap.index[0])
        end_pos = df.index.get_loc(gap.index[-1])
        values_slice = df["value"].iloc[start_pos : end_pos + 1]

        filtered_signal = median_filter(values_slice, size=window_median_filter, mode="nearest")

        value_col = df.columns.get_loc("value")

        df.iloc[start_pos:end_pos + 1, value_col] = filtered_signal

    return df["value"]

def calculate_trunk_lean(frame_df: pd.DataFrame, direction_threshold_deg=0.5) -> pd.DataFrame:
    trunk_valid_columns = [
        "LEFT_SHOULDER_is_valid_landmark",
        "RIGHT_SHOULDER_is_valid_landmark",
        "LEFT_HIP_is_valid_landmark",
        "RIGHT_HIP_is_valid_landmark",
    ]

    frame_df["trunk_valid"] = (
        frame_df["is_valid_frame"]
        & frame_df[trunk_valid_columns].all(axis=1)
    )

    frame_df["shoulder_center_x_px"] = (
        frame_df["LEFT_SHOULDER_x_px"]
        + frame_df["RIGHT_SHOULDER_x_px"]
    ) / 2

    frame_df["shoulder_center_y_px"] = (
        frame_df["LEFT_SHOULDER_y_px"]
        + frame_df["RIGHT_SHOULDER_y_px"]
    ) / 2

    frame_df["hip_center_x_px"] = (
        frame_df["LEFT_HIP_x_px"]
        + frame_df["RIGHT_HIP_x_px"]
    ) / 2

    frame_df["hip_center_y_px"] = (
        frame_df["LEFT_HIP_y_px"]
        + frame_df["RIGHT_HIP_y_px"]
    ) / 2


    dx = (
        frame_df["shoulder_center_x_px"]
        - frame_df["hip_center_x_px"]
    )

    vertical_height = (
        frame_df["hip_center_y_px"]
        - frame_df["shoulder_center_y_px"]
    )

    frame_df["trunk_lean_image_deg"] = np.degrees(np.arctan2(dx, vertical_height))

    frame_df.loc[
            ~frame_df["trunk_valid"],
            "trunk_lean_image_deg",
        ] = np.nan
    

    angle = frame_df["trunk_lean_image_deg"]
    frame_df["trunk_lean_magnitude_deg"] = angle.abs()

    frame_df["trunk_lean_image_direction"] = np.select(
        [
            angle > direction_threshold_deg,
            angle < -direction_threshold_deg,
        ],
        [
            "right",
            "left",
        ],
        default="vertical"
    )
    frame_df.loc[
        ~frame_df["trunk_valid"],
        "trunk_lean_image_direction",
    ] = "unknown"

    valid_hip_x = frame_df.loc[
        frame_df["trunk_valid"],
        "hip_center_x_px",
    ]

    start_x = valid_hip_x.head(10).median()
    end_x = valid_hip_x.tail(10).median()

    running_direction_sign = np.sign(end_x - start_x)

    frame_df["forward_trunk_lean_deg"] = (
        frame_df["trunk_lean_image_deg"]
        * running_direction_sign
    )

    frame_df["running_direction"] = running_direction_sign
    
    return frame_df

def calculate_hip_flexion(frame_df: pd.DataFrame) -> pd.DataFrame:
    frame_df["left_hip_flexion_valid"] = (
        frame_df["trunk_valid"]
        & frame_df["LEFT_KNEE_is_valid_landmark"]
    )

    frame_df["right_hip_flexion_valid"] = (
        frame_df["trunk_valid"]
        & frame_df["RIGHT_KNEE_is_valid_landmark"]
    )

    trunk_dx = (
        frame_df["hip_center_x_px"]
        - frame_df["shoulder_center_x_px"]
    )

    trunk_dy = (
        frame_df["hip_center_y_px"]
        - frame_df["shoulder_center_y_px"]
    )

    left_thigh_dx = (
        frame_df["LEFT_KNEE_x_px"]
        - frame_df["LEFT_HIP_x_px"]
    )
    right_thigh_dx = (
        frame_df["RIGHT_KNEE_x_px"]
        - frame_df["RIGHT_HIP_x_px"]
    )

    left_thigh_dy = (
        frame_df["LEFT_KNEE_y_px"]
        - frame_df["LEFT_HIP_y_px"]
    )
    right_thigh_dy = (
        frame_df["RIGHT_KNEE_y_px"]
        - frame_df["RIGHT_HIP_y_px"]
    )

    left_dot = (
        trunk_dx * left_thigh_dx
        + trunk_dy * left_thigh_dy
    )
    right_dot = (
        trunk_dx * right_thigh_dx
        + trunk_dy * right_thigh_dy
    )

    left_cross = (
        trunk_dx * left_thigh_dy
        - trunk_dy * left_thigh_dx
    )
    right_cross = (
        trunk_dx * right_thigh_dy
        - trunk_dy * right_thigh_dx
    )

    left_angle_image = np.degrees(
        np.arctan2(left_cross, left_dot)
    )
    frame_df["left_hip_flexion_raw"] = (
        -frame_df["running_direction"]
        * left_angle_image
    )
    frame_df.loc[
        ~frame_df["left_hip_flexion_valid"],
        "left_hip_flexion_raw",
    ] = np.nan

    right_angle_image = np.degrees(
        np.arctan2(right_cross, right_dot)
    )
    frame_df["right_hip_flexion_raw"] = (
        -frame_df["running_direction"]
        * right_angle_image
    )
    frame_df.loc[
        ~frame_df["right_hip_flexion_valid"],
        "right_hip_flexion_raw",
    ] = np.nan

    return frame_df
    
def main():
    args = parse_arg()

    csv_path = Path(args.keypoints)
    csv_meta_path = Path(args.metadata)
    csv_frame_table_path = Path(args.frame_table)
    plot_path = Path(args.plot)
    plot_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(str(csv_path))
    df_meta = pd.read_csv(str(csv_meta_path))

    fps = df_meta.loc[0, "fps"]
    width = df_meta.loc[0, "width"]
    height = df_meta.loc[0, "height"]

    #Добавление x_px и y_px
    df["x_px"] = df["x"]*width
    df["y_px"] = df["y"]*height

    frame_df = prepare_frame_level_table(df)
    frame_df = new_column_angle(frame_df)
    frame_df = calculate_trunk_lean(frame_df)
    frame_df = calculate_hip_flexion(frame_df)

    # Заполняем короткие промежутки 
    values_right, mask_right = interpolate_short_gaps(frame_df["right_knee_flexion_raw"], fps)
    values_left, mask_left = interpolate_short_gaps(frame_df["left_knee_flexion_raw"], fps)

    frame_df["left_knee_flexion_interpolated"] = values_left
    frame_df["right_knee_flexion_interpolated"] = values_right

    frame_df["interpolated_mask_left"] = mask_left
    frame_df["interpolated_mask_right"] = mask_right

    frame_df["left_knee_flexion_smooth"] = smooth_continuous_segments(
        frame_df["left_knee_flexion_interpolated"],
        fps=fps,
    )
    frame_df["right_knee_flexion_smooth"] = smooth_continuous_segments(
        frame_df["right_knee_flexion_interpolated"],
        fps=fps
    )

    frame_df["left_hip_flexion_despiked"] = smoothing_hip(
        frame_df["left_hip_flexion_raw"]
    )
    frame_df["right_hip_flexion_despiked"] = smoothing_hip(
        frame_df["right_hip_flexion_raw"]
    )

    frame_df["left_hip_flexion_smooth"] = smooth_continuous_segments(
        frame_df["left_hip_flexion_despiked"],
        fps,
        window_ms=80,
        poly=2
    )
    frame_df["right_hip_flexion_smooth"] = smooth_continuous_segments(
        frame_df["right_hip_flexion_despiked"],
        fps,
        window_ms=80,
        poly=2
    )

    frame_df["cadence_signal"] = frame_df[
        ["left_knee_flexion_smooth", "right_knee_flexion_smooth"]
    ].max(axis=1)
    
    plot_angle(frame_df=frame_df, plot_path=plot_path)
    plot_trunk_lean(frame_df=frame_df, plot_path=plot_path)
    plot_hip(frame_df=frame_df, plot_path=plot_path)

    frame_df.to_csv(str(csv_frame_table_path), index=False)



if __name__ == "__main__":
    main()