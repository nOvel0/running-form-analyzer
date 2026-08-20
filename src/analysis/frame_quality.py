import argparse
import pandas as pd
from pathlib import Path


EXPECTED_LANDMARKS_COUNT = 33
LEGS_LIST_LEFT = ["LEFT_KNEE", "LEFT_ANKLE", "LEFT_HEEL", "LEFT_FOOT_INDEX", "LEFT_HIP"]
LEGS_LIST_RIGHT = ["RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_HEEL", "RIGHT_FOOT_INDEX", "RIGHT_HIP"]
MIN_MEDIAN_ABS_HIP_DX = 0.003
MIN_HIP_X_DISPLACEMENT = 0.05
MIN_HIP_X_RANGE = 0.1

def parse_arg():
    parser = argparse.ArgumentParser(
        description="Analyze frame"
    )

    parser.add_argument(
        "--keypoints",
        type=str,
        required=True,
        help="Path to input csv file"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output csv new file"
    )

    return parser.parse_args()


def check_csv_integrity(df):
    if df.empty:
        raise ValueError("CSV пустой")
    
    required_columns = {
        "frame_index",
        "timestamp_ms",
        "pose_detected",
        "landmark_name",
        "x",
        "y",
        "z",
        "visibility",
        "presence",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"В CSV отсутствуют столбцы: {sorted(missing_columns)}"
        )
    
    detected_rows = df[df["pose_detected"] == 1]

    landmarks_per_frame = (
        detected_rows
        .groupby("frame_index")["landmark_name"]
        .nunique()
    )

    invalid_frames = landmarks_per_frame[
        landmarks_per_frame != EXPECTED_LANDMARKS_COUNT
    ]

    if not invalid_frames.empty:
        print(
            "Предупреждение: найдены кадры с количеством точек не равным 33:"
        )
        print(invalid_frames.head(10))
        raise ValueError(
            f"В CSV найдены кадры с количеством точек не равным 33"
        )
        

    duplicates = detected_rows.duplicated(
        subset=["frame_index", "landmark_name"]
    ).sum()

    if duplicates > 0:
        print(
            f"Предупреждение: найдено дубликатов точек: {duplicates}"
        )
        raise ValueError(
            f"В CSV найдены дубликаты"
        )

    frame_info = (
        df[
            [
                "frame_index",
                "timestamp_ms",
                "pose_detected",
            ]
        ]
        .drop_duplicates()
        .sort_values("frame_index")
    )

    if not frame_info["timestamp_ms"].is_monotonic_increasing:
        print("Предупреждение: timestamp_ms идут не по возрастанию")
        raise ValueError("timestamp_ms идут не по возрастанию")

    print("Проверка структуры CSV завершена")


def detect_valid_landmarks(df, visibility_threshold=0.3, presence_threshold=0.5):
    df["is_valid_landmark"] = 1
    df.loc[df["pose_detected"] == 0, "is_valid_landmark"] = 0

    df.loc[df["visibility"] <= visibility_threshold, "is_valid_landmark"] = 0
    df.loc[df["presence"] <= presence_threshold, "is_valid_landmark"] = 0

    # точки внутри изображения
    df.loc[df["x"] < 0, "is_valid_landmark"] = 0
    df.loc[df["x"] > 1, "is_valid_landmark"] = 0
    df.loc[df["y"] < 0, "is_valid_landmark"] = 0
    df.loc[df["y"] > 1, "is_valid_landmark"] = 0
    

def detect_valid_frame(df, x_min=0.05, x_max=0.95):
    df["is_valid_frame"] = 1
    df.loc[df["pose_detected"] == 0, "is_valid_frame"] = 0

    hip_rows = df[df["landmark_name"].isin(["LEFT_HIP", "RIGHT_HIP"])]
    
    if not (df["pose_detected"] == 1).any():
        raise ValueError("Нет найденых поз")

    # Проверяем валидность обоих бедер для каждого кадра
    hip_valid = hip_rows.pivot(
        index="frame_index",
        columns="landmark_name",
        values="is_valid_landmark"
    )

    valid_hip_frames = hip_valid[
        (hip_valid["LEFT_HIP"] == 1) &
        (hip_valid["RIGHT_HIP"] == 1)
    ].index

    hip_rows = hip_rows[
        hip_rows["frame_index"].isin(valid_hip_frames)
    ]

    hips = hip_rows.pivot(
        index="frame_index",
        columns="landmark_name",
        values="x"
    )

    hips["hip_center_x"] = (
        hips["LEFT_HIP"] + hips["RIGHT_HIP"] 
    ) / 2

    hips["inside_work_zone"] = hips["hip_center_x"].between(
        x_min,
        x_max
    )

    return hips[["hip_center_x", "inside_work_zone"]]

def detect_runner(hips_df: pd.DataFrame):
    frame_df = (
        hips_df[
            [
                "frame_index",
                "timestamp_ms",
                "pose_detected",
                "hip_center_x",
            ]
        ]
        .drop_duplicates(subset="frame_index")
        .sort_values("frame_index")
        .copy()
    )

    frame_df["pose_segment_id"] = frame_df["pose_detected"].ne(frame_df["pose_detected"].shift()).cumsum()
    frame_df["is_runner_segment"] = True

    frame_df.loc[frame_df["pose_detected"] == 0, "is_runner_segment"] = False

    result_parts = []

    for segment_id, segment in frame_df.groupby("pose_segment_id"):
        start_idx = segment["frame_index"].idxmin()
        end_idx = segment["frame_index"].idxmax()

        hip_x_start = segment.loc[start_idx, "hip_center_x"]
        hip_x_end = segment.loc[end_idx, "hip_center_x"]

        new_cols = pd.DataFrame(
        [{
            "pose_segment_id" : segment_id,
            "start_frame" : segment["frame_index"].min(),
            "end_frame" : segment["frame_index"].max(),
            "frame_count" : segment["frame_index"].nunique(),
            "hip_x_start" : hip_x_start,
            "hip_x_end" : hip_x_end,
            "hip_x_displacement" : abs(hip_x_end - hip_x_start),
            "hip_x_range" : segment["hip_center_x"].max() - segment["hip_center_x"].min(),
            "median_abs_hip_dx" : segment["hip_center_x"].diff().abs().median(),
            "is_runner_segment": segment["is_runner_segment"].iloc[0]
        }]    
        )

        result_parts.append(new_cols)

    s_df = pd.concat(
        result_parts,
        ignore_index=True
    )

    s_df.loc[
        (s_df["hip_x_displacement"] <= MIN_HIP_X_DISPLACEMENT) | 
        (s_df["hip_x_range"] <= MIN_HIP_X_RANGE),
        "is_runner_segment"
    ] = False

    frame_df["is_runner_segment"] = frame_df["pose_segment_id"].map(
        s_df.set_index("pose_segment_id")["is_runner_segment"]
    )

    return hips_df["frame_index"].map(
        frame_df.set_index("frame_index")["is_runner_segment"]
    )


def stats(df):
    total_frames = df['frame_index'].nunique(dropna=False)
    detected_pose = df.loc[df['pose_detected'] == 1, 'frame_index'].nunique(dropna=False)
    valid_frames = df.loc[df['is_valid_frame'] == 1, 'frame_index'].nunique(dropna=False)
    valid_frames_ids = df.loc[df['is_valid_frame'] == 1, 'frame_index'].unique()
    
    df_right = df[
        df["frame_index"].isin(valid_frames_ids) &
        df["landmark_name"].isin(LEGS_LIST_RIGHT)
    ]
    valid_frames_right = (
        df_right.groupby("frame_index")
        .filter(lambda g: len(g) == len(LEGS_LIST_RIGHT) and g["is_valid_landmark"].eq(1).all())
        ["frame_index"]
        .nunique()
    )

    df_left = df[
        df["frame_index"].isin(valid_frames_ids) &
        df["landmark_name"].isin(LEGS_LIST_LEFT)
    ]
    valid_frames_left = (
        df_left.groupby("frame_index")
        .filter(lambda g: len(g) == len(LEGS_LIST_LEFT) and g["is_valid_landmark"].eq(1).all())
        ["frame_index"]
        .nunique()
    )
    total_landmarks = (df["pose_detected"] == 1).sum()
    valid_count = (df["is_valid_landmark"] == 1).sum()
    not_valid_count = total_landmarks - valid_count
    proportion_valid = valid_count / total_landmarks

    print(f"Всего кадров: {total_frames}")
    print(f"Кадров с найденной позой: {detected_pose}")
    print(f"Кадров без найденной позы: {total_frames-detected_pose}")
    print(f"Кадров, валидных для анализа: {valid_frames}")
    print(f"Кадров с валидной правой ногой: {valid_frames_right}")
    print(f"Кадров с валидной левой ногой: {valid_frames_left}")
    print(f"Всего точек: {total_landmarks}")
    print(f"Количество валидных точке: {valid_count}")
    print(f"Количество не валидных точке: {not_valid_count}")
    print(f"Доля валидных точек: {proportion_valid}")


def main():
    args = parse_arg()

    csv_path = Path(args.keypoints)
    csv_path_out = Path(args.output)

    df = pd.read_csv(str(csv_path))
    check_csv_integrity(df)

    detect_valid_landmarks(df=df)

    frame_zone = detect_valid_frame(df=df, x_min=0.1, x_max=0.9)

    df = df.merge(
        frame_zone,
        on="frame_index",
        how="left"
    )

    df["inside_work_zone"] = df["inside_work_zone"].fillna(False)
    df.loc[df["inside_work_zone"] == False, "is_valid_frame"] = 0

    df["is_runner_segment"] = detect_runner(df)

    df.loc[df["is_runner_segment"] == False, "is_valid_frame"] = 0
    stats(df)

    df.to_csv(str(csv_path_out), index=False)


if __name__ == "__main__":
    main()