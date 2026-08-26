import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy.signal import find_peaks

def parse_arg():
    parser = argparse.ArgumentParser(
        description="Find gait events"
    )

    parser.add_argument(
        "--frame",
        type=str,
        required=True,
        help="Path to input csv file"
    )

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to metadata csv file"
    )
    parser.add_argument(
        "--plot",
        type=str,
        required=True,
        help="Path to plot file"
    )
    parser.add_argument(
        "--events",
        type=str,
        required=True,
        help="Path to output csv file"
    )

    parser.add_argument(
        "--cycle",
        type=str,
        required=True,
        help="Path to output cycle and cadence csv file"
    )

    return parser.parse_args()

def detect_peaks(frame_table: pd.DataFrame, side, fps, distance_s=0.55, prominence=30):
    if (side != "left") and ( side != "right"):
        raise ValueError("Side должен быть left или right")

    signal_column = f"{side}_knee_flexion_smooth"
    df = frame_table.copy()

    df["has_value"] = df[signal_column].notna()
    df["group_id"] = df["has_value"].ne(df["has_value"].shift()).cumsum()

    distance_frames = round(distance_s*fps)

    result_parts = []

    result_columns = [
        "side",
        "event_type",
        "frame_index",
        "timestamp_ms",
        "flexion",
        "prominence",
    ]
    
    for _, segment in df[df["has_value"] == True].groupby("group_id"):
        peak_index, properties = find_peaks(segment[signal_column], distance=distance_frames, prominence=prominence)
        peak_rows = segment.iloc[peak_index]

        new_rows = pd.DataFrame(
        {
            "side": [side] * len(peak_rows),
            "event_type": ["max_knee_flexion"] * len(peak_rows),
            "frame_index": peak_rows["frame_index"].to_numpy(),
            "timestamp_ms": peak_rows["timestamp_ms"].to_numpy(),
            "flexion": peak_rows[
                signal_column
            ].to_numpy(),
            "prominence": properties["prominences"],
        }
        )

        result_parts.append(new_rows)

    if not result_parts:
        return pd.DataFrame(columns=result_columns)

    result = pd.concat(
        result_parts,
        ignore_index=True,
    )

    return result

def detect_cadence_peaks(frame_table : pd.DataFrame, fps, distance_s=0.273, prominence=10, edge_margin_ms=300):
    df = frame_table.copy()

    df["has_value"] = df["cadence_signal"].notna()
    df["group_id"] = df["has_value"].ne(df["has_value"].shift()).cumsum()

    distance_frames = round(distance_s*fps)

    result_parts = []
    
    result_columns = [
        "segment_id",
        "event_type",
        "frame_index",
        "timestamp_ms",
        "flexion",
        "prominence",
        "left_flexion",
        "right_flexion",
        "peak_source",
        "segment_start_timestamp_ms",
        "segment_end_timestamp_ms",
        "distance_from_segment_start_ms",
        "distance_from_segment_end_ms",
        "is_edge_peak",
    ]

    for group_id, segment in df[df["has_value"] == True].groupby("group_id"):
        peak_index, properties = find_peaks(segment["cadence_signal"], distance=distance_frames, prominence=prominence)

        if len(peak_index) == 0:
            continue

        peak_rows = segment.iloc[peak_index]

        segment_start_timestamp_ms = (
            segment["timestamp_ms"].iloc[0]
        )

        segment_end_timestamp_ms = (
            segment["timestamp_ms"].iloc[-1]
        )

        peak_timestamps = (
            peak_rows["timestamp_ms"].to_numpy()
        )

        distance_from_start = (
            peak_timestamps
            - segment_start_timestamp_ms
        )

        distance_from_end = (
            segment_end_timestamp_ms
            - peak_timestamps
        )

        is_edge_peak = (
            (distance_from_start < edge_margin_ms)
            | (distance_from_end < edge_margin_ms)
        )

        new_rows = pd.DataFrame({
            "segment_id": [group_id] * len(peak_rows),
            "event_type": ["cadence_peak"] * len(peak_rows),
            "frame_index": peak_rows["frame_index"].to_numpy(),
            "timestamp_ms": peak_rows["timestamp_ms"].to_numpy(),
            "flexion": peak_rows["cadence_signal"].to_numpy(),
            "prominence": properties["prominences"],
             "left_flexion": (
                peak_rows["left_knee_flexion_smooth"].to_numpy()
            ),
            "right_flexion": (
                peak_rows["right_knee_flexion_smooth"].to_numpy()
            ),

            "peak_source" : np.where(
                peak_rows["left_knee_flexion_smooth"].to_numpy() == peak_rows["cadence_signal"].to_numpy(), 
                "left", 
                "right"
            ),
            "segment_start_timestamp_ms" : segment_start_timestamp_ms,
            "segment_end_timestamp_ms": segment_end_timestamp_ms,
            "distance_from_segment_start_ms" : distance_from_start,
            "distance_from_segment_end_ms" : distance_from_end,
            "is_edge_peak": is_edge_peak
        })

        result_parts.append(new_rows)

    if not result_parts:
        return pd.DataFrame(columns=result_columns)

    result = pd.concat(
        result_parts,
        ignore_index=True,
    )

    result = result.sort_values(
        ["segment_id", "timestamp_ms"]
    ).reset_index(drop=True)

    return result

def plot_diagnostic(frame_df : pd.DataFrame, events_df : pd.DataFrame, plot_path: Path):
    plt.figure(figsize=(12, 6))

    plt.plot(
        frame_df["timestamp_ms"],
        frame_df["cadence_signal"],
        linewidth=2,
        label="Общий сигнал каденса",
    )

    plt.scatter(
        events_df["timestamp_ms"],
        events_df["flexion"],
        s=70,
        marker="o",
        label="Пики каденса",
        zorder=3,
    )

    plt.xlabel("Время, мс")
    plt.ylabel("Сгибание колена, градусы")
    plt.title("Обнаруженные максимумы сгибания коленей")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        plot_path,
        dpi=300,
    )

    plt.close()

def find_cycle_cadence(df: pd.DataFrame):
    event = df.sort_values("timestamp_ms").copy()

    result = pd.DataFrame({
        "side" : event["side"],
        "start_frame" : event["frame_index"],
        "end_frame" : event.groupby("side")["frame_index"].shift(-1),
        "start_timestamp_ms" : event["timestamp_ms"],
        "end_timestamp_ms" : event.groupby("side")["timestamp_ms"].shift(-1)
    })

    result["stride_duration_ms"] = (
        result["end_timestamp_ms"] - result["start_timestamp_ms"]
    )

    result["cadence_spm"] = (
        120_000 / result["stride_duration_ms"]
    )

    result["is_valid_cycle"] = True
    result.loc[
        (result["cadence_spm"] <= 120) | (result["cadence_spm"] >= 220),
        "is_valid_cycle"
    ] = False
    
    result = result.dropna(
        subset=["end_frame", "end_timestamp_ms"]
    ).copy()

    result["end_frame"] = result["end_frame"].astype(int)
    result["end_timestamp_ms"] = result["end_timestamp_ms"].astype(int)
    result["stride_duration_ms"] = result["stride_duration_ms"].astype(int)
    result["cadence_spm"] = result["cadence_spm"].round(1)

    result = result.reset_index(drop=True)

    return result


def find_stride_cadence(events_df: pd.DataFrame):
    result_columns = [
        "segment_id",
        "start_frame",
        "middle_frame",
        "end_frame",
        "start_timestamp_ms",
        "middle_timestamp_ms",
        "end_timestamp_ms",
        "stride_duration_ms",
        "start_source",
        "middle_source",
        "end_source",
        "start_is_edge",
        "middle_is_edge",
        "end_is_edge",
        "same_source",
        "alternates_correctly",
        "is_edge_cycle",
        "cadence_spm",
        "is_valid_stride_interval",
    ]

    if events_df.empty:
        return pd.DataFrame(columns=result_columns)

    events = (
        events_df
        .sort_values(["segment_id", "timestamp_ms"])
        .reset_index(drop=True)
        .copy()
    )

    result = pd.DataFrame({
        "segment_id": events["segment_id"],
        "start_frame": events["frame_index"],
        "middle_frame" : (
            events.groupby("segment_id")["frame_index"]
            .shift(-1)
        ),
        "end_frame": (
            events.groupby("segment_id")["frame_index"]
            .shift(-2)
        ),
        "start_timestamp_ms": events["timestamp_ms"],
        "middle_timestamp_ms": (
            events.groupby("segment_id")["timestamp_ms"]
            .shift(-1)
        ),
        "end_timestamp_ms": (
            events.groupby("segment_id")["timestamp_ms"]
            .shift(-2)
        ),
        "start_source": events["peak_source"],
        "middle_source": (
            events.groupby("segment_id")["peak_source"]
            .shift(-1)
        ),
        "end_source": (
            events.groupby("segment_id")["peak_source"]
            .shift(-2)
        ),
        "start_is_edge": events["is_edge_peak"],

        "middle_is_edge": (
            events.groupby("segment_id")["is_edge_peak"]
            .shift(-1)
        ),

        "end_is_edge": (
            events.groupby("segment_id")["is_edge_peak"]
            .shift(-2)
        ),
    })

    result = result.dropna(
        subset=[
            "middle_frame",
            "end_frame",
            "middle_timestamp_ms",
            "end_timestamp_ms",
        ]
    ).copy()

    result["stride_duration_ms"] = (
        result["end_timestamp_ms"]
        - result["start_timestamp_ms"]
    )

    result["cadence_spm"] = (
        120_000 / result["stride_duration_ms"]
    )

    result["same_source"] = (
        result["start_source"]
        == result["end_source"]
    )

    result["alternates_correctly"] = (
        result["same_source"]
        & (
            result["middle_source"]
            != result["start_source"]
        )
    )

    result["is_edge_cycle"] = (
        result["start_is_edge"]
        | result["middle_is_edge"]
        | result["end_is_edge"]
    )

    result["is_valid_stride_interval"] = (
        result["cadence_spm"].between(
            120,
            220,
            inclusive="both",
        )
    )

    result["segment_id"] = result["segment_id"].astype(int)
    result["start_frame"] = result["start_frame"].astype(int)
    result["end_frame"] = result["end_frame"].astype(int)
    result["start_timestamp_ms"] = (
        result["start_timestamp_ms"].astype(int)
    )
    result["end_timestamp_ms"] = (
        result["end_timestamp_ms"].astype(int)
    )
    result["stride_duration_ms"] = (
        result["stride_duration_ms"].astype(int)
    )
    result["cadence_spm"] = result["cadence_spm"].round(1)
    result["middle_frame"] = (
        result["middle_frame"].astype(int)
    )

    result["middle_timestamp_ms"] = (
        result["middle_timestamp_ms"].astype(int)
    )
    edge_columns = [
        "start_is_edge",
        "middle_is_edge",
        "end_is_edge",
    ]

    result[edge_columns] = (
        result[edge_columns].astype(bool)
    )

    return result.reset_index(drop=True)

def stats(result: pd.DataFrame, frame_df: pd.DataFrame):
    valid_cadence = result.loc[
        result["is_high_quality"] == True,
        "cadence_spm",
    ]

    if valid_cadence.empty:
        print("High-quality циклы не найдены")
        return

    median_cadence = valid_cadence.median()

    high_quality_count = len(valid_cadence)

    q1 = valid_cadence.quantile(0.25)
    q3 = valid_cadence.quantile(0.75)
    iqr = q3 - q1

    min_cad = valid_cadence.min()
    max_cad = valid_cadence.max()

    std_cadence = valid_cadence.std()

    mean_cadence = valid_cadence.mean()

    cv_percent = (
        std_cadence / mean_cadence * 100
    )

    valid_trunk_lean = frame_df.loc[
        frame_df["trunk_valid"]
        & frame_df["forward_trunk_lean_deg"].notna(),
        "forward_trunk_lean_deg",
    ]

    if valid_trunk_lean.empty:
        print("Валидные значения наклона туловища не найдены")
        return

    median_trunk_lean = valid_trunk_lean.median()
    quality_trunk_frame = len(valid_trunk_lean)

    q1_trunk_lean = valid_trunk_lean.quantile(0.25)
    q3_trunk_lean = valid_trunk_lean.quantile(0.75)
    iqr_t = q3_trunk_lean - q1_trunk_lean

    min_trunk_lean = valid_trunk_lean.min()
    max_trunk_lean = valid_trunk_lean.max()

    std_trunk_lean = valid_trunk_lean.std()
    mean_trunk_lean = valid_trunk_lean.mean()

    cv_percent_trunk_lean = (
        std_trunk_lean / mean_trunk_lean * 100
    )

    print(f"Каденс: {median_cadence:.1f}")
    print(f"Качественных циклов: {high_quality_count}")
    print(f"Q1(25%): {q1:.1f} spm")
    print(f"Q3(75%): {q3:.1f} spm")
    print(f"Центральные 50%: {q1:.1f}-{q3:.1f} spm")
    print(f"IQR: {iqr:.1f} spm")
    print(f"Диапазон: {min_cad:.1f} - {max_cad:.1f}") 
    print(f"Стандартное отклонение: {std_cadence:.1f} spm")
    print(f"Коэффициент вариации: {cv_percent:.1f}%")
    print("\n")
    print(f"Наклон туловища вперед: {median_trunk_lean:.1f}")
    print(f"Валидных кадров: {quality_trunk_frame}")
    print(f"Q1(25%): {q1_trunk_lean:.1f}")
    print(f"Q3(75%): {q3_trunk_lean:.1f}")
    print(f"Центральные 50%: {q1_trunk_lean:.1f}-{q3_trunk_lean:.1f}")
    print(f"IQR: {iqr_t:.1f}")
    print(f"Диапозон: {min_trunk_lean:.1f} - {max_trunk_lean:.1f}")
    print(f"Стандартное отклонение: {std_trunk_lean:.1f}")
    print(f"Коэффициент вариации: {cv_percent_trunk_lean:.1f}%")

def main():
    args = parse_arg()

    csv_path = Path(args.frame)
    csv_meta_path = Path(args.metadata)
    csv_event_path = Path(args.events)
    plot_path = Path(args.plot)
    csv_cycle_path = Path(args.cycle)

    frame_table = pd.read_csv(str(csv_path))
    df_meta = pd.read_csv(str(csv_meta_path))

    fps = df_meta.loc[0, "fps"]

    result = detect_cadence_peaks(frame_table, fps=fps)

    if result.empty:
        raise ValueError("Пики не найдены")

    print(result.head())

    res_cadence = find_stride_cadence(result)

    if res_cadence.empty:
        raise ValueError("Пики есть, полного цикла нет")

    res_cadence["is_high_quality"] = (
        res_cadence["is_valid_stride_interval"]
        & res_cadence["alternates_correctly"]
        & ~res_cadence["is_edge_cycle"]
    )
    
    print(res_cadence.head())
    res_cadence.to_csv(str(csv_cycle_path), index=False)
    stats(res_cadence, frame_table)

    plot_diagnostic(frame_table, result, plot_path)

    result.to_csv(str(csv_event_path), index=False)

if __name__ == "__main__":
    main()

