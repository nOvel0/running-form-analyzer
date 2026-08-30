import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy.signal import find_peaks

IC_TIME_OFFSET_MS = 35
TOE_OFF_COLUMNS = [
    "side",
    "event_type",
    "frame_index",
    "timestamp_ms",
    "ic_frame_index",
    "next_ic_frame_index",
    "ic_timestamp_ms",
    "next_ic_timestamp_ms",
    "stride_duration_ms",
    "contact_time_ms",
    "stance_phase_percent",
    "foot_speed_norm_s",
    "speed_threshold",
    "max_foot_position_jump",
    "is_tracking_stable",
    "quality_reason"
]

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
    parser.add_argument(
        "--contact_foot",
        type=str,
        required=True,
        help="Path to output contact foot csv file"
    )
    parser.add_argument(
        "--toe_off",
        type=str,
        required=True
    )

    return parser.parse_args()

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

def detect_initial_contacts(
        frame_table: pd.DataFrame,
        side: str,
        fps: float,
        cadence_spm: float,
        prominence: float = 0.2,
) -> pd.DataFrame:
    if side not in ("left", "right"):
        raise ValueError("side должен быть left или right")

    signal_column = f"{side}_foot_ahead_norm"

    expected_stride_frames = (
        fps * 120 / cadence_spm
    )
    min_distance_frames = round(
        expected_stride_frames * 0.6
    )

    df = frame_table.copy()

    df["has_value"] = df[signal_column].notna()
    df["group_id"] = (
        df["has_value"]
        .ne(df["has_value"].shift())
        .cumsum()
    )

    result_parts = []

    for _, segment in df[df["has_value"]].groupby("group_id"):
        peak_positions, properties = find_peaks(
            segment[signal_column].to_numpy(),
            distance=min_distance_frames,
            prominence=prominence,
        )

        peak_rows = segment.iloc[peak_positions]

        offset_frames = round(
            fps * IC_TIME_OFFSET_MS / 1000
        )

        contact_positions = np.minimum(
            peak_positions + offset_frames,
            len(segment) - 1,
        )

        contact_rows = segment.iloc[contact_positions]

        result_parts.append(
            pd.DataFrame({
                "side": side,
                "event_type": "initial_contact_candidate",
                "peak_frame_index": (
                    peak_rows["frame_index"].to_numpy()
                ),
                "peak_timestamp_ms": (
                    peak_rows["timestamp_ms"].to_numpy()
                ),
                "frame_index": (
                    contact_rows["frame_index"].to_numpy()
                ),
                "timestamp_ms": (
                    contact_rows["timestamp_ms"].to_numpy()
                ),
                "foot_ahead_norm": (
                    peak_rows[signal_column].to_numpy()
                ),
                "prominence": properties["prominences"],
                "ic_offset_frames": offset_frames,
            })
        )

    if not result_parts:
        return pd.DataFrame(columns=[
            "side",
            "event_type",
            "frame_index",
            "timestamp_ms",
            "foot_ahead_norm",
            "prominence",
        ])

    return pd.concat(result_parts, ignore_index=True)

def build_initial_contact_candidates(res_cadence: pd.DataFrame, frame_df: pd.DataFrame, fps):
    cadence_spm = (
        res_cadence.loc[
            res_cadence["is_high_quality"],
            "cadence_spm",
        ]
        .median()
    )

    if pd.isna(cadence_spm):
        raise ValueError(
            "Невозможно определить IC: нет качественного значения каденса"
        )

    left_contacts = detect_initial_contacts(
        frame_df,
        side="left",
        fps=fps,
        cadence_spm=cadence_spm,
    )

    right_contacts = detect_initial_contacts(
        frame_df,
        side="right",
        fps=fps,
        cadence_spm=cadence_spm,
    )

    contact_candidates = (
        pd.concat(
            [left_contacts, right_contacts],
            ignore_index=True,
        )
        .sort_values("frame_index")
        .reset_index(drop=True)
    )

    return contact_candidates

def detect_toe_off(
        frame_df: pd.DataFrame,
        contact_cycles: pd.DataFrame,
        search_start_ratio=0.3,
        search_end_ratio=0.6,
        threshold_ratio=0.2,
        sustain_frames=3
    ) -> pd.DataFrame:

    result_rows = []

    valid_cycles = contact_cycles.loc[
        contact_cycles["is_valid_cycle"]
    ].copy()

    for _, cycle in valid_cycles.iterrows():
        side = cycle["side"]

        speed_column = f"{side}_foot_speed_norm_s"

        start_timestamp_ms = cycle["start_timestamp_ms"]
        end_timestamp_ms = cycle["end_timestamp_ms"]

        stride_duration_ms = (
            end_timestamp_ms
            - start_timestamp_ms
        )

        cycle_frames = frame_df.loc[
            frame_df["timestamp_ms"].between(
                start_timestamp_ms,
                end_timestamp_ms,
                inclusive="both",
            )
        ].copy()

        cycle_frames["foot_speed_smooth"] = (
            cycle_frames[speed_column]
            .rolling(
                window=3,
                center=True,
                min_periods=1,
            )
            .median()
        )

        speed_signal = (
            cycle_frames["foot_speed_smooth"]
            .dropna()
        )

        if len(speed_signal) < sustain_frames:
            continue

        low_speed = speed_signal.quantile(0.10)
        high_speed = speed_signal.quantile(0.90)

        speed_range = high_speed - low_speed

        if speed_range <= 0:
            continue

        threshold = (
            low_speed
            + threshold_ratio * speed_range
        )

        search_start_ms = (
            start_timestamp_ms
            + search_start_ratio * stride_duration_ms
        )

        search_end_ms = (
            start_timestamp_ms
            + search_end_ratio * stride_duration_ms
        )

        search_frames = cycle_frames.loc[
            cycle_frames["timestamp_ms"].between(
                search_start_ms,
                search_end_ms,
                inclusive="both",
            )
        ].copy()

        position_column = f"{side}_foot_ahead_norm"

        max_foot_position_jump = (
            search_frames[position_column]
            .diff()
            .abs()
            .max()
        )

        is_tracking_stable = (
            pd.notna(max_foot_position_jump)
            and max_foot_position_jump <= 0.30
        )

        above_threshold = (
            search_frames["foot_speed_smooth"]
            >= threshold
        )

        sustained = above_threshold.copy()

        for shift in range(1, sustain_frames):
            sustained &= above_threshold.shift(
                -shift,
                fill_value=False,
            )

        candidate_indexes = sustained[
            sustained
        ].index

        if len(candidate_indexes) == 0:
            continue

        candidate_index = candidate_indexes[0]
        candidate = search_frames.loc[candidate_index]

        toe_off_timestamp_ms = candidate["timestamp_ms"]

        result_rows.append({
            "side": side,
            "event_type": "toe_off_candidate",
            "frame_index": int(candidate["frame_index"]),
            "timestamp_ms": int(toe_off_timestamp_ms),

            "ic_frame_index": int(cycle["start_frame"]),
            "next_ic_frame_index": int(cycle["end_frame"]),

            "ic_timestamp_ms": int(start_timestamp_ms),
            "next_ic_timestamp_ms": int(end_timestamp_ms),

            "stride_duration_ms": int(stride_duration_ms),

            "contact_time_ms": int(
                toe_off_timestamp_ms
                - start_timestamp_ms
            ),

            "stance_phase_percent": (
                (toe_off_timestamp_ms
                - start_timestamp_ms) / stride_duration_ms * 100
            ),

            "foot_speed_norm_s": (
                candidate["foot_speed_smooth"]
            ),

            "speed_threshold": threshold,
            "max_foot_position_jump": max_foot_position_jump,
            "is_tracking_stable": is_tracking_stable,
            "quality_reason": (
                "ok"
                if is_tracking_stable
                else "unstable_tracking"
            ),
        })

    return pd.DataFrame(result_rows, columns=TOE_OFF_COLUMNS)

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

    result["is_valid_cycle"] = (
        result["cadence_spm"].between(
            120,
            220,
            inclusive="both",
        )
    )
    
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

def calculate_knee_amplitude_for_cycle(frame_df: pd.DataFrame, start_timestamp_ms: int, end_timestamp_ms: int):
    cycle_frames = frame_df.loc[
        (frame_df["timestamp_ms"] >= start_timestamp_ms) &
        (frame_df["timestamp_ms"] <= end_timestamp_ms)
    ].copy()

    left_signal = (
        cycle_frames["left_knee_flexion_smooth"]
        .dropna()
    )

    right_signal = (
        cycle_frames["right_knee_flexion_smooth"]
        .dropna()
    )

    if left_signal.empty or right_signal.empty:
        return None

    left_amplitude = (
        left_signal.max()
        - left_signal.min()
    )

    right_amplitude = (
        right_signal.max()
        - right_signal.min()
    )

    amplitude_difference_deg = (
        left_amplitude - right_amplitude
    )

    return {
        "left_amplitude_deg": left_amplitude,
        "right_amplitude_deg": right_amplitude,
        "amplitude_difference_deg": amplitude_difference_deg,
    }

def knee_amplitude(cycle_df: pd.DataFrame, frame_df: pd.DataFrame):

    quality_cycles = len(cycle_df)

    for i in range(quality_cycles):
        res_amp = calculate_knee_amplitude_for_cycle(
            frame_df,
            start_timestamp_ms=int(cycle_df["start_timestamp_ms"].iloc[i]),
            end_timestamp_ms=int(cycle_df["end_timestamp_ms"].iloc[i]),
        )

        if res_amp is None:
            continue

        for key, value in res_amp.items():
            cycle_df.loc[cycle_df.index[i], key] = value

    return cycle_df

def calculate_ankle_metrics_for_cycle(frame_df: pd.DataFrame, start_timestamp_ms: int, end_timestamp_ms: int):
    cycle_frames = frame_df.loc[
        (frame_df["timestamp_ms"] >= start_timestamp_ms) &
        (frame_df["timestamp_ms"] <= end_timestamp_ms)
    ].copy()

    left_signal = (
        cycle_frames["left_ankle_angle_smooth"]
        .dropna()
    )
    right_signal = (
        cycle_frames["right_ankle_angle_smooth"]
        .dropna()
    )

    if left_signal.empty or right_signal.empty:
        return None

    left_min_angle = left_signal.min()
    left_max_angle = left_signal.max()
    left_rom = left_max_angle - left_min_angle

    right_min_angle = right_signal.min()
    right_max_angle = right_signal.max()
    right_rom = right_max_angle - right_min_angle

    left_coverage = len(left_signal) / len(cycle_frames)
    right_coverage = len(right_signal) / len(cycle_frames)

    if left_coverage < 0.8 or right_coverage < 0.8:
        return None

    return {
        "left_ankle_min_angle_deg": left_min_angle,
        "left_ankle_max_angle_deg": left_max_angle,
        "left_ankle_rom_deg": left_rom,

        "right_ankle_min_angle_deg": right_min_angle,
        "right_ankle_max_angle_deg": right_max_angle,
        "right_ankle_rom_deg": right_rom,
    }

def ankle_metrics(cycle_df: pd.DataFrame, frame_df: pd.DataFrame):
    quality_cycles = len(cycle_df)

    for i in range(quality_cycles):
        res_amp = calculate_ankle_metrics_for_cycle(
            frame_df,
            start_timestamp_ms=int(cycle_df["start_timestamp_ms"].iloc[i]),
            end_timestamp_ms=int(cycle_df["end_timestamp_ms"].iloc[i]),
        )

        if res_amp is None:
            continue

        for key, value in res_amp.items():
            cycle_df.loc[cycle_df.index[i], key] = value
    
    return cycle_df

def calculate_hip_amplitude(frame_df: pd.DataFrame, start_timestamp_ms: int, end_timestamp_ms: int):
    cycle_frames = frame_df.loc[
        (frame_df["timestamp_ms"] >= start_timestamp_ms) &
        (frame_df["timestamp_ms"] <= end_timestamp_ms)
    ].copy()

    left_signal = (
        cycle_frames["left_hip_flexion_smooth"]
        .dropna()
    )
    right_signal =(
        cycle_frames["right_hip_flexion_smooth"]
        .dropna()
    )

    if left_signal.empty or right_signal.empty:
        return None

    left_max_flexion = left_signal.max()
    left_min_extension = left_signal.min()

    right_max_flexion = right_signal.max()
    right_min_extension = right_signal.min()

    left_amplitude = left_max_flexion - left_min_extension
    right_amplitude = right_max_flexion - right_min_extension

    return {
        "left_max_flexion" : left_max_flexion,
        "left_min_extension" : left_min_extension,
        "right_max_flexion" : right_max_flexion,
        "right_min_extension" : right_min_extension,
        "left_amplitude" : left_amplitude,
        "right_amplitude" : right_amplitude,
    }

def hip_amplitude(cycle_df: pd.DataFrame, frame_df: pd.DataFrame):
    quality_cycles = len(cycle_df)

    for i in range(quality_cycles):
        res_amp = calculate_hip_amplitude(
            frame_df,
            start_timestamp_ms=int(cycle_df["start_timestamp_ms"].iloc[i]),
            end_timestamp_ms=int(cycle_df["end_timestamp_ms"].iloc[i]),
        )

        if res_amp is None:
            continue

        for key, value in res_amp.items():
            cycle_df.loc[cycle_df.index[i], key] = value

    return cycle_df

def stats_hip_amplitude(cycle_df: pd.DataFrame):
    valid_cycles = (
        cycle_df.loc[cycle_df["is_high_quality"]]
        .dropna(
            subset=[
                "left_max_flexion",
                "left_min_extension",
                "right_max_flexion",
                "right_min_extension",
                "left_amplitude",
                "right_amplitude",
            ]
        )
        .copy()
    )

    if valid_cycles.empty:
        print("Валидные амплитуды бедер не найдены")
        return

    left_max_mean = valid_cycles["left_max_flexion"].mean()
    left_min_mean = valid_cycles["left_min_extension"].mean()
    left_amplitude_mean = valid_cycles["left_amplitude"].mean()

    right_max_mean = valid_cycles["right_max_flexion"].mean()
    right_min_mean = valid_cycles["right_min_extension"].mean()
    right_amplitude_mean = valid_cycles["right_amplitude"].mean()

    left_max_std = valid_cycles["left_max_flexion"].std()
    left_min_std = valid_cycles["left_min_extension"].std()
    left_amplitude_std = valid_cycles["left_amplitude"].std()

    right_max_std = valid_cycles["right_max_flexion"].std()
    right_min_std = valid_cycles["right_min_extension"].std()
    right_amplitude_std = valid_cycles["right_amplitude"].std()

    extension_diff = abs(abs(left_min_mean) - abs(right_min_mean))
    flexion_diff = abs(abs(left_max_mean) - abs(right_max_mean))
    rom_diff = abs(abs(left_amplitude_mean) - abs(right_amplitude_mean))

    extension_percent = (
        extension_diff /
        ((abs(left_min_mean) + abs(right_min_mean)) / 2) * 100
    )
    flexion_percent = (
        flexion_diff /
        ((abs(left_max_mean) + abs(right_max_mean)) / 2) * 100
    )
    rom_percent = (
        rom_diff /
        ((abs(left_amplitude_mean) + abs(right_amplitude_mean)) / 2) * 100
    )


    print("\n")
    print("Угол бедра относительно туловища")
    print(f"Левая:")
    print(f"  Макс. сгибание: {left_max_mean:.1f}° ± {left_max_std:.1f}°")
    print(f"  Макс. разгибание: {left_min_mean:.1f}° ± {left_min_std:.1f}°")
    print(f"  Амплитуда (ROM): {left_amplitude_mean:.1f}° ± {left_amplitude_std:.1f}°")
    print("\n")
    print(f"Правая:")
    print(f"  Макс. сгибание: {right_max_mean:.1f}° ± {right_max_std:.1f}°")
    print(f"  Макс. разгибание: {right_min_mean:.1f}° ± {right_min_std:.1f}°")
    print(f"  Амплитуда (ROM): {right_amplitude_mean:.1f}° ± {right_amplitude_std:.1f}°")
    print("\n")
    print("Разница L/R:")
    print(f"  Сгибание: {flexion_diff:.1f}° ({flexion_percent:.1f}%)")
    print(f"  Разгибание: {extension_diff:.1f}° ({extension_percent:.1f}%)")
    print(f"  (ROM): {rom_diff:.1f}° ({rom_percent:.1f}%)")
    print(f"Качественных циклов с углами бедра: {len(valid_cycles)}")

def stats_ankle_metrics(cycle_df: pd.DataFrame):
    valid_cycles = (
        cycle_df.loc[cycle_df["is_high_quality"]]
        .dropna(
            subset=[
                "left_ankle_min_angle_deg",
                "left_ankle_max_angle_deg",
                "left_ankle_rom_deg",
                "right_ankle_min_angle_deg",
                "right_ankle_max_angle_deg",
                "right_ankle_rom_deg",
            ]
        )
        .copy()
    )

    if valid_cycles.empty:
        print("Валидные метрики голеностопа не найдены")
        return

    valid_count = len(valid_cycles)

    left_min_mean = valid_cycles["left_ankle_min_angle_deg"].mean()
    right_min_mean = valid_cycles["right_ankle_min_angle_deg"].mean()

    left_max_mean = valid_cycles["left_ankle_max_angle_deg"].mean()
    right_max_mean = valid_cycles["right_ankle_max_angle_deg"].mean()

    left_rom_mean = valid_cycles["left_ankle_rom_deg"].mean()
    right_rom_mean = valid_cycles["right_ankle_rom_deg"].mean()

    left_min_std = valid_cycles["left_ankle_min_angle_deg"].std()
    right_min_std = valid_cycles["right_ankle_min_angle_deg"].std()

    left_max_std = valid_cycles["left_ankle_max_angle_deg"].std()
    right_max_std = valid_cycles["right_ankle_max_angle_deg"].std()

    left_rom_std = valid_cycles["left_ankle_rom_deg"].std()
    right_rom_std = valid_cycles["right_ankle_rom_deg"].std()

    rom_difference = (
        valid_cycles["left_ankle_rom_deg"]
        - valid_cycles["right_ankle_rom_deg"]
    )

    median_rom_difference = rom_difference.abs().median()

    rom_asymmetry_percent = (
        rom_difference.abs()
        / (
            (
                valid_cycles["left_ankle_rom_deg"]
                + valid_cycles["right_ankle_rom_deg"]
            ) / 2
        )
        * 100
    ).median()

    print("\n")
    print("Голеностопный угол")
    print(f"Качественных циклов с валидным сигналом: {valid_count}")
    print("\n")
    print("Левый:")
    print(f"  Минимальный угол: {left_min_mean:.1f}° ± {left_min_std:.1f}° SD")
    print(f"  Максимальный угол: {left_max_mean:.1f}° ± {left_max_std:.1f}° SD")
    print(f"  Амплитуда (ROM): {left_rom_mean:.1f}° ± {left_rom_std:.1f}° SD")
    print("\n")
    print("Правый:")
    print(f"  Минимальный угол: {right_min_mean:.1f}° ± {right_min_std:.1f}° SD")
    print(f"  Максимальный угол: {right_max_mean:.1f}° ± {right_max_std:.1f}° SD")
    print(f"  Амплитуда (ROM): {right_rom_mean:.1f}° ± {right_rom_std:.1f}° SD")
    print(f"\nМедианная разница ROM: {median_rom_difference:.1f}°")
    print(f"Асимметрия ROM: {rom_asymmetry_percent:.1f}%")

def stats_knee_amplitude(cycle_df: pd.DataFrame):
    valid_cycles = (
        cycle_df.loc[cycle_df["is_high_quality"]]
        .dropna(
            subset=[
                "left_amplitude_deg",
                "right_amplitude_deg",
                "amplitude_difference_deg",
            ]
        )
        .copy()
    )

    if valid_cycles.empty:
        print("Валидные амплитуды коленей не найдены")
        return

    left_knee_mean = valid_cycles["left_amplitude_deg"].mean()
    right_knee_mean = valid_cycles["right_amplitude_deg"].mean()

    left_knee_std = valid_cycles["left_amplitude_deg"].std()
    right_knee_std = valid_cycles["right_amplitude_deg"].std()

    asymetry_deg = valid_cycles["amplitude_difference_deg"].abs().median()

    pos_count = (valid_cycles["amplitude_difference_deg"] > 0).sum()
    neg_count = (valid_cycles["amplitude_difference_deg"] < 0).sum()

    directional_count = pos_count + neg_count

    print("\n")
    print(f"Средняя амплитуда левого колена: {left_knee_mean:.1f}° ± {left_knee_std:.1f}° SD")
    print(f"Средняя амплитуда правого колена: {right_knee_mean:.1f}° ± {right_knee_std:.1f}° SD")
    print(f"Медианная разница амплитуд: {asymetry_deg:.1f}°")
    print(f"Левая амплитуда больше: {pos_count} из {directional_count} циклов")
    print(f"Правая амплитуда больше: {neg_count} из {directional_count} циклов")


def stats_trunk_lean(frame_df: pd.DataFrame):
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

    std_trunk_lean = valid_trunk_lean.std()

    val_direction = frame_df["running_direction"].iloc[0]

    if val_direction == 1:
        direction = "Вправо"
    elif val_direction == -1:
        direction = "Влево"
    else:
        direction = "Не определено"

    print("\n")
    print(f"Наклон туловища вперед: {median_trunk_lean:.1f}°")
    print(f"Валидных кадров: {quality_trunk_frame}")
    print(f"Центральные 50%: {q1_trunk_lean:.1f}°-{q3_trunk_lean:.1f}°")
    print(f"Стандартное отклонение: {std_trunk_lean:.1f}°")
    print(f"Направление бега: {direction}")

def stats_contact_cadence(contact_cycles: pd.DataFrame):
    valid_cycles = contact_cycles.loc[
        contact_cycles["is_valid_cycle"]
    ].copy()

    if valid_cycles.empty:
        print("\nКаденс по IC определить невозможно")
        return

    cadence_values = valid_cycles["cadence_spm"]

    median_cadence = cadence_values.median()
    min_cadence = cadence_values.min()
    max_cadence = cadence_values.max()

    left_cadence = (
        valid_cycles.loc[
            valid_cycles["side"] == "left",
            "cadence_spm",
        ]
        .median()
    )

    right_cadence = (
        valid_cycles.loc[
            valid_cycles["side"] == "right",
            "cadence_spm",
        ]
        .median()
    )

    if len(valid_cycles) < 2:
        print(
            "Предупреждение: слишком мало IC-циклов "
            "для устойчивой оценки каденса"
        )

    print("\n")
    print(f"Итоговый каденс по IC: {median_cadence:.1f} spm")
    print(f"Валидных IC-циклов: {len(valid_cycles)}")
    print(
        f"Диапазон: "
        f"{min_cadence:.1f}-{max_cadence:.1f} spm"
    )

    if pd.notna(left_cadence):
        print(f"По левой ноге: {left_cadence:.1f} spm")

    if pd.notna(right_cadence):
        print(f"По правой ноге: {right_cadence:.1f} spm")

def stats_contact(contact_stats: dict):
    if contact_stats is None:
        print("\nTO-кандидаты не найдены")
    elif contact_stats["stable_count"] == 0:
        print("\nСтабильные TO не найдены")
    else:
        print(f"\nОценочное время контакта: {contact_stats['median_contact_time_ms']:.0f} мс")
        print(f"Центральные 50%: {contact_stats['q1_contact_time_ms']:.0f}–{contact_stats['q3_contact_time_ms']:.0f} мс")
        print(f"Стабильных найденных TO: {contact_stats['stable_count']} из {contact_stats['total_count']}")
        print(f"Доля опорной фазы: {contact_stats['median_stance_phase_percent']:.1f}%")

        if contact_stats["left_contact_time_ms"] is not None:
            print(f"Левая нога: {contact_stats['left_contact_time_ms']:.0f} мс")

        if contact_stats["right_contact_time_ms"] is not None:
            print(f"Правая нога: {contact_stats['right_contact_time_ms']:.0f} мс")

        if contact_stats["contact_asymmetry_percent"] is None:
            print(f"Асимметрия времени контакта: недостаточно надёжных циклов")
        else:
            print(f"Асимметрия времени контакта: {contact_stats['contact_asymmetry_percent']:.1f}%")


def calculate_contact_stats(toe_off_df: pd.DataFrame) -> dict | None:
    if toe_off_df.empty:
        return None

    required_columns = {
        "side",
        "contact_time_ms",
        "stride_duration_ms",
        "is_tracking_stable",
    }

    missing_columns = required_columns - set(toe_off_df.columns)
    if missing_columns:
        raise ValueError(
            "Для статистики времени контакта отсутствуют столбцы: "
            f"{sorted(missing_columns)}"
        )

    total_count = len(toe_off_df)

    stable_toe_offs = toe_off_df.loc[
        toe_off_df["is_tracking_stable"].eq(True)
    ].copy()

    stable_toe_offs["contact_time_ms"] = pd.to_numeric(
        stable_toe_offs["contact_time_ms"],
        errors="coerce",
    )
    stable_toe_offs["stride_duration_ms"] = pd.to_numeric(
        stable_toe_offs["stride_duration_ms"],
        errors="coerce",
    )

    stable_toe_offs = stable_toe_offs.dropna(
        subset=["contact_time_ms", "stride_duration_ms"]
    ).copy()

    stable_toe_offs = stable_toe_offs.loc[
        (stable_toe_offs["contact_time_ms"] > 0)
        & (stable_toe_offs["stride_duration_ms"] > 0)
        & (
            stable_toe_offs["contact_time_ms"]
            < stable_toe_offs["stride_duration_ms"]
        )
    ].copy()

    stable_toe_offs["stance_phase_percent"] = (
        stable_toe_offs["contact_time_ms"]
        / stable_toe_offs["stride_duration_ms"]
        * 100
    )

    stable_count = len(stable_toe_offs)

    empty_stats = {
        "stable_count": stable_count,
        "total_count": total_count,
        "median_contact_time_ms": None,
        "q1_contact_time_ms": None,
        "q3_contact_time_ms": None,
        "median_stance_phase_percent": None,
        "left_contact_time_ms": None,
        "right_contact_time_ms": None,
        "contact_asymmetry_percent": None,
    }

    if stable_toe_offs.empty:
        return empty_stats

    contact_times = stable_toe_offs["contact_time_ms"]

    left_contact_times = stable_toe_offs.loc[
        stable_toe_offs["side"].astype(str).str.lower().eq("left"),
        "contact_time_ms",
    ]
    right_contact_times = stable_toe_offs.loc[
        stable_toe_offs["side"].astype(str).str.lower().eq("right"),
        "contact_time_ms",
    ]

    left_contact_time_ms = (
        float(left_contact_times.median())
        if not left_contact_times.empty
        else None
    )
    right_contact_time_ms = (
        float(right_contact_times.median())
        if not right_contact_times.empty
        else None
    )

    contact_asymmetry_percent = None
    if (
        len(left_contact_times) >= 2
        and len(right_contact_times) >= 2
        and left_contact_time_ms is not None
        and right_contact_time_ms is not None
    ):
        mean_contact_time_ms = (
            left_contact_time_ms + right_contact_time_ms
        ) / 2

        if mean_contact_time_ms > 0:
            contact_asymmetry_percent = (
                abs(left_contact_time_ms - right_contact_time_ms)
                / mean_contact_time_ms
                * 100
            )

    return {
        "stable_count": stable_count,
        "total_count": total_count,
        "median_contact_time_ms": float(contact_times.median()),
        "q1_contact_time_ms": float(contact_times.quantile(0.25)),
        "q3_contact_time_ms": float(contact_times.quantile(0.75)),
        "median_stance_phase_percent": float(
            stable_toe_offs["stance_phase_percent"].median()
        ),
        "left_contact_time_ms": left_contact_time_ms,
        "right_contact_time_ms": right_contact_time_ms,
        "contact_asymmetry_percent": contact_asymmetry_percent,
    }

def main():
    args = parse_arg()

    csv_path = Path(args.frame)
    csv_meta_path = Path(args.metadata)
    csv_event_path = Path(args.events)
    plot_path = Path(args.plot)
    csv_cycle_path = Path(args.cycle)
    csv_contact_path = Path(args.contact_foot)
    csv_toe_off_path = Path(args.toe_off)

    for output_path in (
        csv_event_path,
        plot_path,
        csv_cycle_path,
        csv_contact_path,
        csv_toe_off_path,
    ):
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    frame_table = pd.read_csv(str(csv_path))
    df_meta = pd.read_csv(str(csv_meta_path))

    fps = df_meta.loc[0, "fps"]

    result = detect_cadence_peaks(frame_table, fps=fps)

    if result.empty:
        raise ValueError("Пики не найдены")

    res_cadence = find_stride_cadence(result)

    if res_cadence.empty:
        raise ValueError("Пики есть, полного цикла нет")

    res_cadence["is_high_quality"] = (
        res_cadence["is_valid_stride_interval"]
        & res_cadence["alternates_correctly"]
        & ~res_cadence["is_edge_cycle"]
    )

    res_cadence = knee_amplitude(res_cadence, frame_table)
    res_cadence = hip_amplitude(res_cadence, frame_table)
    res_cadence = ankle_metrics(res_cadence, frame_table)
    contact_candidates = build_initial_contact_candidates(res_cadence, frame_table, fps)
    contact_cadence_cycles = find_cycle_cadence(
        contact_candidates
    )

    toe_off_candidates = detect_toe_off(
        frame_table,
        contact_cadence_cycles,
    )

    contact_stats = calculate_contact_stats(toe_off_candidates)

    quality_cycle_count = int(
        res_cadence["is_high_quality"].sum()
    )

    print(
        f"\nКачественных циклов для угловых метрик: "
        f"{quality_cycle_count}"
    )
    stats_contact_cadence(contact_cadence_cycles)
    stats_contact(contact_stats)
    stats_trunk_lean(frame_table)
    stats_knee_amplitude(res_cadence)
    stats_hip_amplitude(res_cadence)
    stats_ankle_metrics(res_cadence)

    plot_diagnostic(frame_table, result, plot_path)

    result.to_csv(str(csv_event_path), index=False)
    res_cadence.to_csv(str(csv_cycle_path), index=False)
    contact_candidates.to_csv(str(csv_contact_path), index=False)
    toe_off_candidates.to_csv(str(csv_toe_off_path), index=False)

if __name__ == "__main__":
    main()
