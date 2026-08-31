import pandas as pd
import argparse
import json
from pathlib import Path

def parse_arg():
    parser = argparse.ArgumentParser(
        description="Analyze running metrics csv"
    )

    parser.add_argument(
        "--frame_df",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--cadence_df",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--contact_df",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--toe_off_df",
        type=str,
        required=True,
    )

    return parser.parse_args()

def optional_float(value) -> float | None:
    if pd.isna(value):
        return None

    return float(value)

def true_mask(series: pd.Series) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

def symmetric_difference_percent(left_value: float | None, right_value: float | None) -> float | None:
    if left_value is None or right_value is None:
        return None

    left_magnitude = abs(left_value)
    right_magnitude = abs(right_value)

    mean_magnitude = (
        left_magnitude + right_magnitude
    ) / 2

    if mean_magnitude == 0:
        return None

    difference = abs(
        left_magnitude - right_magnitude
    )

    return float(
        difference / mean_magnitude * 100
    )

def build_cadence_summary(contact_cycles_df: pd.DataFrame) -> dict:
    empty_result = {
        "valid_cycle_count": 0,
        "median_spm": None,
        "min_spm": None,
        "max_spm": None,
        "left_median_spm": None,
        "right_median_spm": None,
    }

    if contact_cycles_df.empty:
        return empty_result

    required_columns = {
        "side",
        "cadence_spm",
        "is_valid_cycle",
    }

    missing_columns = (
        required_columns
        - set(contact_cycles_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Для статистики каденса отсутствуют столбцы: "
            f"{sorted(missing_columns)}"
        )

    valid_cycles = contact_cycles_df.loc[
        true_mask(
            contact_cycles_df["is_valid_cycle"]
        )
    ].copy()

    if valid_cycles.empty:
        return empty_result

    cadence = pd.to_numeric(
        valid_cycles["cadence_spm"],
        errors="coerce",
    ).dropna()

    if cadence.empty:
        return empty_result

    left_cadence = pd.to_numeric(
        valid_cycles.loc[
            valid_cycles["side"].eq("left"),
            "cadence_spm",
        ],
        errors="coerce",
    ).dropna()

    right_cadence = pd.to_numeric(
        valid_cycles.loc[
            valid_cycles["side"].eq("right"),
            "cadence_spm",
        ],
        errors="coerce",
    ).dropna()

    return {
        "valid_cycle_count": int(len(cadence)),
        "median_spm": optional_float(cadence.median()),
        "min_spm": optional_float(cadence.min()),
        "max_spm": optional_float(cadence.max()),
        "left_median_spm": optional_float(
            left_cadence.median()
        ),
        "right_median_spm": optional_float(
            right_cadence.median()
        ),
    }

def build_contact_summary(toe_off_df: pd.DataFrame) -> dict:
    empty_result = {
        "total_count": int(len(toe_off_df)),
        "stable_count": 0,
        "left_stable_count": 0,
        "right_stable_count": 0,
        "median_time_ms": None,
        "q1_time_ms": None,
        "q3_time_ms": None,
        "median_stance_percent": None,
        "left_median_time_ms": None,
        "right_median_time_ms": None,
        "asymmetry_percent": None,
    }

    if toe_off_df.empty:
        return empty_result

    required_columns = {
        "side",
        "contact_time_ms",
        "stride_duration_ms",
        "is_tracking_stable",
    }

    missing_columns = (
        required_columns
        - set(toe_off_df.columns)
    )

    if missing_columns:
        raise ValueError(f"Для статистики контакта отсутствуют столбцы: {sorted(missing_columns)}")

    stable_mask = true_mask(
        toe_off_df["is_tracking_stable"]
    )

    stable_toe_offs = toe_off_df.loc[
        stable_mask
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
        subset=[
            "contact_time_ms",
            "stride_duration_ms",
        ]
    ).copy()

    stable_toe_offs = stable_toe_offs.loc[
        (stable_toe_offs["contact_time_ms"] > 0)
        & (stable_toe_offs["stride_duration_ms"] > 0)
        & (
            stable_toe_offs["contact_time_ms"]
            < stable_toe_offs["stride_duration_ms"]
        )
    ].copy()

    if stable_toe_offs.empty:
        return empty_result

    stable_toe_offs["stance_percent"] = (
        stable_toe_offs["contact_time_ms"]
        / stable_toe_offs["stride_duration_ms"]
        * 100
    )

    normalized_side = (
        stable_toe_offs["side"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    left_contact_times = stable_toe_offs.loc[
        normalized_side.eq("left"),
        "contact_time_ms",
    ]

    right_contact_times = stable_toe_offs.loc[
        normalized_side.eq("right"),
        "contact_time_ms",
    ]

    left_median_time_ms = optional_float(
        left_contact_times.median()
    )

    right_median_time_ms = optional_float(
        right_contact_times.median()
    )

    asymmetry_percent = None

    if (
        len(left_contact_times) >= 2
        and len(right_contact_times) >= 2
        and left_median_time_ms is not None
        and right_median_time_ms is not None
    ):
        mean_contact_time_ms = (
            left_median_time_ms
            + right_median_time_ms
        ) / 2

        if mean_contact_time_ms > 0:
            asymmetry_percent = (
                abs(
                    left_median_time_ms
                    - right_median_time_ms
                )
                / mean_contact_time_ms
                * 100
            )

    contact_times = stable_toe_offs["contact_time_ms"]

    return {
        "total_count": int(len(toe_off_df)),
        "stable_count": int(len(stable_toe_offs)),
        "left_stable_count": int(
            len(left_contact_times)
        ),
        "right_stable_count": int(
            len(right_contact_times)
        ),
        "median_time_ms": optional_float(
            contact_times.median()
        ),
        "q1_time_ms": optional_float(
            contact_times.quantile(0.25)
        ),
        "q3_time_ms": optional_float(
            contact_times.quantile(0.75)
        ),
        "median_stance_percent": optional_float(
            stable_toe_offs["stance_percent"].median()
        ),
        "left_median_time_ms": left_median_time_ms,
        "right_median_time_ms": right_median_time_ms,
        "asymmetry_percent": optional_float(
            asymmetry_percent
        ),
    }

def build_trunk_summary(frame_df: pd.DataFrame) -> dict:
    empty_result = {
        "valid_frame_count": 0,
        "median_lean_deg": None,
        "q1_lean_deg": None,
        "q3_lean_deg": None,
        "std_lean_deg": None,
        "running_direction": "unknown",
    }

    if frame_df.empty:
        return empty_result

    required_columns = {
        "trunk_valid",
        "forward_trunk_lean_deg",
        "running_direction",
    }

    missing_columns = (
        required_columns
        - set(frame_df.columns)
    )

    if missing_columns:
        raise ValueError(f"Для статистики наклона туловища отсутствуют столбцы: {sorted(missing_columns)}")

    trunk_valid_mask = true_mask(
        frame_df["trunk_valid"]
    )

    valid_trunk_lean = pd.to_numeric(
        frame_df.loc[
            trunk_valid_mask,
            "forward_trunk_lean_deg",
        ],
        errors="coerce",
    ).dropna()

    direction_values = pd.to_numeric(
        frame_df["running_direction"],
        errors="coerce",
    ).dropna()

    running_direction = "unknown"

    if not direction_values.empty:
        direction_sign = direction_values.median()

        if direction_sign > 0:
            running_direction = "right"
        elif direction_sign < 0:
            running_direction = "left"

    if valid_trunk_lean.empty:
        empty_result["running_direction"] = (
            running_direction
        )
        return empty_result

    return {
        "valid_frame_count": int(
            len(valid_trunk_lean)
        ),
        "median_lean_deg": optional_float(
            valid_trunk_lean.median()
        ),
        "q1_lean_deg": optional_float(
            valid_trunk_lean.quantile(0.25)
        ),
        "q3_lean_deg": optional_float(
            valid_trunk_lean.quantile(0.75)
        ),
        "std_lean_deg": optional_float(
            valid_trunk_lean.std()
        ),
        "running_direction": running_direction,
    }

def build_knee_summary(angle_cycles_df: pd.DataFrame) -> dict:
    empty_result = {
        "valid_cycle_count": 0,
        "left_mean_rom_deg": None,
        "left_std_rom_deg": None,
        "right_mean_rom_deg": None,
        "right_std_rom_deg": None,
        "median_difference_deg": None,
        "left_greater_cycle_count": 0,
        "right_greater_cycle_count": 0,
    }

    if angle_cycles_df.empty:
        return empty_result

    required_columns = {
        "is_high_quality",
        "left_amplitude_deg",
        "right_amplitude_deg",
        "amplitude_difference_deg",
    }

    missing_columns = (
        required_columns
        - set(angle_cycles_df.columns)
    )

    if missing_columns:
        raise ValueError(f"Для статистики коленей отсутствуют столбцы: {sorted(missing_columns)}")

    quality_mask = true_mask(
        angle_cycles_df["is_high_quality"]
    )

    valid_cycles = angle_cycles_df.loc[
        quality_mask,
        [
            "left_amplitude_deg",
            "right_amplitude_deg",
            "amplitude_difference_deg",
        ],
    ].copy()

    numeric_columns = [
        "left_amplitude_deg",
        "right_amplitude_deg",
        "amplitude_difference_deg",
    ]

    for column in numeric_columns:
        valid_cycles[column] = pd.to_numeric(
            valid_cycles[column],
            errors="coerce",
        )

    valid_cycles = valid_cycles.dropna(
        subset=numeric_columns
    ).copy()

    if valid_cycles.empty:
        return empty_result

    left_amplitude = (
        valid_cycles["left_amplitude_deg"]
    )

    right_amplitude = (
        valid_cycles["right_amplitude_deg"]
    )

    amplitude_difference = (
        valid_cycles["amplitude_difference_deg"]
    )

    return {
        "valid_cycle_count": int(
            len(valid_cycles)
        ),
        "left_mean_rom_deg": optional_float(
            left_amplitude.mean()
        ),
        "left_std_rom_deg": optional_float(
            left_amplitude.std()
        ),
        "right_mean_rom_deg": optional_float(
            right_amplitude.mean()
        ),
        "right_std_rom_deg": optional_float(
            right_amplitude.std()
        ),
        "median_difference_deg": optional_float(
            amplitude_difference.abs().median()
        ),
        "left_greater_cycle_count": int(
            (amplitude_difference > 0).sum()
        ),
        "right_greater_cycle_count": int(
            (amplitude_difference < 0).sum()
        ),
    }

def build_hip_summary(angle_cycles_df: pd.DataFrame) -> dict:
    empty_result = {
        "valid_cycle_count": 0,

        "left_mean_flexion_deg": None,
        "left_std_flexion_deg": None,
        "left_mean_extension_deg": None,
        "left_std_extension_deg": None,
        "left_mean_rom_deg": None,
        "left_std_rom_deg": None,

        "right_mean_flexion_deg": None,
        "right_std_flexion_deg": None,
        "right_mean_extension_deg": None,
        "right_std_extension_deg": None,
        "right_mean_rom_deg": None,
        "right_std_rom_deg": None,

        "flexion_difference_deg": None,
        "flexion_difference_percent": None,
        "extension_difference_deg": None,
        "extension_difference_percent": None,
        "rom_difference_deg": None,
        "rom_difference_percent": None,
    }

    if angle_cycles_df.empty:
        return empty_result

    metric_columns = [
        "left_max_flexion",
        "left_min_extension",
        "left_amplitude",
        "right_max_flexion",
        "right_min_extension",
        "right_amplitude",
    ]

    required_columns = {
        "is_high_quality",
        *metric_columns,
    }

    missing_columns = (
        required_columns
        - set(angle_cycles_df.columns)
    )

    if missing_columns:
        raise ValueError(f"Для статистики бедра отсутствуют столбцы: {sorted(missing_columns)}")

    quality_mask = true_mask(
        angle_cycles_df["is_high_quality"]
    )

    valid_cycles = angle_cycles_df.loc[
        quality_mask,
        metric_columns,
    ].copy()

    for column in metric_columns:
        valid_cycles[column] = pd.to_numeric(
            valid_cycles[column],
            errors="coerce",
        )

    valid_cycles = valid_cycles.dropna(
        subset=metric_columns
    ).copy()

    if valid_cycles.empty:
        return empty_result

    left_flexion = valid_cycles[
        "left_max_flexion"
    ]
    left_extension = valid_cycles[
        "left_min_extension"
    ]
    left_rom = valid_cycles[
        "left_amplitude"
    ]

    right_flexion = valid_cycles[
        "right_max_flexion"
    ]
    right_extension = valid_cycles[
        "right_min_extension"
    ]
    right_rom = valid_cycles[
        "right_amplitude"
    ]

    left_mean_flexion = optional_float(
        left_flexion.mean()
    )
    left_mean_extension = optional_float(
        left_extension.mean()
    )
    left_mean_rom = optional_float(
        left_rom.mean()
    )

    right_mean_flexion = optional_float(
        right_flexion.mean()
    )
    right_mean_extension = optional_float(
        right_extension.mean()
    )
    right_mean_rom = optional_float(
        right_rom.mean()
    )

    flexion_difference_deg = abs(
        abs(left_mean_flexion)
        - abs(right_mean_flexion)
    )

    extension_difference_deg = abs(
        abs(left_mean_extension)
        - abs(right_mean_extension)
    )

    rom_difference_deg = abs(
        abs(left_mean_rom)
        - abs(right_mean_rom)
    )

    return {
        "valid_cycle_count": int(
            len(valid_cycles)
        ),

        "left_mean_flexion_deg": left_mean_flexion,
        "left_std_flexion_deg": optional_float(
            left_flexion.std()
        ),
        "left_mean_extension_deg": left_mean_extension,
        "left_std_extension_deg": optional_float(
            left_extension.std()
        ),
        "left_mean_rom_deg": left_mean_rom,
        "left_std_rom_deg": optional_float(
            left_rom.std()
        ),

        "right_mean_flexion_deg": right_mean_flexion,
        "right_std_flexion_deg": optional_float(
            right_flexion.std()
        ),
        "right_mean_extension_deg": right_mean_extension,
        "right_std_extension_deg": optional_float(
            right_extension.std()
        ),
        "right_mean_rom_deg": right_mean_rom,
        "right_std_rom_deg": optional_float(
            right_rom.std()
        ),

        "flexion_difference_deg": optional_float(
            flexion_difference_deg
        ),
        "flexion_difference_percent": (
            symmetric_difference_percent(
                left_mean_flexion,
                right_mean_flexion,
            )
        ),

        "extension_difference_deg": optional_float(
            extension_difference_deg
        ),
        "extension_difference_percent": (
            symmetric_difference_percent(
                left_mean_extension,
                right_mean_extension,
            )
        ),

        "rom_difference_deg": optional_float(
            rom_difference_deg
        ),
        "rom_difference_percent": (
            symmetric_difference_percent(
                left_mean_rom,
                right_mean_rom,
            )
        ),
    }

def build_ankle_summary(angle_cycles_df: pd.DataFrame) -> dict:
    empty_result = {
        "valid_cycle_count": 0,

        "left_mean_min_angle_deg": None,
        "left_std_min_angle_deg": None,
        "left_mean_max_angle_deg": None,
        "left_std_max_angle_deg": None,
        "left_mean_rom_deg": None,
        "left_std_rom_deg": None,

        "right_mean_min_angle_deg": None,
        "right_std_min_angle_deg": None,
        "right_mean_max_angle_deg": None,
        "right_std_max_angle_deg": None,
        "right_mean_rom_deg": None,
        "right_std_rom_deg": None,

        "median_rom_difference_deg": None,
        "median_rom_asymmetry_percent": None,
    }

    if angle_cycles_df.empty:
        return empty_result

    metric_columns = [
        "left_ankle_min_angle_deg",
        "left_ankle_max_angle_deg",
        "left_ankle_rom_deg",
        "right_ankle_min_angle_deg",
        "right_ankle_max_angle_deg",
        "right_ankle_rom_deg",
    ]

    required_columns = {
        "is_high_quality",
        *metric_columns,
    }

    missing_columns = (
        required_columns
        - set(angle_cycles_df.columns)
    )

    if missing_columns:
        raise ValueError(f"Для статистики голеностопа отсутствуют столбцы: {sorted(missing_columns)}")

    quality_mask = true_mask(
        angle_cycles_df["is_high_quality"]
    )

    valid_cycles = angle_cycles_df.loc[
        quality_mask,
        metric_columns,
    ].copy()

    for column in metric_columns:
        valid_cycles[column] = pd.to_numeric(
            valid_cycles[column],
            errors="coerce",
        )

    valid_cycles = valid_cycles.dropna(
        subset=metric_columns
    ).copy()

    if valid_cycles.empty:
        return empty_result

    left_min_angle = valid_cycles[
        "left_ankle_min_angle_deg"
    ]
    left_max_angle = valid_cycles[
        "left_ankle_max_angle_deg"
    ]
    left_rom = valid_cycles[
        "left_ankle_rom_deg"
    ]

    right_min_angle = valid_cycles[
        "right_ankle_min_angle_deg"
    ]
    right_max_angle = valid_cycles[
        "right_ankle_max_angle_deg"
    ]
    right_rom = valid_cycles[
        "right_ankle_rom_deg"
    ]

    rom_difference = (
        left_rom - right_rom
    ).abs()

    mean_rom = (
        left_rom + right_rom
    ) / 2

    rom_asymmetry_percent = (
        rom_difference
        / mean_rom.where(mean_rom > 0)
        * 100
    ).dropna()

    return {
        "valid_cycle_count": int(
            len(valid_cycles)
        ),

        "left_mean_min_angle_deg": optional_float(
            left_min_angle.mean()
        ),
        "left_std_min_angle_deg": optional_float(
            left_min_angle.std()
        ),
        "left_mean_max_angle_deg": optional_float(
            left_max_angle.mean()
        ),
        "left_std_max_angle_deg": optional_float(
            left_max_angle.std()
        ),
        "left_mean_rom_deg": optional_float(
            left_rom.mean()
        ),
        "left_std_rom_deg": optional_float(
            left_rom.std()
        ),

        "right_mean_min_angle_deg": optional_float(
            right_min_angle.mean()
        ),
        "right_std_min_angle_deg": optional_float(
            right_min_angle.std()
        ),
        "right_mean_max_angle_deg": optional_float(
            right_max_angle.mean()
        ),
        "right_std_max_angle_deg": optional_float(
            right_max_angle.std()
        ),
        "right_mean_rom_deg": optional_float(
            right_rom.mean()
        ),
        "right_std_rom_deg": optional_float(
            right_rom.std()
        ),

        "median_rom_difference_deg": optional_float(
            rom_difference.median()
        ),
        "median_rom_asymmetry_percent": optional_float(
            rom_asymmetry_percent.median()
        ),
    }

def build_warnings(
    cadence: dict,
    contact: dict,
    trunk: dict,
    knee: dict,
    hip: dict,
    ankle: dict,
) -> list[str]:
    warnings = []

    cadence_cycle_count = (
        cadence["valid_cycle_count"]
    )

    if cadence_cycle_count == 0:
        warnings.append(
            "Каденс определить невозможно"
        )
    elif cadence_cycle_count < 2:
        warnings.append(
            "Недостаточно IC-циклов для устойчивой оценки каденса"
        )

    if trunk["valid_frame_count"] == 0:
        warnings.append(
            "Наклон туловища определить невозможно"
        )

    missing_angle_metrics = []

    if knee["valid_cycle_count"] == 0:
        missing_angle_metrics.append("колено")

    if hip["valid_cycle_count"] == 0:
        missing_angle_metrics.append("бедро")

    if ankle["valid_cycle_count"] == 0:
        missing_angle_metrics.append("голеностоп")

    if len(missing_angle_metrics) == 3:
        warnings.append(
            "Угловые метрики определить невозможно"
        )
    else:
        for metric_name in missing_angle_metrics:
            warnings.append(
                f"Недостаточно данных для метрики: {metric_name}"
            )

    stable_to_count = contact["stable_count"]

    if stable_to_count == 0:
        warnings.append(
            "Стабильные TO не найдены"
        )
    else:
        if stable_to_count < 3:
            warnings.append(
                "Мало стабильных TO для оценки времени контакта"
            )

        if (
            contact["left_stable_count"] < 2
            or contact["right_stable_count"] < 2
        ):
            warnings.append(
                "Недостаточно стабильных TO для оценки асимметрии контакта"
            )

    return warnings

def build_analysis_summary(
    frame_df: pd.DataFrame,
    angle_cycles_df: pd.DataFrame,
    contact_cycles_df: pd.DataFrame,
    toe_off_df: pd.DataFrame,
) -> dict:
    cadence = build_cadence_summary(contact_cycles_df)
    contact = build_contact_summary(toe_off_df)
    trunk = build_trunk_summary(frame_df)
    knee = build_knee_summary(angle_cycles_df)
    hip = build_hip_summary(angle_cycles_df)
    ankle = build_ankle_summary(angle_cycles_df)

    angle_cycle_count = 0

    if "is_high_quality" in angle_cycles_df.columns:
        angle_cycle_count = int(true_mask(angle_cycles_df["is_high_quality"]).sum())

    warnings = build_warnings(cadence, contact, trunk, knee, hip, ankle)

    return {
        "quality": {
            "has_warnings": bool(warnings),
            "angle_cycle_count": angle_cycle_count,
            "ic_cycle_count": cadence["valid_cycle_count"],
            "stable_to_count": contact["stable_count"],
            "total_to_count": contact["total_count"],
        },
        "warnings": warnings,
        "cadence": cadence,
        "contact": contact,
        "trunk": trunk,
        "knee": knee,
        "hip": hip,
        "ankle": ankle,
    }

def round_for_display(value, digits: int = 2):
    if isinstance(value, dict):
        return {
            key: round_for_display(item, digits)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            round_for_display(item, digits)
            for item in value
        ]

    if isinstance(value, float):
        return round(value, digits)

    return value

def main():
    args = parse_arg()

    csv_frame_df = Path(args.frame_df)
    csv_cadence_df = Path(args.cadence_df)
    csv_contact_df = Path(args.contact_df)
    csv_toe_off_df = Path(args.toe_off_df)

    frame_table = pd.read_csv(csv_frame_df)
    res_cadence = pd.read_csv(csv_cadence_df)
    contact_cadence_cycles = pd.read_csv(csv_contact_df)
    toe_off_candidates = pd.read_csv(csv_toe_off_df)

    summary = build_analysis_summary(
        frame_df=frame_table,
        angle_cycles_df=res_cadence,
        contact_cycles_df=contact_cadence_cycles,
        toe_off_df=toe_off_candidates,
    )

    display_summary = round_for_display(
        summary,
        digits=2,
    )

    print(json.dumps(display_summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()