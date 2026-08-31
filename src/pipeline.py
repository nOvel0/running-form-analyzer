from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

import pandas as pd

from src.analysis.analysis_summary import build_analysis_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models/pose_landmarker_full.task"

def run_command(stage_name: str, script_path: str, arguments: list[str | Path], on_stage: Callable[[str], None] | None = None) -> None:
    if on_stage is not None:
        on_stage(stage_name)

    command = [
        sys.executable,
        str(PROJECT_ROOT / script_path),
        *[str(argument) for argument in arguments],
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip()

        # Не выводим в интерфейс десятки тысяч символов.
        error_text = error_text[-4000:]

        raise RuntimeError(f"Ошибка на этапе «{stage_name}»:\n{error_text}")

def run_pipeline(video_path: Path, output_dir: Path, on_stage: Callable[[str], None] | None = None) -> dict:
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель MediaPipe не найдена: {MODEL_PATH}")

    output_dir.mkdir(parents=True, exist_ok=True)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "raw_keypoints": output_dir / "raw_keypoints.csv",
        "valid_keypoints": output_dir / "valid_keypoints.csv",
        "metadata": output_dir / "metadata.csv",
        "frame_table": output_dir / "frame_table.csv",
        "events": output_dir / "events.csv",
        "angle_cycles": output_dir / "angle_cycles.csv",
        "contacts": output_dir / "contacts.csv",
        "toe_off": output_dir / "toe_off.csv",
        "contact_cycles": output_dir / "contact_cycles.csv",
        "normalized_video": output_dir / "normalized.mp4",
        "annotated_video_raw": output_dir / "annotated_raw.mp4",
        "annotated_video": output_dir / "annotated.mp4",
        "diagnostic_plot": plots_dir / "diagnostic.png",
        "plots_dir": plots_dir,
        "input_video": video_path,
    }

    run_command(
        "Распознавание позы",
        "src/video/extract_pose.py",
        [
            "--video", video_path,
            "--keypoints", paths["raw_keypoints"],
            "--output", paths["annotated_video_raw"],
            "--model", MODEL_PATH,
            "--output_meta", paths["metadata"],
            "--normalized_video", paths["normalized_video"],
        ],
        on_stage,
    )

    convert_video_for_browser(
        input_path=paths["annotated_video_raw"],
        output_path=paths["annotated_video"],
        on_stage=on_stage,
    )

    run_command(
        "Проверка качества кадров",
        "src/analysis/frame_quality.py",
        [
            "--keypoints", paths["raw_keypoints"],
            "--output", paths["valid_keypoints"],
        ],
        on_stage,
    )

    run_command(
        "Расчёт углов",
        "src/analysis/calculate_metrics.py",
        [
            "--keypoints", paths["valid_keypoints"],
            "--metadata", paths["metadata"],
            "--frame_table", paths["frame_table"],
            "--plot", plots_dir,
        ],
        on_stage,
    )

    run_command(
        "Поиск беговых событий",
        "src/analysis/detect_gait_events.py",
        [
            "--frame", paths["frame_table"],
            "--metadata", paths["metadata"],
            "--plot", paths["diagnostic_plot"],
            "--events", paths["events"],
            "--cycle", paths["angle_cycles"],
            "--contact_foot", paths["contacts"],
            "--toe_off", paths["toe_off"],
            "--cadence", paths["contact_cycles"],
        ],
        on_stage,
    )

    if on_stage is not None:
        on_stage("Формирование итогового отчёта")

    summary = build_analysis_summary(
        frame_df=pd.read_csv(paths["frame_table"]),
        angle_cycles_df=pd.read_csv(paths["angle_cycles"]),
        contact_cycles_df=pd.read_csv(paths["contact_cycles"]),
        toe_off_df=pd.read_csv(paths["toe_off"]),
    )

    return {
        "summary": summary,
        "paths": paths,
    }

def convert_video_for_browser(input_path: Path, output_path: Path, on_stage: Callable[[str], None] | None = None) -> None:
    stage_name = "Подготовка видео для браузера"

    if on_stage is not None:
        on_stage(stage_name)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        error_text = (
            result.stderr.strip()
            or result.stdout.strip()
        )

        raise RuntimeError(
            f"Не удалось подготовить видео для браузера:\n{error_text[-4000:]}"
        )

    if not output_path.exists():
        raise RuntimeError(
            "FFmpeg завершился без ошибки, но итоговое видео не создано."
        )