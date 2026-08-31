from pathlib import Path
from uuid import uuid4

import streamlit as st
import json

from src.pipeline import PROJECT_ROOT, run_pipeline

APP_RUNS_DIR = PROJECT_ROOT / "data" / "app_runs"

PIPELINE_STAGES = [
    "Распознавание позы",
    "Подготовка видео для браузера",
    "Проверка качества кадров",
    "Расчёт углов",
    "Поиск беговых событий",
    "Формирование итогового отчёта",
]

def format_value(value: float | None, unit: str = "", digits: int = 1, missing_value: str = "—") -> str:
    if value is None:
        return missing_value

    formatted_value = f"{value:.{digits}f}"

    if unit:
        return f"{formatted_value} {unit}"

    return formatted_value

def format_range(minimum: float | None, maximum: float | None, unit: str = "", digits: int = 1) -> str:
    if minimum is None or maximum is None:
        return "—"

    result = (f"{minimum:.{digits}f}–{maximum:.{digits}f}")

    if unit:
        return f"{result} {unit}"

    return result


def format_running_direction(direction: str) -> str:
    direction_names = {
        "left": "Влево",
        "right": "Вправо",
        "unknown": "Не определено",
    }

    return direction_names.get(
        direction,
        "Не определено",
    )

def format_mean_std(mean: float | None, std: float | None, unit: str = "", digits: int = 1 ) -> str:
    if mean is None:
        return "—"

    if std is None:
        return format_value(
            mean,
            unit,
            digits,
        )

    result = (
        f"{mean:.{digits}f} ± "
        f"{std:.{digits}f}"
    )

    if unit:
        return f"{result} {unit}"

    return result

def render_knee_summary(knee: dict) -> None:
    if knee["valid_cycle_count"] == 0:
        st.info(
            "Недостаточно качественных циклов "
            "для анализа коленей."
        )
        return

    st.write(
        "Амплитуда показывает диапазон изменения угла колена в течение одного бегового цикла."
    )

    knee_table = [
        {
            "Показатель": "Амплитуда сгибания",
            "Левая сторона": format_mean_std(
                knee["left_mean_rom_deg"],
                knee["left_std_rom_deg"],
                "°",
            ),
            "Правая сторона": format_mean_std(
                knee["right_mean_rom_deg"],
                knee["right_std_rom_deg"],
                "°",
            ),
        }
    ]

    st.dataframe(
        knee_table,
        hide_index=True,
        use_container_width=True,
    )

    column_1, column_2, column_3 = st.columns(3)

    column_1.metric(
        "Медианная разница",
        format_value(
            knee["median_difference_deg"],
            "°",
        ),
    )

    column_2.metric(
        "Левая амплитуда больше",
        f"{knee['left_greater_cycle_count']} циклов",
    )

    column_3.metric(
        "Правая амплитуда больше",
        f"{knee['right_greater_cycle_count']} циклов",
    )

    st.caption(
        f"Качественных циклов для расчёта: {knee['valid_cycle_count']}."
    )

def render_hip_summary(hip: dict) -> None:
    if hip["valid_cycle_count"] == 0:
        st.info(
            "Недостаточно качественных циклов для анализа движения бёдер."
        )
        return

    hip_table = [
        {
            "Показатель": "Максимальное сгибание",
            "Левая сторона": format_mean_std(
                hip["left_mean_flexion_deg"],
                hip["left_std_flexion_deg"],
                "°",
            ),
            "Правая сторона": format_mean_std(
                hip["right_mean_flexion_deg"],
                hip["right_std_flexion_deg"],
                "°",
            ),
            "Разница": format_value(
                hip["flexion_difference_deg"],
                "°",
            ),
            "Разница, %": format_value(
                hip["flexion_difference_percent"],
                "%",
            ),
        },
        {
            "Показатель": "Максимальное разгибание",
            "Левая сторона": format_mean_std(
                hip["left_mean_extension_deg"],
                hip["left_std_extension_deg"],
                "°",
            ),
            "Правая сторона": format_mean_std(
                hip["right_mean_extension_deg"],
                hip["right_std_extension_deg"],
                "°",
            ),
            "Разница": format_value(
                hip["extension_difference_deg"],
                "°",
            ),
            "Разница, %": format_value(
                hip["extension_difference_percent"],
                "%",
            ),
        },
        {
            "Показатель": "Амплитуда движения",
            "Левая сторона": format_mean_std(
                hip["left_mean_rom_deg"],
                hip["left_std_rom_deg"],
                "°",
            ),
            "Правая сторона": format_mean_std(
                hip["right_mean_rom_deg"],
                hip["right_std_rom_deg"],
                "°",
            ),
            "Разница": format_value(
                hip["rom_difference_deg"],
                "°",
            ),
            "Разница, %": format_value(
                hip["rom_difference_percent"],
                "%",
            ),
        },
    ]

    st.dataframe(
        hip_table,
        hide_index=True,
        use_container_width=True,
    )

    st.caption(
        "Качественных циклов для расчёта: "
        f"{hip['valid_cycle_count']}. "
        "Знак угла сгибания и разгибания зависит "
        "от принятой геометрической системы координат."
    )

def render_ankle_summary(ankle: dict) -> None:
    if ankle["valid_cycle_count"] == 0:
        st.info(
            "Недостаточно качественных циклов для анализа голеностопа."
        )
        return

    ankle_table = [
        {
            "Показатель": "Минимальный угол",
            "Левая сторона": format_mean_std(
                ankle["left_mean_min_angle_deg"],
                ankle["left_std_min_angle_deg"],
                "°",
            ),
            "Правая сторона": format_mean_std(
                ankle["right_mean_min_angle_deg"],
                ankle["right_std_min_angle_deg"],
                "°",
            ),
        },
        {
            "Показатель": "Максимальный угол",
            "Левая сторона": format_mean_std(
                ankle["left_mean_max_angle_deg"],
                ankle["left_std_max_angle_deg"],
                "°",
            ),
            "Правая сторона": format_mean_std(
                ankle["right_mean_max_angle_deg"],
                ankle["right_std_max_angle_deg"],
                "°",
            ),
        },
        {
            "Показатель": "Амплитуда движения",
            "Левая сторона": format_mean_std(
                ankle["left_mean_rom_deg"],
                ankle["left_std_rom_deg"],
                "°",
            ),
            "Правая сторона": format_mean_std(
                ankle["right_mean_rom_deg"],
                ankle["right_std_rom_deg"],
                "°",
            ),
        },
    ]

    st.dataframe(
        ankle_table,
        hide_index=True,
        use_container_width=True,
    )

    column_1, column_2 = st.columns(2)

    column_1.metric(
        "Медианная разница амплитуды",
        format_value(
            ankle["median_rom_difference_deg"],
            "°",
        ),
    )

    column_2.metric(
        "Медианная асимметрия амплитуды",
        format_value(
            ankle["median_rom_asymmetry_percent"],
            "%",
        ),
    )

    st.caption(
        f"Качественных циклов для расчёта: {ankle['valid_cycle_count']}."
    )

def get_user_error_message(error: Exception) -> str:
    error_text = str(error).lower()

    pose_errors = [
        "нет найденых поз",
        "нет найденных поз",
    ]

    gait_errors = [
        "нет качественного значения каденса",
        "невозможно определить ic",
        "пики не найдены",
        "полного цикла нет",
    ]

    if any(
        phrase in error_text
        for phrase in pose_errors
    ):
        return (
            "Не удалось обнаружить бегуна в видео. "
            "Убедитесь, что человек полностью находится "
            "в кадре и хорошо виден."
        )

    if any(
        phrase in error_text
        for phrase in gait_errors
    ):
        return (
            "Не удалось определить полные беговые циклы. "
            "Убедитесь, что на видео присутствует бегун, "
            "снятый сбоку, и видно несколько полных шагов."
        )

    if "ffmpeg" in error_text:
        return (
            "Не удалось подготовить видео для анализа. "
            "Попробуйте использовать файл MP4 или MOV."
        )

    return (
        "Не удалось обработать видео. "
        "Проверьте качество записи и попробуйте другой файл."
    )

st.set_page_config(
    page_title="Анализ техники бега",
    page_icon="🏃",
    layout="wide",
)

with st.sidebar:
    st.header("Требования к видео")

    st.markdown(
        """
        - бегун снят сбоку;
        - камера неподвижна;
        - всё тело находится в кадре;
        - освещение позволяет различить ноги;
        - рекомендуется короткое видео длительностью 5–15 секунд.
        """
    )

    st.info(
        "Результаты предназначены для анализа движения "
        "и не являются медицинской диагностикой."
    )
    developer_mode = st.checkbox(
        "Показывать технические ошибки",
        value=False,
    )

st.title("Анализ техники бега по видео")

st.write(
    "Загрузите короткое видео бегуна, снятое сбоку. "
    "Система определит беговые события и рассчитает основные метрики."
)

uploaded_video = st.file_uploader(
    "Выберите видео",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_video is not None:
    uploaded_signature = (
        uploaded_video.name,
        uploaded_video.size,
    )

    previous_signature = st.session_state.get(
        "uploaded_signature"
    )

    if uploaded_signature != previous_signature:
        st.session_state.pop(
            "analysis_result",
            None,
        )
        st.session_state.pop(
            "analysis_run_id",
            None,
        )

        st.session_state["uploaded_signature"] = (
            uploaded_signature
        )

    if "analysis_result" not in st.session_state:
        st.subheader("Исходное видео")
        st.video(uploaded_video)

    start_analysis = st.button(
        "Начать анализ",
        type="primary",
        use_container_width=True,
    )

    if start_analysis:
        st.session_state.pop(
            "analysis_result",
            None,
        )
        st.session_state.pop(
            "analysis_run_id",
            None,
        )

        run_id = uuid4().hex[:12]
        run_dir = APP_RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        video_suffix = Path(uploaded_video.name).suffix.lower()

        if not video_suffix:
            video_suffix = ".mp4"

        input_video_path = run_dir / f"input{video_suffix}"
        input_video_path.write_bytes(uploaded_video.getbuffer())

        progress_bar = st.progress(
            0.0,
            text="Подготовка к анализу",
        )

        status_placeholder = st.empty()

        def update_stage(stage_name: str) -> None:
            if stage_name in PIPELINE_STAGES:
                stage_index = PIPELINE_STAGES.index(stage_name) + 1
            else:
                stage_index = 1

            progress = stage_index / len(PIPELINE_STAGES)

            progress_bar.progress(
                progress,
                text=stage_name,
            )

            status_placeholder.info(
                f"Выполняется: {stage_name}"
            )

        try:
            result = run_pipeline(
                video_path=input_video_path,
                output_dir=run_dir,
                on_stage=update_stage,
            )

        except Exception as error:
            progress_bar.empty()
            status_placeholder.empty()

            st.error(
                get_user_error_message(error)
            )

            if developer_mode:
                with st.expander(
                    "Техническая информация об ошибке"
                ):
                    st.code(str(error))

        else:
            progress_bar.progress(
                1.0,
                text="Анализ завершён",
            )

            status_placeholder.success(
                "Видео успешно обработано"
            )

            st.session_state["analysis_result"] = result
            st.session_state["analysis_run_id"] = run_id

if "analysis_result" in st.session_state:
    result = st.session_state["analysis_result"]

    summary = result["summary"]
    paths = result["paths"]

    quality = summary["quality"]
    warnings = summary["warnings"]
    cadence = summary["cadence"]
    contact = summary["contact"]
    trunk = summary["trunk"]

    st.divider()
    st.header("Результаты анализа")

    if warnings:
        st.subheader("Ограничения анализа")

        for warning in warnings:
            st.warning(warning)
    else:
        st.success(
            "Данных достаточно для расчёта всех метрик."
        )

    st.subheader("Основные показатели")

    column_1, column_2, column_3 = st.columns(3)

    column_1.metric(
        "Каденс",
        format_value(
            cadence["median_spm"],
            "шаг/мин",
        ),
    )

    column_2.metric(
        "Время контакта",
        format_value(
            contact["median_time_ms"],
            "мс",
        ),
    )

    column_3.metric(
        "Наклон туловища",
        format_value(
            trunk["median_lean_deg"],
            "°",
        ),
    )

    st.subheader("Каденс")

    column_1, column_2, column_3, column_4 = (
        st.columns(4)
    )

    column_1.metric(
        "Медианный каденс",
        format_value(
            cadence["median_spm"],
            "шаг/мин",
        ),
    )

    column_2.metric(
        "Диапазон",
        format_range(
            cadence["min_spm"],
            cadence["max_spm"],
            "шаг/мин",
        ),
    )

    column_3.metric(
        "Левая сторона",
        format_value(
            cadence["left_median_spm"],
            "шаг/мин",
        ),
    )

    column_4.metric(
        "Правая сторона",
        format_value(
            cadence["right_median_spm"],
            "шаг/мин",
        ),
    )

    st.caption(
        f"Для расчёта использовано циклов: {cadence['valid_cycle_count']}."
    )

    st.subheader("Контакт стопы с землёй")

    column_1, column_2, column_3, column_4 = (
        st.columns(4)
    )

    column_1.metric(
        "Медианное время",
        format_value(
            contact["median_time_ms"],
            "мс",
        ),
    )

    column_2.metric(
        "Доля опорной фазы",
        format_value(
            contact["median_stance_percent"],
            "%",
        ),
    )

    column_3.metric(
        "Левая сторона",
        format_value(
            contact["left_median_time_ms"],
            "мс",
        ),
    )

    column_4.metric(
        "Правая сторона",
        format_value(
            contact["right_median_time_ms"],
            "мс",
        ),
    )

    st.metric(
        "Асимметрия времени контакта",
        format_value(
            contact["asymmetry_percent"],
            "%",
            missing_value="Недостаточно данных",
        ),
    )

    st.caption(
        f"Стабильных событий отрыва стопы: {contact['stable_count']} из {contact['total_count']}. Асимметрия показывает относительную разницу между левой и правой стороной."
    )

    st.subheader("Наклон туловища")

    column_1, column_2, column_3 = st.columns(3)

    column_1.metric(
        "Медианный наклон",
        format_value(
            trunk["median_lean_deg"],
            "°",
        ),
    )

    column_2.metric(
        "Центральные 50%",
        format_range(
            trunk["q1_lean_deg"],
            trunk["q3_lean_deg"],
            "°",
        ),
    )

    column_3.metric(
        "Стандартное отклонение",
        format_value(
            trunk["std_lean_deg"],
            "°",
        ),
    )

    st.write(
        f"Направление движения: **{format_running_direction(trunk['running_direction'])}**"
    )

    st.caption(
        f"Валидных кадров для расчёта: {trunk['valid_frame_count']}."
    )

    with st.expander("Качество исходных данных"):
        column_1, column_2, column_3, column_4 = (
            st.columns(4)
        )

        column_1.metric(
            "IC-циклы",
            quality["ic_cycle_count"],
        )

        column_2.metric(
            "Угловые циклы",
            quality["angle_cycle_count"],
        )

        column_3.metric(
            "Стабильные TO",
            quality["stable_to_count"],
        )

        column_4.metric(
            "Все TO",
            quality["total_to_count"],
        )

    st.subheader("Углы суставов")

    knee_tab, hip_tab, ankle_tab = st.tabs(
        [
            "Колени",
            "Бёдра",
            "Голеностоп",
        ]
    )

    with knee_tab:
        render_knee_summary(
            summary["knee"]
        )

    with hip_tab:
        render_hip_summary(
            summary["hip"]
        )

    with ankle_tab:
        render_ankle_summary(
            summary["ankle"]
        )

    input_video_path = Path(
        paths["input_video"]
    )

    annotated_video_path = Path(
        paths["annotated_video"]
    )

    st.subheader("Сравнение видео")

    source_column, result_column = st.columns(2)

    with source_column:
        st.write("**Исходное видео**")

        if input_video_path.exists():
            st.video(str(input_video_path))
        else:
            st.warning("Исходное видео не найдено.")

    with result_column:
        st.write("**Видео с ключевыми точками**")

        if annotated_video_path.exists():
            st.video(str(annotated_video_path))
        else:
            st.warning(
                "Обработанное видео не найдено."
            )

    summary_json = json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    )

    st.download_button(
        label="Скачать результаты анализа",
        data=summary_json,
        file_name="running_analysis.json",
        mime="application/json",
        use_container_width=True,
    )

    with st.expander("Технические данные"):
        st.json(
            summary,
            expanded=False,
    )