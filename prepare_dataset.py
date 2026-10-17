from pathlib import Path
import re
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

VIDEOS_DIR = DATASET_DIR
XLSX_PATH = DATASET_DIR / "open_labels.xlsx"
CSV_PATH = DATASET_DIR / "labels.csv"


def extract_number(filename: str) -> int:
    """
    Extract video number from names like:
    X-1.mov
    Х-2.mov
    X-10.mov
    """
    match = re.search(r"[-_](\d+)", filename)

    if not match:
        raise ValueError(f"Не удалось найти номер в имени файла: {filename}")

    return int(match.group(1))


def parse_timestamp(value) -> float | None:
    """
    Convert values like:
        '22 секунда'
        '57 секунда'
        NaN

    into:
        22
        57
        None
    """
    if pd.isna(value):
        return None

    text = str(value).strip().lower().replace(",", ".")
    number = r"(\d+(?:\.\d+)?)"
    minute = re.search(number + r"\s*(?:минута|минуты|минут|мин\.?)", text)
    second = re.search(number + r"\s*(?:секунда|секунды|секунд|сек\.?)", text)
    if minute:
        return float(minute.group(1)) * 60 + (float(second.group(1)) if second else 0.0)
    if second:
        return float(second.group(1))
    fallback = re.search(number, text)
    return float(fallback.group(1)) if fallback else None


def parse_violation(value) -> bool:
    """
    Convert Russian labels to boolean.
    """
    value = str(value).strip().lower()

    if value == "нарушение":
        return True

    if value == "без нарушения":
        return False

    raise ValueError(f"Неизвестная метка нарушения: {value}")


def normalize_name(name: str) -> str:
    """
    Normalize X/Х difference and return numeric video name.

    X-1  -> 1.mov
    Х-1  -> 1.mov
    """
    number = extract_number(name)
    return f"{number}.mov"


def rename_videos():
    """
    Rename all videos in two stages using temporary names.
    """

    video_extensions = {".mov", ".mp4", ".avi", ".mkv"}

    videos = [
        path
        for path in VIDEOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in video_extensions
    ]

    if not videos:
        raise RuntimeError(f"Видео не найдены в {VIDEOS_DIR}")

    mapping = []

    for path in videos:
        number = extract_number(path.name)
        new_name = f"{number}.mov"

        mapping.append((path, VIDEOS_DIR / new_name))

    # Проверяем дубликаты номеров
    target_names = [target.name for _, target in mapping]

    if len(target_names) != len(set(target_names)):
        raise RuntimeError(
            "Обнаружены несколько видео с одинаковым номером."
        )

    # Проверяем, что разные файлы действительно не пытаются стать одним
    source_names = {source.name for source, _ in mapping}

    # Первый этап: временные имена
    temp_mapping = []

    for index, (source, target) in enumerate(mapping):
        temp_path = VIDEOS_DIR / f".tmp_video_{index}.mov"

        if temp_path.exists():
            temp_path.unlink()

        source.rename(temp_path)
        temp_mapping.append((temp_path, target))

    # Второй этап: окончательные имена
    for temp_path, target in temp_mapping:
        if target.exists():
            target.unlink()

        temp_path.rename(target)

    print(f"Переименовано видео: {len(mapping)}")

    for _, target in mapping:
        print(f"  -> {target.name}")


def create_csv():
    """
    Read XLSX and convert it to normalized CSV.
    """

    if not XLSX_PATH.exists():
        raise FileNotFoundError(f"XLSX не найден: {XLSX_PATH}")

    df = pd.read_excel(XLSX_PATH)

    required_columns = {
        "наименование файла",
        "таймстемп",
        "нарушение / без нарушения",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"В XLSX отсутствуют колонки: {', '.join(missing)}"
        )

    result = pd.DataFrame()

    result["name"] = df["наименование файла"].apply(normalize_name)
    result["timestamp"] = df["таймстемп"].apply(parse_timestamp)
    result["violation"] = (
        df["нарушение / без нарушения"]
        .apply(parse_violation)
    )

    result.to_csv(
        CSV_PATH,
        index=False,
        encoding="utf-8",
    )

    print(f"\nCSV создан: {CSV_PATH}")
    print(f"Строк: {len(result)}")


def main():
    print("=== Dataset preparation ===\n")

    print("[1/2] Переименование видео...")
    rename_videos()

    print("\n[2/2] Создание CSV...")
    create_csv()

    print("\nГотово!")


if __name__ == "__main__":
    main()
