import os
import json
import hashlib
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


BASE_DIR = Path(r"D:\other\QTrustPhish\Dataset")
OUTPUT_DIR = Path(r"D:\other\QTrustPhish\output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_OUT = OUTPUT_DIR / "qtrustphish_full_dataset_summary.xlsx"
JSON_OUT = OUTPUT_DIR / "qtrustphish_full_dataset_summary.json"


TEXT_EXTS = [".csv", ".txt", ".tsv"]
IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]


def file_hash(path, block_size=65536):
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(block_size), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None


def safe_read_table(path):
    encodings = ["utf-8", "latin1", "cp1252", "ISO-8859-1"]

    for enc in encodings:
        try:
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)

            if path.suffix.lower() in [".txt", ".tsv"]:
                try:
                    return pd.read_csv(path, sep="\t", encoding=enc, on_bad_lines="skip", low_memory=False)
                except Exception:
                    return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")

        except Exception:
            continue

    raise ValueError("Could not read table file")


def detect_label_columns(df):
    candidates = []
    keywords = ["label", "class", "category", "target", "type", "result", "status"]

    for col in df.columns:
        col_l = str(col).lower()
        if any(k in col_l for k in keywords):
            candidates.append(col)

    return candidates


def detect_text_columns(df):
    text_cols = []

    for col in df.columns:
        try:
            if df[col].dtype == "object":
                avg_len = df[col].astype(str).str.len().mean()
                unique_ratio = df[col].nunique(dropna=True) / max(len(df), 1)

                if avg_len >= 20 or unique_ratio > 0.5:
                    text_cols.append(col)
        except Exception:
            pass

    return text_cols


def inspect_table(path):
    df = safe_read_table(path)

    rows, cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    missing_cells = int(df.isna().sum().sum())

    label_cols = detect_label_columns(df)
    text_cols = detect_text_columns(df)

    file_summary = {
        "relative_path": str(path.relative_to(BASE_DIR)),
        "file_name": path.name,
        "file_type": path.suffix.lower(),
        "size_mb": round(path.stat().st_size / (1024 * 1024), 4),
        "rows": rows,
        "columns": cols,
        "duplicate_rows": duplicate_rows,
        "missing_cells": missing_cells,
        "missing_percent": round((missing_cells / max(rows * cols, 1)) * 100, 4),
        "label_columns_detected": ", ".join(map(str, label_cols)),
        "text_columns_detected": ", ".join(map(str, text_cols)),
        "md5": file_hash(path),
        "status": "OK",
        "error": ""
    }

    column_rows = []
    for col in df.columns:
        s = df[col]
        column_rows.append({
            "file": str(path.relative_to(BASE_DIR)),
            "column": str(col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_percent": round((s.isna().sum() / max(len(s), 1)) * 100, 4),
            "unique_values": int(s.nunique(dropna=True)),
            "sample_values": " | ".join(s.dropna().astype(str).head(5).tolist())
        })

    label_rows = []
    for col in label_cols:
        counts = df[col].astype(str).value_counts(dropna=False)
        for label, count in counts.items():
            label_rows.append({
                "file": str(path.relative_to(BASE_DIR)),
                "label_column": str(col),
                "label": str(label),
                "count": int(count),
                "percent": round((count / max(rows, 1)) * 100, 4)
            })

    text_stats_rows = []
    for col in text_cols:
        lengths = df[col].dropna().astype(str).str.len()
        words = df[col].dropna().astype(str).str.split().str.len()

        if len(lengths) > 0:
            text_stats_rows.append({
                "file": str(path.relative_to(BASE_DIR)),
                "text_column": str(col),
                "avg_char_len": round(float(lengths.mean()), 2),
                "min_char_len": int(lengths.min()),
                "max_char_len": int(lengths.max()),
                "avg_word_count": round(float(words.mean()), 2),
                "min_word_count": int(words.min()),
                "max_word_count": int(words.max()),
                "empty_text_rows": int((df[col].astype(str).str.strip() == "").sum())
            })

    sample_rows = []
    sample_df = df.head(10).copy()
    sample_df.insert(0, "source_file", str(path.relative_to(BASE_DIR)))
    sample_rows = sample_df.astype(str).to_dict(orient="records")

    return file_summary, column_rows, label_rows, text_stats_rows, sample_rows


def inspect_image(path):
    try:
        with Image.open(path) as img:
            width, height = img.size
            mode = img.mode

        return {
            "relative_path": str(path.relative_to(BASE_DIR)),
            "file_name": path.name,
            "folder": str(path.parent.relative_to(BASE_DIR)),
            "file_type": path.suffix.lower(),
            "size_kb": round(path.stat().st_size / 1024, 4),
            "width": width,
            "height": height,
            "mode": mode,
            "md5": file_hash(path),
            "status": "OK",
            "error": ""
        }

    except Exception as e:
        return {
            "relative_path": str(path.relative_to(BASE_DIR)),
            "file_name": path.name,
            "folder": str(path.parent.relative_to(BASE_DIR)),
            "file_type": path.suffix.lower(),
            "size_kb": round(path.stat().st_size / 1024, 4),
            "width": "",
            "height": "",
            "mode": "",
            "md5": file_hash(path),
            "status": "ERROR",
            "error": str(e)
        }


def main():
    print(f"Scanning dataset folder: {BASE_DIR}")

    all_paths = [p for p in BASE_DIR.rglob("*") if p.is_file()]

    table_files = [p for p in all_paths if p.suffix.lower() in TEXT_EXTS]
    image_files = [p for p in all_paths if p.suffix.lower() in IMAGE_EXTS]
    other_files = [p for p in all_paths if p.suffix.lower() not in TEXT_EXTS + IMAGE_EXTS]

    folder_rows = []
    for folder, subdirs, files in os.walk(BASE_DIR):
        folder_path = Path(folder)
        folder_rows.append({
            "folder": str(folder_path.relative_to(BASE_DIR)),
            "num_files": len(files),
            "num_subfolders": len(subdirs),
            "total_size_mb": round(
                sum((folder_path / f).stat().st_size for f in files if (folder_path / f).exists()) / (1024 * 1024),
                4
            )
        })

    file_summary_rows = []
    column_rows = []
    label_rows = []
    text_stats_rows = []
    sample_rows = []

    print(f"Found table files: {len(table_files)}")
    for path in tqdm(table_files, desc="Inspecting CSV/TXT files"):
        try:
            fs, cr, lr, tr, sr = inspect_table(path)
            file_summary_rows.append(fs)
            column_rows.extend(cr)
            label_rows.extend(lr)
            text_stats_rows.extend(tr)
            sample_rows.extend(sr)

        except Exception as e:
            file_summary_rows.append({
                "relative_path": str(path.relative_to(BASE_DIR)),
                "file_name": path.name,
                "file_type": path.suffix.lower(),
                "size_mb": round(path.stat().st_size / (1024 * 1024), 4),
                "rows": "",
                "columns": "",
                "duplicate_rows": "",
                "missing_cells": "",
                "missing_percent": "",
                "label_columns_detected": "",
                "text_columns_detected": "",
                "md5": file_hash(path),
                "status": "ERROR",
                "error": str(e)
            })

    print(f"Found image files: {len(image_files)}")
    image_rows = []
    for path in tqdm(image_files, desc="Inspecting image files"):
        image_rows.append(inspect_image(path))

    image_folder_summary = []
    if image_rows:
        img_df = pd.DataFrame(image_rows)
        grouped = img_df.groupby("folder", dropna=False)

        for folder, g in grouped:
            labels_guess = ""
            folder_l = str(folder).lower()
            if "benign" in folder_l or "legit" in folder_l:
                labels_guess = "benign"
            elif "malicious" in folder_l or "phish" in folder_l:
                labels_guess = "malicious/phishing"

            image_folder_summary.append({
                "folder": folder,
                "count": len(g),
                "avg_size_kb": round(pd.to_numeric(g["size_kb"], errors="coerce").mean(), 4),
                "width_values": ", ".join(map(str, sorted(g["width"].dropna().astype(str).unique())[:10])),
                "height_values": ", ".join(map(str, sorted(g["height"].dropna().astype(str).unique())[:10])),
                "label_guess_from_folder": labels_guess
            })

    other_file_rows = []
    for p in other_files:
        other_file_rows.append({
            "relative_path": str(p.relative_to(BASE_DIR)),
            "file_name": p.name,
            "extension": p.suffix.lower(),
            "size_mb": round(p.stat().st_size / (1024 * 1024), 4),
            "md5": file_hash(p)
        })

    duplicate_file_rows = []
    hashes = Counter([file_hash(p) for p in all_paths])
    for p in all_paths:
        h = file_hash(p)
        if h and hashes[h] > 1:
            duplicate_file_rows.append({
                "relative_path": str(p.relative_to(BASE_DIR)),
                "file_name": p.name,
                "md5": h,
                "duplicate_count": hashes[h]
            })

    project_overview = [{
        "dataset_root": str(BASE_DIR),
        "total_files": len(all_paths),
        "table_files": len(table_files),
        "image_files": len(image_files),
        "other_files": len(other_files),
        "total_size_mb": round(sum(p.stat().st_size for p in all_paths) / (1024 * 1024), 4),
        "output_excel": str(EXCEL_OUT),
        "output_json": str(JSON_OUT)
    }]

    sheets = {
        "project_overview": pd.DataFrame(project_overview),
        "folder_summary": pd.DataFrame(folder_rows),
        "file_summary": pd.DataFrame(file_summary_rows),
        "columns": pd.DataFrame(column_rows),
        "label_balance": pd.DataFrame(label_rows),
        "text_statistics": pd.DataFrame(text_stats_rows),
        "sample_rows": pd.DataFrame(sample_rows),
        "image_files": pd.DataFrame(image_rows),
        "image_folder_summary": pd.DataFrame(image_folder_summary),
        "other_files": pd.DataFrame(other_file_rows),
        "duplicate_files": pd.DataFrame(duplicate_file_rows)
    }

    with pd.ExcelWriter(EXCEL_OUT, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            safe_name = sheet_name[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)

    json_data = {name: df.to_dict(orient="records") for name, df in sheets.items()}
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print("\nDONE")
    print(f"Excel summary saved at: {EXCEL_OUT}")
    print(f"JSON summary saved at:  {JSON_OUT}")


if __name__ == "__main__":
    main()