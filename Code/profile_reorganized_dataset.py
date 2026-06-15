from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


BASE_DIR = Path(r"D:\other\QTrustPhish\Dataset_Reorganized")
REPORT_DIR = BASE_DIR / "07_reports"
SPLIT_DIR = BASE_DIR / "08_splits"
FINAL_DIR = BASE_DIR / "06_final_ready"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = REPORT_DIR / "qtrustphish_reorganized_dataset_profile.xlsx"


def read_csv_safely(path):
    encodings = ["utf-8", "latin1", "cp1252", "ISO-8859-1"]

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, on_bad_lines="skip", low_memory=False)
        except Exception:
            continue

    raise ValueError(f"Could not read file: {path}")


def read_txt_safely(path):
    encodings = ["utf-8", "latin1", "cp1252", "ISO-8859-1"]

    for enc in encodings:
        try:
            return pd.read_csv(path, sep="\t", encoding=enc, on_bad_lines="skip", header=None)
        except Exception:
            try:
                return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip", header=None)
            except Exception:
                continue

    raise ValueError(f"Could not read file: {path}")


def load_table(path):
    if path.suffix.lower() == ".csv":
        return read_csv_safely(path)
    if path.suffix.lower() in [".txt", ".tsv"]:
        return read_txt_safely(path)
    raise ValueError(f"Unsupported table file: {path}")


def detect_label_columns(df):
    keywords = ["label", "class", "category", "target", "type", "status", "result"]
    cols = []

    for col in df.columns:
        c = str(col).lower()
        if any(k in c for k in keywords):
            cols.append(col)

    if not cols:
        for col in df.columns:
            try:
                unique_count = df[col].nunique(dropna=True)
                if 2 <= unique_count <= 10:
                    cols.append(col)
            except Exception:
                pass

    return cols


def detect_text_columns(df):
    cols = []

    for col in df.columns:
        try:
            s = df[col].dropna().astype(str)
            if len(s) == 0:
                continue

            avg_len = s.str.len().mean()
            max_len = s.str.len().max()

            if avg_len >= 20 or max_len >= 80:
                cols.append(col)

        except Exception:
            pass

    return cols


def label_balance_rows(df, label_cols, file_name):
    rows = []

    for col in label_cols:
        vc = df[col].astype(str).value_counts(dropna=False)
        total = len(df)

        for label, count in vc.items():
            rows.append({
                "file": file_name,
                "label_column": str(col),
                "label": str(label),
                "count": int(count),
                "percent": round((count / max(total, 1)) * 100, 4)
            })

    return rows


def dataset_inventory():
    table_files = list(BASE_DIR.rglob("*.csv")) + list(BASE_DIR.rglob("*.txt")) + list(BASE_DIR.rglob("*.tsv"))
    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
        image_files.extend((BASE_DIR / "04_qr").rglob(ext))

    rows = []
    label_rows = []

    for path in table_files:
        try:
            df = load_table(path)
            label_cols = detect_label_columns(df)

            rows.append({
                "relative_path": str(path.relative_to(BASE_DIR)),
                "file_name": path.name,
                "rows": len(df),
                "columns": len(df.columns),
                "duplicate_rows": int(df.duplicated().sum()),
                "missing_cells": int(df.isna().sum().sum()),
                "label_columns": ", ".join(map(str, label_cols)),
                "text_columns": ", ".join(map(str, detect_text_columns(df))),
                "status": "OK",
                "error": ""
            })

            label_rows.extend(label_balance_rows(df, label_cols, str(path.relative_to(BASE_DIR))))

        except Exception as e:
            rows.append({
                "relative_path": str(path.relative_to(BASE_DIR)),
                "file_name": path.name,
                "rows": "",
                "columns": "",
                "duplicate_rows": "",
                "missing_cells": "",
                "label_columns": "",
                "text_columns": "",
                "status": "ERROR",
                "error": str(e)
            })

    qr_counts = Counter()
    for p in image_files:
        parent = p.parent.name.lower()
        if "benign" in parent:
            qr_counts["qr_benign"] += 1
        elif "malicious" in parent:
            qr_counts["qr_malicious"] += 1
        else:
            qr_counts["qr_unknown"] += 1

    for k, v in qr_counts.items():
        rows.append({
            "relative_path": f"04_qr/{k}",
            "file_name": k,
            "rows": v,
            "columns": "image_files",
            "duplicate_rows": "",
            "missing_cells": "",
            "label_columns": "folder_name",
            "text_columns": "",
            "status": "OK",
            "error": ""
        })

        label_rows.append({
            "file": f"04_qr/{k}",
            "label_column": "folder_name",
            "label": k.replace("qr_", ""),
            "count": v,
            "percent": 100.0
        })

    return pd.DataFrame(rows), pd.DataFrame(label_rows)


def feature_analysis():
    path = BASE_DIR / "01_url" / "url_phishing_11430_89features.csv"
    df = read_csv_safely(path)

    label_cols = detect_label_columns(df)
    label_col = label_cols[0] if label_cols else ""

    rows = []

    for col in df.columns:
        s = df[col]

        row = {
            "feature_name": str(col),
            "is_label_column": str(col) == str(label_col),
            "dtype": str(s.dtype),
            "missing": int(s.isna().sum()),
            "missing_percent": round((s.isna().sum() / max(len(s), 1)) * 100, 4),
            "unique_values": int(s.nunique(dropna=True)),
            "sample_values": " | ".join(s.dropna().astype(str).head(5).tolist())
        }

        if pd.api.types.is_numeric_dtype(s):
            row.update({
                "min": float(s.min()) if s.notna().any() else "",
                "max": float(s.max()) if s.notna().any() else "",
                "mean": float(s.mean()) if s.notna().any() else "",
                "std": float(s.std()) if s.notna().any() else ""
            })
        else:
            row.update({
                "min": "",
                "max": "",
                "mean": "",
                "std": ""
            })

        rows.append(row)

    return pd.DataFrame(rows)


def email_analysis():
    email_root = BASE_DIR / "03_email_human_llm"
    csv_files = list(email_root.rglob("*.csv"))

    rows = []
    sample_rows = []

    for path in csv_files:
        try:
            df = read_csv_safely(path)
            text_cols = detect_text_columns(df)
            label_cols = detect_label_columns(df)

            folder_parts = path.relative_to(email_root).parts
            source_group = folder_parts[0] if len(folder_parts) > 1 else "unknown"

            inferred_author = "llm" if "llm" in source_group.lower() else "human"
            inferred_label = "phishing" if "phishing" in source_group.lower() else "legit"

            for col in text_cols:
                s = df[col].dropna().astype(str)
                rows.append({
                    "file": str(path.relative_to(BASE_DIR)),
                    "source_group": source_group,
                    "inferred_author": inferred_author,
                    "inferred_label": inferred_label,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "detected_text_column": str(col),
                    "detected_label_columns": ", ".join(map(str, label_cols)),
                    "avg_char_length": round(float(s.str.len().mean()), 2) if len(s) else 0,
                    "max_char_length": int(s.str.len().max()) if len(s) else 0,
                    "avg_word_count": round(float(s.str.split().str.len().mean()), 2) if len(s) else 0,
                    "max_word_count": int(s.str.split().str.len().max()) if len(s) else 0,
                    "missing_text": int(df[col].isna().sum()),
                    "duplicates": int(df.duplicated().sum())
                })

            temp = df.head(3).copy()
            temp.insert(0, "source_file", str(path.relative_to(BASE_DIR)))
            sample_rows.extend(temp.astype(str).to_dict(orient="records"))

        except Exception as e:
            rows.append({
                "file": str(path.relative_to(BASE_DIR)),
                "source_group": "",
                "inferred_author": "",
                "inferred_label": "",
                "rows": "",
                "columns": "",
                "detected_text_column": "",
                "detected_label_columns": "",
                "avg_char_length": "",
                "max_char_length": "",
                "avg_word_count": "",
                "max_word_count": "",
                "missing_text": "",
                "duplicates": "",
                "error": str(e)
            })

    return pd.DataFrame(rows), pd.DataFrame(sample_rows)


def sms_analysis():
    sms_root = BASE_DIR / "02_sms"
    files = list(sms_root.rglob("*.csv")) + list(sms_root.rglob("*.txt"))

    rows = []
    label_rows = []
    sample_rows = []

    for path in files:
        try:
            df = load_table(path)

            if path.suffix.lower() == ".txt":
                if len(df.columns) >= 2:
                    df.columns = ["label", "text"] + [f"extra_{i}" for i in range(2, len(df.columns))]

            text_cols = detect_text_columns(df)
            label_cols = detect_label_columns(df)

            for col in text_cols:
                s = df[col].dropna().astype(str)
                rows.append({
                    "file": str(path.relative_to(BASE_DIR)),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "detected_text_column": str(col),
                    "detected_label_columns": ", ".join(map(str, label_cols)),
                    "avg_char_length": round(float(s.str.len().mean()), 2) if len(s) else 0,
                    "max_char_length": int(s.str.len().max()) if len(s) else 0,
                    "avg_word_count": round(float(s.str.split().str.len().mean()), 2) if len(s) else 0,
                    "max_word_count": int(s.str.split().str.len().max()) if len(s) else 0,
                    "missing_text": int(df[col].isna().sum()),
                    "duplicates": int(df.duplicated().sum())
                })

            label_rows.extend(label_balance_rows(df, label_cols, str(path.relative_to(BASE_DIR))))

            temp = df.head(3).copy()
            temp.insert(0, "source_file", str(path.relative_to(BASE_DIR)))
            sample_rows.extend(temp.astype(str).to_dict(orient="records"))

        except Exception as e:
            rows.append({
                "file": str(path.relative_to(BASE_DIR)),
                "rows": "",
                "columns": "",
                "detected_text_column": "",
                "detected_label_columns": "",
                "avg_char_length": "",
                "max_char_length": "",
                "avg_word_count": "",
                "max_word_count": "",
                "missing_text": "",
                "duplicates": "",
                "error": str(e)
            })

    return pd.DataFrame(rows), pd.DataFrame(label_rows), pd.DataFrame(sample_rows)


def qr_analysis(max_images_per_class=5000):
    qr_root = BASE_DIR / "04_qr"

    rows = []
    detail_rows = []

    for class_folder in ["benign", "malicious"]:
        folder = qr_root / class_folder
        image_files = []

        for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
            image_files.extend(folder.glob(ext))

        widths = []
        heights = []
        modes = []
        channels = []
        sizes_kb = []

        sample_files = image_files[:max_images_per_class]

        for path in tqdm(sample_files, desc=f"Analyzing QR {class_folder}"):
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    mode = img.mode

                ch = {"1": 1, "L": 1, "P": 1, "RGB": 3, "RGBA": 4, "CMYK": 4}.get(mode, "")

                widths.append(w)
                heights.append(h)
                modes.append(mode)
                channels.append(ch)
                sizes_kb.append(path.stat().st_size / 1024)

                detail_rows.append({
                    "file": str(path.relative_to(BASE_DIR)),
                    "label": class_folder,
                    "width": w,
                    "height": h,
                    "mode": mode,
                    "channels": ch,
                    "size_kb": round(path.stat().st_size / 1024, 4)
                })

            except Exception as e:
                detail_rows.append({
                    "file": str(path.relative_to(BASE_DIR)),
                    "label": class_folder,
                    "width": "",
                    "height": "",
                    "mode": "",
                    "channels": "",
                    "size_kb": "",
                    "error": str(e)
                })

        rows.append({
            "class": class_folder,
            "image_count": len(image_files),
            "sampled_for_dimension_check": len(sample_files),
            "min_width": min(widths) if widths else "",
            "max_width": max(widths) if widths else "",
            "most_common_width": Counter(widths).most_common(1)[0][0] if widths else "",
            "min_height": min(heights) if heights else "",
            "max_height": max(heights) if heights else "",
            "most_common_height": Counter(heights).most_common(1)[0][0] if heights else "",
            "modes": ", ".join(sorted(set(map(str, modes)))),
            "channels": ", ".join(sorted(set(map(str, channels)))),
            "avg_size_kb": round(float(np.mean(sizes_kb)), 4) if sizes_kb else ""
        })

    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def main():
    print("Profiling reorganized QTrustPhish dataset...")
    print(f"Base folder: {BASE_DIR}")

    inventory_df, inventory_labels_df = dataset_inventory()
    feature_df = feature_analysis()
    email_df, email_samples_df = email_analysis()
    sms_df, sms_labels_df, sms_samples_df = sms_analysis()
    qr_df, qr_details_df = qr_analysis(max_images_per_class=5000)

    with pd.ExcelWriter(OUT_FILE, engine="openpyxl") as writer:
        inventory_df.to_excel(writer, sheet_name="dataset_inventory", index=False)
        inventory_labels_df.to_excel(writer, sheet_name="inventory_class_balance", index=False)
        feature_df.to_excel(writer, sheet_name="feature_analysis", index=False)
        email_df.to_excel(writer, sheet_name="email_analysis", index=False)
        sms_df.to_excel(writer, sheet_name="sms_analysis", index=False)
        sms_labels_df.to_excel(writer, sheet_name="sms_class_balance", index=False)
        qr_df.to_excel(writer, sheet_name="qr_analysis", index=False)
        qr_details_df.head(20000).to_excel(writer, sheet_name="qr_image_samples", index=False)
        email_samples_df.to_excel(writer, sheet_name="email_samples", index=False)
        sms_samples_df.to_excel(writer, sheet_name="sms_samples", index=False)

    print("\nDONE")
    print(f"Saved profile file:")
    print(OUT_FILE)


if __name__ == "__main__":
    main()