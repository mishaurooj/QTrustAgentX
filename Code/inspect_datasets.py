import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"D:\other\QTrustPhish\Dataset")

summary_rows = []
column_rows = []
label_rows = []

def load_file(file_path):

    encodings = ["utf-8", "latin1", "cp1252"]

    for enc in encodings:

        try:

            if file_path.suffix.lower() == ".csv":

                return pd.read_csv(
                    file_path,
                    encoding=enc,
                    on_bad_lines="skip",
                    low_memory=False
                )

            elif file_path.suffix.lower() == ".txt":

                try:
                    return pd.read_csv(
                        file_path,
                        sep="\t",
                        encoding=enc,
                        on_bad_lines="skip",
                        low_memory=False
                    )
                except:
                    return pd.read_csv(
                        file_path,
                        sep=None,
                        engine="python",
                        encoding=enc,
                        on_bad_lines="skip"
                    )

        except:
            continue

    raise Exception("Unable to read file")


all_files = list(BASE_DIR.rglob("*.csv")) + list(BASE_DIR.rglob("*.txt"))

print(f"\nFound {len(all_files)} files\n")

for file in all_files:

    print(f"Processing: {file.name}")

    try:

        df = load_file(file)

        rows, cols = df.shape

        duplicate_rows = int(df.duplicated().sum())

        missing_cells = int(df.isna().sum().sum())

        summary_rows.append({
            "file": str(file.relative_to(BASE_DIR)),
            "rows": rows,
            "columns": cols,
            "duplicates": duplicate_rows,
            "missing_cells": missing_cells
        })

        for col in df.columns:

            column_rows.append({
                "file": str(file.relative_to(BASE_DIR)),
                "column": str(col),
                "dtype": str(df[col].dtype),
                "missing": int(df[col].isna().sum()),
                "unique_values": int(df[col].nunique(dropna=True))
            })

        candidate_labels = []

        for col in df.columns:

            c = str(col).lower()

            if any(x in c for x in [
                "label",
                "class",
                "category",
                "target",
                "spam",
                "type"
            ]):
                candidate_labels.append(col)

        for col in candidate_labels:

            vc = df[col].astype(str).value_counts()

            for label, count in vc.items():

                label_rows.append({
                    "file": str(file.relative_to(BASE_DIR)),
                    "label_column": str(col),
                    "label": label,
                    "count": int(count)
                })

        print(f"  OK -> {rows} rows, {cols} columns")

    except Exception as e:

        summary_rows.append({
            "file": str(file.relative_to(BASE_DIR)),
            "rows": "ERROR",
            "columns": "ERROR",
            "duplicates": "",
            "missing_cells": "",
            "error": str(e)
        })

        print(f"  ERROR -> {e}")

summary_df = pd.DataFrame(summary_rows)
columns_df = pd.DataFrame(column_rows)
labels_df = pd.DataFrame(label_rows)

output_file = BASE_DIR / "dataset_inspection_summary_v3.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:

    summary_df.to_excel(
        writer,
        sheet_name="file_summary",
        index=False
    )

    columns_df.to_excel(
        writer,
        sheet_name="columns",
        index=False
    )

    labels_df.to_excel(
        writer,
        sheet_name="label_balance",
        index=False
    )

print("\n================================")
print("FINISHED")
print(f"Saved: {output_file}")
print("================================")