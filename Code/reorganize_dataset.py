from pathlib import Path
import shutil
import os
import pandas as pd

SOURCE = Path(r"D:\other\QTrustPhish\Dataset")
TARGET = Path(r"D:\other\QTrustPhish\Dataset_Reorganized")

DRY_RUN = False
USE_HARDLINKS_FOR_QR = True


def make_dir(path):
    if not DRY_RUN:
        path.mkdir(parents=True, exist_ok=True)


def copy_file(src, dst):
    make_dir(dst.parent)

    if DRY_RUN:
        print(f"[DRY RUN] COPY FILE: {src} -> {dst}")
        return

    shutil.copy2(src, dst)
    print(f"COPIED FILE: {dst}")


def link_or_copy_image(src, dst):
    make_dir(dst.parent)

    if DRY_RUN:
        print(f"[DRY RUN] IMAGE: {src} -> {dst}")
        return

    if dst.exists():
        return

    try:
        if USE_HARDLINKS_FOR_QR:
            os.link(src, dst)
        else:
            shutil.copy2(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def copy_folder_images(src_folder, dst_folder, prefix):
    make_dir(dst_folder)

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
        image_files.extend(src_folder.glob(ext))

    print(f"Found {len(image_files)} images in {src_folder}")

    for i, src in enumerate(image_files):
        new_name = f"{prefix}_{i:06d}{src.suffix.lower()}"
        dst = dst_folder / new_name
        link_or_copy_image(src, dst)

    print(f"Finished: {dst_folder}")


def main():
    print("Starting dataset reorganization...")

    folders = [
        "00_sources",
        "01_url",
        "02_sms",
        "03_email_human_llm/human_legit",
        "03_email_human_llm/human_phishing",
        "03_email_human_llm/llm_legit",
        "03_email_human_llm/llm_phishing",
        "04_qr/benign",
        "04_qr/malicious",
        "05_metadata",
        "06_final_ready",
    ]

    for f in folders:
        make_dir(TARGET / f)

    mapping = [
        ("dataset_phishing.csv", "01_url/url_phishing_11430_89features.csv"),
        ("Dataset_5971.csv", "02_sms/sms_phishing_5971.csv"),
        ("SMSSmishCollection.txt", "02_sms/sms_smishing_5571.txt"),
        ("spam.csv", "02_sms/sms_spam_raw_duplicate_check.csv"),
        ("phishing_dataset_with_category.csv", "05_metadata/phishing_intent_category_1000.csv"),
        ("dataset-links.txt", "00_sources/dataset_links.txt"),
        ("readme", "00_sources/readme"),
        (
            r"Human-LLM generated phishing-legitimate emails\human-generated\legit.csv",
            "03_email_human_llm/human_legit/human_legit_email_1000.csv"
        ),
        (
            r"Human-LLM generated phishing-legitimate emails\human-generated\phishing.csv",
            "03_email_human_llm/human_phishing/human_phishing_email_1000.csv"
        ),
        (
            r"Human-LLM generated phishing-legitimate emails\llm-generated\legit.csv",
            "03_email_human_llm/llm_legit/llm_legit_email_1000.csv"
        ),
        (
            r"Human-LLM generated phishing-legitimate emails\llm-generated\phishing.csv",
            "03_email_human_llm/llm_phishing/llm_phishing_email_595.csv"
        ),
    ]

    for src_rel, dst_rel in mapping:
        src = SOURCE / src_rel
        dst = TARGET / dst_rel

        if src.exists():
            copy_file(src, dst)
        else:
            print(f"WARNING: Missing file: {src}")

    qr_benign_src = SOURCE / "QR codes" / "benign"
    qr_malicious_src = SOURCE / "QR codes" / "malicious"

    if qr_benign_src.exists():
        copy_folder_images(
            qr_benign_src,
            TARGET / "04_qr" / "benign",
            "qr_benign"
        )

    if qr_malicious_src.exists():
        copy_folder_images(
            qr_malicious_src,
            TARGET / "04_qr" / "malicious",
            "qr_malicious"
        )

    manifest = {
        "url": "01_url/url_phishing_11430_89features.csv",
        "sms_main": "02_sms/sms_phishing_5971.csv",
        "sms_smishing": "02_sms/sms_smishing_5571.txt",
        "sms_spam_duplicate_check": "02_sms/sms_spam_raw_duplicate_check.csv",
        "human_legit_email": "03_email_human_llm/human_legit/human_legit_email_1000.csv",
        "human_phishing_email": "03_email_human_llm/human_phishing/human_phishing_email_1000.csv",
        "llm_legit_email": "03_email_human_llm/llm_legit/llm_legit_email_1000.csv",
        "llm_phishing_email": "03_email_human_llm/llm_phishing/llm_phishing_email_595.csv",
        "qr_benign": "04_qr/benign",
        "qr_malicious": "04_qr/malicious",
    }

    manifest_df = pd.DataFrame(
        [{"dataset_type": k, "path": v} for k, v in manifest.items()]
    )

    manifest_path = TARGET / "05_metadata" / "dataset_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)

    print("\nDONE")
    print(f"Reorganized dataset saved at: {TARGET}")
    print(f"Manifest saved at: {manifest_path}")


if __name__ == "__main__":
    main()