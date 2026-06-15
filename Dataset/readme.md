# Phishing Detection Dataset Collection

This repository contains a curated collection of phishing-related datasets from Kaggle. The datasets cover SMS smishing, spam detection, phishing emails, LLM-generated phishing text, malicious QR codes, URL phishing, and phishing persuasion techniques.

## Dataset Sources

| # | Dataset                                              | Link                                                                                            |
| - | ---------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1 | SMS Smishing Collection Dataset                      | https://www.kaggle.com/datasets/galactus007/sms-smishing-collection-data-set                    |
| 2 | SMS Spam Collection Dataset                          | https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset                               |
| 3 | SMS Phishing Dataset                                 | https://www.kaggle.com/datasets/fadlifatih/sms-phishing-dataset                                 |
| 4 | Phishing Urgency, Authority, and Persuasion Dataset  | https://www.kaggle.com/datasets/ahmadtijjani/phishing-urgency-authority-persuasion              |
| 5 | Web Page Phishing Detection Dataset                  | https://www.kaggle.com/datasets/shashwatwork/web-page-phishing-detection-dataset                |
| 6 | Benign and Malicious QR Codes Dataset                | https://www.kaggle.com/datasets/samahsadiq/benign-and-malicious-qr-codes                        |
| 7 | Human and LLM-Generated Phishing / Legitimate Emails | https://www.kaggle.com/datasets/francescogreco97/human-llm-generated-phishing-legitimate-emails |

## Citation

The Human and LLM-Generated Phishing / Legitimate Emails dataset was used in the ITASEC 2024 cybersecurity paper:

> **David versus Goliath: Can Machine Learning Detect LLM-Generated Text? A Case Study in the Detection of Phishing Emails**
> F. Greco, G. Desolda, A. Esposito, A. Carelli, 2024.

## Dataset Summary

| File                                                                 |   Rows | Columns | Duplicates | Missing Cells | Label Column   | Text Column |
| -------------------------------------------------------------------- | -----: | ------: | ---------: | ------------: | -------------- | ----------- |
| 🟦 `01_url/url_phishing_11430_89features.csv`                        | 11,430 |      89 |          0 |             0 | `status`       | `url`       |
| 🟩 `02_sms/sms_phishing_5971.csv`                                    |  5,971 |       5 |         17 |             0 | `LABEL`        | `TEXT`      |
| 🟨 `03_email_human_llm/human_legit/human_legit_email_1000.csv`       |  1,000 |       7 |          0 |            16 | `label`        | `body`      |
| 🟧 `03_email_human_llm/human_phishing/human_phishing_email_1000.csv` |  1,000 |       7 |        496 |            24 | `label`        | `body`      |
| 🟪 `03_email_human_llm/llm_legit/llm_legit_email_1000.csv`           |  1,000 |       2 |          2 |             0 | `label`        | `text`      |
| 🟥 `03_email_human_llm/llm_phishing/llm_phishing_email_595.csv`      |    595 |       2 |        429 |           631 | `label`        | `text`      |
| ⬛ `05_metadata/dataset_manifest.csv`                                 |     10 |       2 |          0 |             0 | `dataset_type` | `path`      |
| 🟫 `05_metadata/phishing_intent_category_1000.csv`                   |  1,000 |       3 |        921 |             0 | `category`     | `text`      |
| 🟦 `05_metadata/sms_spam_raw_duplicate_check.csv`                    |  5,572 |       5 |        403 |        16,648 | `v1`           | `v2`        |
| ⬜ `00_sources/dataset_links.txt`                                     |      8 |       1 |          0 |             0 | `0`            | `0`         |

## Notes

* The datasets include URL, SMS, email, QR code, and metadata-based phishing resources.
* Some files contain duplicate rows and missing values, especially the LLM phishing and metadata files.
* Duplicate and missing-value handling should be performed before model training.
* The `dataset_manifest.csv` file can be used to track dataset categories and file paths.
