# QTrustAgentX

> Explainable Multi-Channel Phishing Detection Through Agentic Decision Arbitration, Quantum-Inspired Representation Learning, and Trust Graph Modeling

Official repository containing the source code, reorganized datasets, trained models, figures, and reproducible experiments.

## Repository Structure

```text
QTrustAgentX/
├── Code/
├── Dataset/
├── Results/
├── qtrustagentx_architecture_1080p.png
├── LICENSE
└── README.md
```

# Architecture

1. Input Orchestration Agent
2. URL Analysis Agent
3. Email Analysis Agent
4. SMS Analysis Agent
5. QR Analysis Agent
6. Semantic Compression Agent
7. Quantum Feature Encoding Agent
8. Trust Graph Reasoning Agent
9. Risk Arbitration and Explanation Agent

# Novel Contributions

- Multi-channel phishing detection across URL, Email, SMS, and QR modalities
- Agentic decision arbitration
- Quantum-inspired representation learning
- Trust graph reasoning
- Explainable modality contribution scoring
- Robustness evaluation under poisoned evidence
- Human versus LLM phishing generalization analysis

# Dataset Summary

| Dataset | Rows | Columns | Duplicates | Missing Cells | Label Column | Text Column |
|---------|------:|---------:|------------:|---------------:|--------------|-------------|
| url_phishing_11430_89features.csv | 11,430 | 89 | 0 | 0 | status | url |
| sms_phishing_5971.csv | 5,971 | 5 | 17 | 0 | LABEL | TEXT |
| human_legit_email_1000.csv | 1,000 | 7 | 0 | 16 | label | body |
| human_phishing_email_1000.csv | 1,000 | 7 | 496 | 24 | label | body |
| llm_legit_email_1000.csv | 1,000 | 2 | 2 | 0 | label | text |
| llm_phishing_email_595.csv | 595 | 2 | 429 | 631 | label | text |
| dataset_manifest.csv | 10 | 2 | 0 | 0 | dataset_type | path |
| phishing_intent_category_1000.csv | 1,000 | 3 | 921 | 0 | category | text |
| sms_spam_raw_duplicate_check.csv | 5,572 | 5 | 403 | 16,648 | v1 | v2 |
| dataset_links.txt | 8 | 1 | 0 | 0 | N/A | N/A |

QR corpus:
- 100,000 benign QR images
- 100,000 malicious QR images

# Conda Setup

```bash
conda create -n qtrustagentx python=3.11 -y
conda activate qtrustagentx

pip install pandas numpy scipy scikit-learn matplotlib seaborn
pip install pillow tqdm openpyxl joblib networkx shap lime
pip install torch torchvision torchaudio
```

# Running the Code

```bash
python reorganize_dataset.py
python profile_reorganized_dataset.py
python deep_dataset_inspector.py
python inspect_datasets.py
python qtrustagentx_final_pipeline.py --mode all --qr_limit 20000 --qr_epochs 10
```

# Generated Results

Running the full pipeline generates:

- Dataset profiles
- Train, validation, and test splits
- Trained models
- CSV result tables
- ROC curves
- Confusion matrices
- Ablation studies
- Explainability reports
- Figures for the paper

# Main Results

| Experiment | Best F1 |
|------------|---------:|
| QR Specialist | 0.9995 |
| Email Compression | 0.9925 |
| URL Detection | 0.9621 |
| Quantum Arbitration | 0.7727 |
| Majority Voting | 0.7360 |

# Ablation Studies

A. Agentic Orchestration  
B. Quantum and Graph Reasoning  
C. Modality Contribution  
D. Human vs LLM Generalization  
E. Robustness to Poisoned Evidence  
F. Explainability Faithfulness

# Figures Included

- QTrustAgentX architecture diagram
- Agent evidence analysis
- ROC curves
- Confusion matrices
- Modality contribution plots
- Robustness analysis figures
- Explainability figures

# Citation

```bibtex
@article{khan2025qtrustagentx,
  title={QTrustAgent-X: Explainable Multi-Channel Phishing Detection Through Agentic Decision Arbitration, Quantum-Inspired Representation Learning, and Trust Graph Modeling},
  author={Khan, Misha Urooj and Suleman, Ahmad and Adarbah, Haitham},
  journal={IEEE Open Journal of the Computer Society},
  year={2025}
}
```

# License

Apache License 2.0
