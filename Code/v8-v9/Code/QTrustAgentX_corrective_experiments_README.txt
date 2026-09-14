QTrustAgent-X corrective experiment package
==========================================

Main script:
  qtrustagentx_corrective_experiments.py

Windows launcher:
  RUN_QTRUSTAGENTX_CORRECTIVE_WINDOWS.bat

WHAT IT RUNS
------------
A. Stronger Human<->LLM transfer
   - word TF-IDF SVM
   - character TF-IDF SVM
   - hybrid word+character SVM
   - source-validation DeBERTa ensemble
   - source-calibration Brier fusion
   - source-calibration stacked fusion
   - threshold/model selection is SOURCE ONLY
   - target labels are evaluation only

B. Genuine multimodal graph-arbitration validation
   - requires INCIDENT evidence with >=2 observed modalities per row
   - refuses synthetic cross-dataset joining
   - MaskedAverage
   - LateFusion
   - TGA SelfOnly
   - TGA FullGraph
   - QTG Local
   - QTG Pair
   - pair-activation and local-vs-pair feature audits
   - SAVES exact QTG Local and QTG Pair circuit artifacts

C. 2x2 reliability corrective factorial
   - one ParameterizedRCQTG circuit implementation
   - phase coefficient: 0.5 vs 1.0
   - reliability: uniform vs training-estimated
   - lambda fixed at 0.35
   - 20 quantum observables in every cell
   - no appended reliability summaries in any cell
   - same specialist evidence, split, readout, classifier
   - SAVES circuit artifacts for every factorial cell

QUANTUM CIRCUIT OUTPUTS
-----------------------
The corrective script now saves the quantum circuits themselves.

Publication figures:
  figures/quantum_circuits/*.png   (600 dpi)
  figures/quantum_circuits/*.jpg   (450 dpi)
  figures/quantum_circuits/*.pdf
  figures/quantum_circuits/*.svg

Exact audit representations:
  audit/quantum_circuits/*.txt
  audit/quantum_circuits/*.qasm
  audit/quantum_circuits/*.json

Multimodal experiment:
  - QTG Local on a genuine incident-aligned multimodal test row
  - QTG Pair on the same genuine multimodal test row
  - QTG Pair all-channel design template, explicitly marked as visualization only
  - audit/quantum_circuits/multimodal_circuit_manifest_seed<SEED>.json

Reliability x phase factorial:
  For every cell, the script saves:
    - the actual DISJOINT empirical-row circuit
    - an all-channel design-template circuit for topology visualization only
  - audit/quantum_circuits/reliability_factorial_circuit_manifest_seed<SEED>.json

The all-channel templates are NEVER used for performance evaluation. They exist only
so a paper figure can display the complete pair-capable four-qubit topology. The JSON
metadata records whether each circuit is empirical or design-template only.

PAPER-READY OUTPUTS
-------------------
  tables/human_llm_corrective_transfer.tex
  tables/multimodal_graph_validation.tex
  tables/reliability_phase_factorial_seed42.tex
  tables/QTrustAgentX_corrective_tables_all.tex

CSV results and predictions are saved under results/ and predictions/.
Audit files are saved under audit/.
Figures are saved under figures/.

IMPORTANT SCIENTIFIC BOUNDARIES
-------------------------------
1. The code does not guarantee improved LLM transfer metrics. It strengthens the
   protocol while preventing target-domain model selection.
2. A multimodal interaction claim is produced only when genuine incident-aligned
   evidence exists. The script will not fabricate alignments from the six source datasets.
3. Use --full_qiskit_factorial if you want every factorial row evaluated directly
   through Qiskit Statevector. The default DISJOINT fast path is exact and first
   validates its features against Qiskit on a sample.
4. Circuit design-template figures do not constitute empirical multimodal evidence.
