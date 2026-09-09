# TrustAgent-X

## Reliability-Calibrated Quantum Evidence Arbitration for Heterogeneous Phishing Detection with LLM Domain Transfer

TrustAgent-X is a heterogeneous phishing-detection framework that combines modality-specific URL, email, SMS, and QR specialists through a common probability-and-mask interface. The proposed RC-QTG arbiter uses training-derived reliability to scale a four-qubit Qiskit feature transformation, followed by classical probability arbitration. The evaluation uses strict nested cross-fitting, matched classical and quantum controls, robustness tests, component ablations, and Human-to-LLM / LLM-to-Human email transfer experiments.

## Proposed Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/mishaurooj/QTrustAgentX/main/TrustAgentX_architecture.jpg"
       alt="TrustAgent-X proposed architecture"
       width="100%">
</p>


## Main Contributions

- A strict nested specialist-to-arbiter evaluation protocol that prevents outer-test information from entering specialist fitting, preprocessing, reliability estimation, or arbiter training.
- A common score-and-mask evidence representation for URL, email, SMS, and QR phishing specialists.
- RC-QTG, a reliability-calibrated four-qubit Qiskit Statevector feature transformation using training-derived Brier-skill reliability.
- Matched controls covering masked averaging, majority voting, learned late fusion, TrustGraphArbitration, Random Fourier Features, QTG-local, and QTG-pair variants.
- Component ablations that separate reliability-scaled circuit encoding, post-quantum reliability features, and the active-score skip connection.
- Robustness tests under score corruption and source removal, together with Human-to-LLM and LLM-to-Human email transfer evaluation.

## Model and Ablation Variants
<p align="center">
  <img src="https://raw.githubusercontent.com/mishaurooj/QTrustAgentX/main/FINAL_V0_V9C_blue_white_clean_no_overlap.jpg"
       alt="TrustAgent-X V0-V9C model and ablation variants"
       width="100%">
</p>
## Evaluation Scope

The reported quantum experiments use Qiskit Statevector simulation. In the DISJOINT evaluation corpus, each empirical row contains one observed modality. The results support reliability-aware quantum evidence arbitration, but do not establish quantum advantage or an empirical entanglement benefit.

## License

Apache-2.0.

