
"""
QTrustAgentX Reviewer 3: expanded Qiskit ablation benchmark.

Requires the ORIGINAL reviewer3_qiskit.py in the same directory.
Uses its verified dataset loader and modality specialists.

Quantum configurations:
    4 interaction graphs x 3 circuit depths x 2 readout sets
    = 24 quantum variants.

Classical controls:
    Logistic regression, RBF SVM, ExtraTrees, HistGB,
    Random Fourier Features, polynomial features, MLP.

All comparisons use the same source-level train/test indices.
Circuit parameters are predefined, not optimized on test labels.
The benchmark runs on Qiskit's classical statevector simulator.
"""

import argparse
import importlib.util
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector, partial_trace, entropy

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    StandardScaler, PolynomialFeatures
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)
from sklearn.kernel_approximation import RBFSampler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    brier_score_loss
)

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "reviewer3_qiskit.py"

if not ORIGINAL.exists():
    raise FileNotFoundError(
        "Put reviewer3_qiskit_v2.py beside reviewer3_qiskit.py"
    )

spec = importlib.util.spec_from_file_location(
    "original_reviewer3", ORIGINAL
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

OUTPUT = (
    Path(r"E:\other\QTrustAgentX")
    / "Reviewer3_24_Quantum_Variants"
)
OUTPUT.mkdir(parents=True, exist_ok=True)

N = 6
GRAPHS = ("ring", "chain", "star", "dense")
DEPTHS = (1, 2, 3)
READOUTS = ("local", "pair")


def graph_edges(graph):
    if graph == "ring":
        return [(i, (i + 1) % N) for i in range(N)]
    if graph == "chain":
        return [(i, i + 1) for i in range(N - 1)]
    if graph == "star":
        return [(0, i) for i in range(1, N)]
    if graph == "dense":
        return list(itertools.combinations(range(N), 2))
    raise ValueError(graph)


def circuit(angles, graph, depth, entangled=True):
    """Actual six-qubit Qiskit circuit."""
    qc = QuantumCircuit(N)
    angles = np.asarray(angles, dtype=float)

    qc.h(range(N))

    edges = graph_edges(graph)

    for layer in range(depth):
        for i in range(N):
            theta = float(angles[i])
            qc.ry(
                (1.0 + layer * 0.19) * theta, i
            )
            qc.rz(
                0.45 * np.sin(theta + layer * 0.31), i
            )

        if entangled:
            for i, j in edges:
                phase = (
                    0.25
                    + 0.70
                    * np.sin(angles[i] + 0.13 * layer)
                    * np.cos(angles[j] - 0.17 * layer)
                )
                qc.cx(i, j)
                qc.rz(float(phase), j)
                qc.cx(i, j)

        for i in range(N):
            qc.rx(
                float(
                    0.38
                    * np.sin(angles[i] + 0.4 * layer)
                ),
                i
            )

    return qc


def local_pauli_readout(state):
    """Returns real X, Y, Z expectations, 18 features."""
    psi = np.asarray(state, dtype=np.complex128)
    idx = np.arange(1 << N)
    probs = np.abs(psi) ** 2
    out = []

    for i in range(N):
        bit = (idx >> i) & 1
        signs = 1.0 - 2.0 * bit
        flipped = psi[idx ^ (1 << i)]

        z = float(probs @ signs)
        x = float(np.real(np.vdot(psi, flipped)))

        # Y|0> = i|1>, Y|1> = -i|0>.
        y = float(
            np.real(
                np.vdot(
                    psi,
                    -1j * signs * flipped
                )
            )
        )
        out.extend((x, y, z))

    return np.asarray(out, dtype=float)


def pair_pauli_readout(state):
    """Local X/Y/Z and all pairwise XX, ZZ: 48 features."""
    psi = np.asarray(state, dtype=np.complex128)
    idx = np.arange(1 << N)
    prob = np.abs(psi) ** 2
    local = local_pauli_readout(psi).tolist()

    for i, j in itertools.combinations(range(N), 2):
        bit_i = 1 - 2 * ((idx >> i) & 1)
        bit_j = 1 - 2 * ((idx >> j) & 1)

        zz = float(prob @ (bit_i * bit_j))
        flipped = psi[idx ^ (1 << i) ^ (1 << j)]
        xx = float(np.real(np.vdot(psi, flipped)))
        local.extend((xx, zz))

    return np.asarray(local, dtype=float)


def verify_readout():
    """Checks direct amplitude readout against Qiskit operators."""
    from qiskit.quantum_info import SparsePauliOp

    rng = np.random.default_rng(2026)
    sample = rng.uniform(-2, 2, size=N)
    sv = Statevector.from_instruction(
        circuit(sample, "ring", 2)
    )
    local = local_pauli_readout(sv.data)

    for qubit in range(N):
        for k, axis in enumerate(("X", "Y", "Z")):
            label = ["I"] * N
            label[N - 1 - qubit] = axis
            op = SparsePauliOp("".join(label))
            expected = float(
                np.real(sv.expectation_value(op))
            )
            actual = local[3 * qubit + k]
            if not np.isclose(
                actual, expected, atol=1e-9
            ):
                raise AssertionError(
                    f"Readout error: {axis}{qubit}, "
                    f"{actual} != {expected}"
                )

    print("Verified X/Y/Z readouts against Qiskit.")


def produce_features(
    angles, graph, depth, readout, cache_path
):
    """Cache computed Qiskit statevector features."""
    if cache_path.exists():
        with np.load(cache_path) as saved:
            features = saved["features"]
        if len(features) == len(angles):
            print("Loaded cached:", cache_path.name)
            return features

    reader = (
        local_pauli_readout
        if readout == "local"
        else pair_pauli_readout
    )
    size = 18 if readout == "local" else 48
    result = np.empty((len(angles), size), dtype=np.float64)

    started = time.time()

    for k, row in enumerate(angles):
        qc = circuit(row, graph, depth)
        state = Statevector.from_instruction(qc)
        result[k] = reader(state.data)

        if (k + 1) % 2000 == 0:
            print(
                graph, depth, readout,
                k + 1, "/", len(angles),
                "seconds:", round(time.time() - started, 1),
                flush=True
            )

    np.savez_compressed(
        cache_path, features=result
    )
    return result


def classical_models(seed):
    return {
        "Classical Logistic": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1500,
                class_weight="balanced"
            )
        ),
        "Classical RBF SVM": make_pipeline(
            StandardScaler(),
            SVC(
                C=1.0, gamma="scale",
                class_weight="balanced"
            )
        ),
        "Classical ExtraTrees": ExtraTreesClassifier(
            n_estimators=350,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced"
        ),
        "Classical HistGB": HistGradientBoostingClassifier(
            max_iter=200,
            max_leaf_nodes=15,
            random_state=seed
        ),
        "Classical RFF512": make_pipeline(
            StandardScaler(),
            RBFSampler(
                n_components=512,
                gamma=0.5,
                random_state=seed
            ),
            LogisticRegression(max_iter=1500)
        ),
        "Classical Polynomial2": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(
                degree=2, include_bias=False
            ),
            StandardScaler(),
            LogisticRegression(max_iter=1500)
        ),
        "Classical MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=0.01,
                max_iter=350,
                early_stopping=True,
                random_state=seed
            )
        ),
    }


def quantum_classifier():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            C=1.0
        )
    )


def metrics(y, predicted, scores):
    return {
        "f1": float(f1_score(y, predicted)),
        "accuracy": float(
            accuracy_score(y, predicted)
        ),
        "precision": float(
            precision_score(y, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predicted, zero_division=0)
        ),
        "mcc": float(
            matthews_corrcoef(y, predicted)
        ),
        "auc": float(
            roc_auc_score(y, scores)
        ),
        "brier": float(
            brier_score_loss(y, scores)
        )
    }


def evaluate(name, model, xtr, ytr, xte, yte, seed):
    start = time.time()
    model.fit(xtr, ytr)

    pred = model.predict(xte)

    if hasattr(model, "predict_proba"):
        score = model.predict_proba(xte)[:, 1]
    else:
        raw = model.decision_function(xte)
        score = 1.0 / (
            1.0 + np.exp(-np.clip(raw, -30, 30))
        )

    row = {
        "seed": seed,
        "model": name,
        "n_train": len(ytr),
        "n_test": len(yte),
        "fit_seconds": round(time.time() - start, 3),
        **metrics(yte, pred, score)
    }

    return row, pred, score


def latex_output(summary):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Qiskit statevector architecture ablation "
        r"versus matched classical controls. "
        r"Values are mean outer-test metrics.}",
        r"\label{tab:qtrust_r3_24_architectures}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & F1 & Accuracy & AUC & MCC \\",
        r"\midrule"
    ]

    for _, row in summary.iterrows():
        name = (
            str(row["model"])
            .replace("_", r"\_")
            .replace("&", r"\&")
        )
        lines.append(
            f"{name} & "
            f"{row['f1']:.4f} & "
            f"{row['accuracy']:.4f} & "
            f"{row['auc']:.4f} & "
            f"{row['mcc']:.4f} "
            + r"\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])

    (OUTPUT / "FULL_ABLATION_TABLE.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="One seed and only first six quantum variants"
    )
    parser.add_argument(
        "--seeds", type=int, default=3
    )
    parser.add_argument(
        "--max-per-class", type=int, default=800
    )
    args = parser.parse_args()

    verify_readout()

    base.OUT = OUTPUT
    base.MAX_PER_CLASS_PER_MODALITY = args.max_per_class

    data = base.load_dataset()

    all_results = []
    all_predictions = []
    diagnostics = []

    variants = list(itertools.product(
        GRAPHS, DEPTHS, READOUTS
    ))

    if args.quick:
        variants = variants[:6]
        seeds = (42,)
    else:
        seeds = tuple(range(42, 42 + args.seeds))

    for seed in seeds:
        print("\n", "=" * 65)
        print("SEED:", seed)
        print("=" * 65, flush=True)

        strata = (
            data.modality.astype(str)
            + "_"
            + data.label.astype(str)
        )

        tr, te = train_test_split(
            np.arange(len(data)),
            test_size=0.20,
            random_state=seed,
            stratify=strata
        )

        ytr = data.iloc[tr].label.to_numpy(dtype=int)
        yte = data.iloc[te].label.to_numpy(dtype=int)

        # Reuse the original training-only specialist protocol.
        xtr, xte = base.specialist_evidence(
            data, tr, te, seed
        )

        scaler = StandardScaler().fit(xtr[:, :N])
        atr = np.pi * np.tanh(
            scaler.transform(xtr[:, :N]) / 2.0
        )
        ate = np.pi * np.tanh(
            scaler.transform(xte[:, :N]) / 2.0
        )

        print("Fitting seven classical controls...")

        best_classical = []

        for name, model in classical_models(seed).items():
            row, pred, scores = evaluate(
                name, model,
                xtr, ytr, xte, yte, seed
            )
            all_results.append(row)
            best_classical.append(row["f1"])

            all_predictions.append(
                pd.DataFrame({
                    "seed": seed,
                    "model": name,
                    "source_id": data.iloc[te].source_id.values,
                    "y_true": yte,
                    "y_pred": pred,
                    "y_score": scores
                })
            )

            print(
                name, "F1:",
                round(row["f1"], 5),
                flush=True
            )

        for index, (graph, depth, readout) in enumerate(
            variants, start=1
        ):
            name = (
                f"Qiskit_{graph}_d{depth}_{readout}"
            )

            print(
                "\nQuantum variant",
                index, "/", len(variants),
                name, flush=True
            )

            # Calculate a full local+pair readout only once
            # per graph/depth; local is its first 18 entries.
            full_train_cache = OUTPUT / (
                f"qtrain_seed{seed}_{graph}_d{depth}.npz"
            )
            full_test_cache = OUTPUT / (
                f"qtest_seed{seed}_{graph}_d{depth}.npz"
            )

            full_tr = produce_features(
                atr, graph, depth,
                "pair", full_train_cache
            )
            full_te = produce_features(
                ate, graph, depth,
                "pair", full_test_cache
            )

            # pair_pauli_readout starts with 18 local values.
            if readout == "local":
                qtr = full_tr[:, :18]
                qte = full_te[:, :18]
            else:
                qtr, qte = full_tr, full_te

            # Matched inputs: preserve original evidence.
            ztr = np.column_stack((xtr, qtr))
            zte = np.column_stack((xte, qte))

            row, pred, scores = evaluate(
                name,
                quantum_classifier(),
                ztr, ytr, zte, yte, seed
            )
            row["graph"] = graph
            row["depth"] = depth
            row["readout"] = readout
            all_results.append(row)

            all_predictions.append(
                pd.DataFrame({
                    "seed": seed,
                    "model": name,
                    "source_id": data.iloc[te].source_id.values,
                    "y_true": yte,
                    "y_pred": pred,
                    "y_score": scores
                })
            )

            print(
                name,
                "F1:", round(row["f1"], 5),
                "AUC:", round(row["auc"], 5),
                flush=True
            )

            # A circuit diagram and QASM export.
            if seed == seeds[0] and readout == "pair":
                example = circuit(
                    ate[0], graph, depth
                )
                (OUTPUT / f"{name}.qasm").write_text(
                    qasm2.dumps(example),
                    encoding="utf-8"
                )
                (OUTPUT / f"{name}.txt").write_text(
                    str(example.draw(output="text")),
                    encoding="utf-8"
                )

                entropy_values = []
                for angles in ate[:20]:
                    sv = Statevector.from_instruction(
                        circuit(angles, graph, depth)
                    )
                    reduced = partial_trace(
                        sv, list(range(1, N))
                    )
                    entropy_values.append(
                        float(entropy(reduced))
                    )
                diagnostics.append({
                    "model": name,
                    "mean_single_qubit_entropy":
                        float(np.mean(entropy_values)),
                    "max_single_qubit_entropy":
                        float(np.max(entropy_values))
                })

            # Save after every model so a stopped run
            # retains already completed results.
            pd.DataFrame(all_results).to_csv(
                OUTPUT / "RUNNING_RESULTS.csv",
                index=False
            )

    results = pd.DataFrame(all_results)

    results.to_csv(
        OUTPUT / "ALL_SEED_METRICS.csv",
        index=False
    )

    predictions = pd.concat(
        all_predictions, ignore_index=True
    )
    predictions.to_csv(
        OUTPUT / "ALL_TEST_PREDICTIONS.csv",
        index=False
    )

    pd.DataFrame(diagnostics).to_csv(
        OUTPUT / "ENTANGLEMENT_DIAGNOSTICS.csv",
        index=False
    )

    summary = (
        results
        .groupby("model", as_index=False)[
            ["f1", "accuracy", "precision", "recall",
             "mcc", "auc", "brier"]
        ]
        .mean()
        .sort_values("f1", ascending=False)
    )

    summary.to_csv(
        OUTPUT / "FINAL_24_VARIANT_ABLATION.csv",
        index=False
    )

    # Compare every quantum variant with the strongest
    # classical F1 from the SAME seed.
    comparisons = []

    for seed in seeds:
        part = results[results.seed == seed]

        classical = part[
            ~part.model.str.startswith("Qiskit_")
        ]
        best = classical.sort_values(
            "f1", ascending=False
        ).iloc[0]

        quantum = part[
            part.model.str.startswith("Qiskit_")
        ]

        for _, row in quantum.iterrows():
            comparisons.append({
                "seed": seed,
                "quantum_model": row["model"],
                "quantum_f1": row["f1"],
                "best_classical_model": best["model"],
                "best_classical_f1": best["f1"],
                "delta_f1": row["f1"] - best["f1"]
            })

    pd.DataFrame(comparisons).to_csv(
        OUTPUT / "PAIRED_QUANTUM_VS_CLASSICAL.csv",
        index=False
    )

    latex_output(summary)

    with open(
        OUTPUT / "EXPERIMENT_MANIFEST.json",
        "w", encoding="utf-8"
    ) as f:
        json.dump({
            "quantum_variants": [
                f"Qiskit_{g}_d{d}_{r}"
                for g, d, r in variants
            ],
            "seeds": list(seeds),
            "qubits": N,
            "backend": "Qiskit exact Statevector",
            "source_directories": [
                str(p) for p in base.DATA_ROOTS
            ],
            "note": (
                "Circuit simulation is classical. "
                "No quantum computational speedup is claimed. "
                "All 24 architecture results are exploratory. "
                "Selecting the highest test F1 is not "
                "independent validation."
            )
        }, f, indent=2)

    print("\nFINAL COMPARISON")
    print(summary.to_string(index=False))

    print("\nOutput directory:", OUTPUT)
    print("Completed quantum variants:", len(variants))
    print("Completed seeds:", len(seeds))


if __name__ == "__main__":
    main()
