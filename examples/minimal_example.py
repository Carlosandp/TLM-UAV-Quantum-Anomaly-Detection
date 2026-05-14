"""Minimal smoke-test for the qiskit-data-reuploading classifier.

Runs entirely on synthetic data — no TLM:UAV dataset required.
Validates that the pinned `qdr` dependency is installed correctly and
that a small DRU circuit can be trained end-to-end on the local Aer
simulator.

Usage:
    python examples/minimal_example.py
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.model_selection import train_test_split

from qdr import DataReuploadingClassifier


def main() -> int:
    rng = np.random.default_rng(0)

    # Synthetic linearly-separable problem, scaled to [-pi/2, pi/2]
    n_samples, n_features = 120, 2
    X = rng.uniform(-np.pi / 2, np.pi / 2, size=(n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0, stratify=y
    )

    clf = DataReuploadingClassifier(
        n_qubits=2,
        n_layers=3,
        max_iter=200,
        seed=0,
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"DRU train accuracy: {train_acc:.3f}")
    print(f"DRU test  accuracy: {test_acc:.3f}")

    if train_acc < 0.70:
        print(
            "ERROR: training accuracy is implausibly low — check the install.",
            file=sys.stderr,
        )
        return 1

    print("OK — qdr is installed and the DRU classifier trains correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
