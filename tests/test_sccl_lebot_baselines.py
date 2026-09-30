from pathlib import Path

import pandas as pd

ROOT = Path("results/additional_clustering_baselines")


def read_mean_nmi(path: Path) -> float:
    frame = pd.read_csv(path)
    return float(frame.groupby("level")["nmi"].mean().mean())


def test_sccl_and_lebot_results():
    expected = {
        ROOT / "sccl/sccl_native_metrics.csv": 0.282212344,
        ROOT / "sccl/sccl_spherical_metrics.csv": 0.338876516,
        ROOT / "lebot/lebot_qwen06b_metrics.csv": 0.089246460,
    }

    for path, expected_value in expected.items():
        assert path.exists(), path
        observed = read_mean_nmi(path)
        assert abs(observed - expected_value) < 1e-7
