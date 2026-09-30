# ruff: noqa
# Auto-exported from the executed Google Colab experiment.
# Requires the processed USPTO records in the paths documented below.

# ============================================================
# Publish SCCL + LeBoT results to GitHub and Hugging Face
# ============================================================

import os
import re
import sys
import json
import shutil
import hashlib
import getpass
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------
# 1. Configuration
# ------------------------------------------------------------
GITHUB_OWNER = "Yongmin-Yoo"
GITHUB_REPO = "claimsem"
BRANCH = "experiments/sccl-lebot-baselines"
COMMIT_MESSAGE = "Add SCCL and LeBoT-style clustering baselines"

HF_REPO_ID = "yongminyoo91/roots-additional-clustering-baselines"
HF_PATH_IN_REPO = "sccl_lebot"

DRIVE_ROOT = Path("/content/drive/MyDrive/claimsem_artifacts")

SCCL_ROOT = DRIVE_ROOT / "sccl_document_clustering"
SCCL_RESULTS = SCCL_ROOT / "results"
SCCL_PROTOCOL = SCCL_ROOT / "protocol.json"

LEBOT_ROOT = DRIVE_ROOT / "lebot_document_clustering"
LEBOT_RESULTS = LEBOT_ROOT / "results"
LEBOT_PROTOCOL = LEBOT_ROOT / "protocol.json"

REPO_DIR = Path("/content/claimsem-sccl-lebot-publication")
HF_BUNDLE = Path("/content/sccl-lebot-hf-bundle")

RESULT_ROOT_REL = Path("results/additional_clustering_baselines")
GITHUB_RESULT_ROOT = REPO_DIR / RESULT_ROOT_REL
GITHUB_SCCL_DIR = GITHUB_RESULT_ROOT / "sccl"
GITHUB_LEBOT_DIR = GITHUB_RESULT_ROOT / "lebot"
SCRIPT_DIR = REPO_DIR / "scripts"
CONFIG_DIR = REPO_DIR / "configs"
DOC_DIR = REPO_DIR / "docs"
TEST_DIR = REPO_DIR / "tests"

EXPECTED = {
    "sccl_native": 0.282212,
    "sccl_spherical": 0.338877,
    "lebot_qwen06b": 0.089246,
}

# GitHub에는 소형·텍스트 결과만 저장합니다.
GITHUB_EXTENSIONS = {".csv", ".json", ".tex"}

# Hugging Face에는 재현용 예측·표현도 포함합니다.
HF_EXTENSIONS = {".csv", ".json", ".tex", ".npz", ".npy"}

# 원문 또는 대형 체크포인트 관련 파일은 제외합니다.
BLOCKED_NAME_PARTS = {
    "checkpoint",
    "input_ids",
    "attention_mask",
    "patent_snippets",
    "records.pkl",
    "train_records",
    "dev_records",
    "test_records",
}


# ------------------------------------------------------------
# 2. Helpers
# ------------------------------------------------------------
def run(command, cwd=None, env=None, capture=False):
    print("$", " ".join(map(str, command)))
    result = subprocess.run(
        list(map(str, command)),
        cwd=cwd,
        env=env,
        text=True,
        capture_output=capture,
    )
    if capture:
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            + " ".join(map(str, command))
        )
    return result


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_blocked(path):
    lowered = path.as_posix().lower()
    return any(part in lowered for part in BLOCKED_NAME_PARTS)


def copy_selected(source_dir, destination_dir, extensions, max_bytes=None):
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = []

    for source in sorted(source_dir.rglob("*")):
        if not source.is_file():
            continue
        if source.suffix.lower() not in extensions:
            continue
        if is_blocked(source):
            continue
        if max_bytes is not None and source.stat().st_size > max_bytes:
            continue

        relative = source.relative_to(source_dir)
        destination = destination_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    return copied


def mean_nmi_from_csv(path):
    frame = pd.read_csv(path)
    required = {"level", "nmi"}
    assert required.issubset(frame.columns), (
        f"Missing columns in {path}: {required - set(frame.columns)}"
    )

    level_means = frame.groupby("level")["nmi"].mean()
    return float(level_means.mean()), {
        str(level): float(value) for level, value in level_means.items()
    }


def find_result_file(root, exact_name):
    matches = list(root.rglob(exact_name))
    assert len(matches) == 1, (
        f"Expected exactly one {exact_name} under {root}, found {matches}"
    )
    return matches[0]


def normalize_text_files(root):
    text_extensions = {
        ".py",
        ".json",
        ".csv",
        ".tex",
        ".md",
        ".txt",
        ".sha256",
        ".yml",
        ".yaml",
    }

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_extensions:
            continue

        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")

        # Remove trailing spaces without modifying intentional blank lines.
        lines = [line.rstrip(b" \t") for line in data.split(b"\n")]
        normalized = b"\n".join(lines)
        if normalized and not normalized.endswith(b"\n"):
            normalized += b"\n"
        path.write_bytes(normalized)


def check_for_secrets(root):
    token_patterns = [
        re.compile(rb"hf_[A-Za-z0-9]{20,}"),
        re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
        re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    ]

    violations = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            continue

        try:
            data = path.read_bytes()
        except OSError:
            continue

        if any(pattern.search(data) for pattern in token_patterns):
            violations.append(str(path))

    assert not violations, f"Possible tokens found: {violations}"


def write_checksums(root, output_name="checksums.sha256"):
    output = root / output_name
    lines = []

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(root).as_posix()
        lines.append(f"{sha256(path)}  {relative}")

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


# ------------------------------------------------------------
# 3. Verify source artifacts
# ------------------------------------------------------------
required_files = [
    SCCL_RESULTS / "sccl_native_metrics.csv",
    SCCL_RESULTS / "sccl_spherical_metrics.csv",
    SCCL_RESULTS / "sccl_results.json",
    SCCL_RESULTS / "sccl_rows.tex",
    SCCL_PROTOCOL,
    LEBOT_RESULTS / "lebot_qwen06b_metrics.csv",
    LEBOT_RESULTS / "lebot_qwen06b_results.json",
    LEBOT_RESULTS / "lebot_qwen06b_predictions.npz",
    LEBOT_RESULTS / "lebot_qwen06b_bot_vectors.npy",
    LEBOT_RESULTS / "lebot_qwen06b_row.tex",
    LEBOT_PROTOCOL,
]

missing = [str(path) for path in required_files if not path.exists()]
assert not missing, "Missing source artifacts:\n" + "\n".join(missing)

native_mean, native_levels = mean_nmi_from_csv(SCCL_RESULTS / "sccl_native_metrics.csv")
spherical_mean, spherical_levels = mean_nmi_from_csv(
    SCCL_RESULTS / "sccl_spherical_metrics.csv"
)
lebot_mean, lebot_levels = mean_nmi_from_csv(
    LEBOT_RESULTS / "lebot_qwen06b_metrics.csv"
)

assert abs(native_mean - EXPECTED["sccl_native"]) < 5e-5, native_mean
assert abs(spherical_mean - EXPECTED["sccl_spherical"]) < 5e-5, spherical_mean
assert abs(lebot_mean - EXPECTED["lebot_qwen06b"]) < 5e-5, lebot_mean

print("\nVerified mean NMI:")
print(f"  SCCL native:    {native_mean:.6f}")
print(f"  SCCL spherical: {spherical_mean:.6f}")
print(f"  LeBoT-style:    {lebot_mean:.6f}")

# ------------------------------------------------------------
# 4. Recover executed notebook code from Colab history
# ------------------------------------------------------------
history = list(get_ipython().user_ns.get("In", []))


def find_history_cell(marker):
    candidates = [
        source for source in history if isinstance(source, str) and marker in source
    ]
    return candidates[-1] if candidates else None


source_cells = {
    "prepare_sccl_corpus.py": find_history_cell("SCCL test corpus prepared"),
    "tokenize_sccl_inputs.py": find_history_cell("SCCL token cache prepared"),
    "run_sccl_document_baseline.py": find_history_cell("FINAL SCCL RESULTS"),
    "run_lebot_document_baseline.py": find_history_cell("FINAL LEBOT RESULTS"),
}

assert source_cells["run_sccl_document_baseline.py"], (
    "SCCL 실행 셀을 Colab 입력 기록에서 찾지 못했습니다."
)
assert source_cells["run_lebot_document_baseline.py"], (
    "LeBoT 실행 셀을 Colab 입력 기록에서 찾지 못했습니다."
)

# ------------------------------------------------------------
# 5. Clone current main and create publication branch
# ------------------------------------------------------------
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

run(
    [
        "git",
        "clone",
        f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git",
        str(REPO_DIR),
    ]
)
run(["git", "checkout", "-b", BRANCH], cwd=REPO_DIR)

main_sha = run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO_DIR,
    capture=True,
).stdout.strip()

print("Base main SHA:", main_sha)

for directory in (
    GITHUB_SCCL_DIR,
    GITHUB_LEBOT_DIR,
    SCRIPT_DIR,
    CONFIG_DIR,
    DOC_DIR,
    TEST_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 6. Copy small GitHub result artifacts
# ------------------------------------------------------------
github_sccl_files = copy_selected(
    SCCL_RESULTS,
    GITHUB_SCCL_DIR,
    GITHUB_EXTENSIONS,
    max_bytes=5 * 1024 * 1024,
)
github_lebot_files = copy_selected(
    LEBOT_RESULTS,
    GITHUB_LEBOT_DIR,
    GITHUB_EXTENSIONS,
    max_bytes=5 * 1024 * 1024,
)

shutil.copy2(SCCL_PROTOCOL, GITHUB_SCCL_DIR / "protocol.json")
shutil.copy2(LEBOT_PROTOCOL, GITHUB_LEBOT_DIR / "protocol.json")

# ------------------------------------------------------------
# 7. Save notebook source snapshots
# ------------------------------------------------------------
for filename, source in source_cells.items():
    if not source:
        continue

    output = SCRIPT_DIR / filename
    output.write_text(
        "# ruff: noqa\n"
        "# Auto-exported from the executed Google Colab experiment.\n"
        "# Requires the processed USPTO records in the paths documented below.\n\n"
        + source.strip()
        + "\n",
        encoding="utf-8",
    )

# ------------------------------------------------------------
# 8. Write configuration
# ------------------------------------------------------------
config = {
    "experiment": "sccl_lebot_document_clustering",
    "dataset": "USPTO-70k processed TEST split",
    "n_documents": 9881,
    "cpc_cardinality": {
        "section": 9,
        "class": 121,
        "subclass": 466,
    },
    "n_clusters": 30,
    "clustering_seeds": [17, 42, 73],
    "input_policy": {
        "claim_order": "numeric claim-ID order",
        "maximum_tokens": 512,
        "labels_used_during_training": False,
    },
    "sccl": {
        "encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "adaptation": "modern document-level SCCL adaptation",
        "native_mean_nmi": native_mean,
        "spherical_mean_nmi": spherical_mean,
        "native_level_nmi": native_levels,
        "spherical_level_nmi": spherical_levels,
    },
    "lebot": {
        "name": "LeBoT-style L4 adaptation",
        "llm": "Qwen/Qwen3-0.6B",
        "retriever": "sentence-transformers/all-MiniLM-L6-v2",
        "bot_dimension": 1024,
        "maximum_refinement_iterations": 2,
        "mean_nmi": lebot_mean,
        "level_nmi": lebot_levels,
        "exact_original_reproduction": False,
    },
    "publication_policy": {
        "raw_patent_records_uploaded": False,
        "claim_text_uploaded": False,
        "token_cache_uploaded": False,
        "training_checkpoints_uploaded": False,
        "large_reproducibility_artifacts": (
            f"https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/{HF_PATH_IN_REPO}"
        ),
    },
}

(CONFIG_DIR / "sccl_lebot_baselines.json").write_text(
    json.dumps(config, indent=2) + "\n",
    encoding="utf-8",
)

# ------------------------------------------------------------
# 9. Write concise documentation
# ------------------------------------------------------------
documentation = f"""# SCCL and LeBoT-style clustering baselines

This experiment evaluates document-level adaptations of SCCL and LeBoT on
the fixed USPTO-70k TEST split containing 9,881 patents.

## Results

| Method | Mean TEST NMI |
|---|---:|
| SCCL native clustering head | {native_mean:.6f} |
| SCCL representations + spherical K-means | {spherical_mean:.6f} |
| LeBoT-style (Qwen3-0.6B) | {lebot_mean:.6f} |

The main paper reports SCCL representations with spherical K-means because
this produces exactly 30 clusters under the matched clustering protocol.
The native SCCL result is retained for completeness.

The LeBoT-style experiment is an L4-efficient patent adaptation using
Qwen3-0.6B, 512-token truncation, MiniLM retrieval, and a 1,024-dimensional
bag-of-texts representation. It is not an exact reproduction of the
Gemma-2-9B configuration in the original paper.

## Data policy

Raw USPTO records, patent claim text, token caches, and training checkpoints
are not redistributed. Large prediction and representation artifacts are
stored in the associated Hugging Face dataset:

https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/{HF_PATH_IN_REPO}
"""

(DOC_DIR / "sccl_lebot_baselines.md").write_text(
    documentation,
    encoding="utf-8",
)

# ------------------------------------------------------------
# 10. Update additional-baseline summaries
# ------------------------------------------------------------
summary_csv = GITHUB_RESULT_ROOT / "summary.csv"

if summary_csv.exists():
    summary_frame = pd.read_csv(summary_csv)
else:
    summary_frame = pd.DataFrame(
        columns=[
            "method",
            "mean_nmi",
            "section_nmi",
            "class_nmi",
            "subclass_nmi",
        ]
    )

new_rows = pd.DataFrame(
    [
        {
            "method": "SCCL + spherical K-means",
            "mean_nmi": spherical_mean,
            "section_nmi": spherical_levels["section"],
            "class_nmi": spherical_levels["class"],
            "subclass_nmi": spherical_levels["subclass"],
        },
        {
            "method": "LeBoT-style (Qwen3-0.6B)",
            "mean_nmi": lebot_mean,
            "section_nmi": lebot_levels["section"],
            "class_nmi": lebot_levels["class"],
            "subclass_nmi": lebot_levels["subclass"],
        },
    ]
)

replace_methods = set(new_rows["method"])
summary_frame = summary_frame[~summary_frame["method"].isin(replace_methods)]
summary_frame = pd.concat(
    [summary_frame, new_rows],
    ignore_index=True,
)

summary_frame.to_csv(
    summary_csv,
    index=False,
    lineterminator="\n",
    float_format="%.6f",
)

summary_json = GITHUB_RESULT_ROOT / "summary.json"
summary_json.write_text(
    json.dumps(
        summary_frame.to_dict(orient="records"),
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

# ------------------------------------------------------------
# 11. Add automated result test
# ------------------------------------------------------------
test_source = f"""from pathlib import Path

import pandas as pd


ROOT = Path("results/additional_clustering_baselines")


def read_mean_nmi(path: Path) -> float:
    frame = pd.read_csv(path)
    return float(frame.groupby("level")["nmi"].mean().mean())


def test_sccl_and_lebot_results():
    expected = {{
        ROOT / "sccl/sccl_native_metrics.csv": {native_mean:.9f},
        ROOT / "sccl/sccl_spherical_metrics.csv": {spherical_mean:.9f},
        ROOT / "lebot/lebot_qwen06b_metrics.csv": {lebot_mean:.9f},
    }}

    for path, expected_value in expected.items():
        assert path.exists(), path
        observed = read_mean_nmi(path)
        assert abs(observed - expected_value) < 1e-7
"""

(TEST_DIR / "test_sccl_lebot_baselines.py").write_text(
    test_source,
    encoding="utf-8",
)

# ------------------------------------------------------------
# 12. Recompute GitHub result checksums
# ------------------------------------------------------------
write_checksums(GITHUB_RESULT_ROOT)

normalize_text_files(REPO_DIR)
check_for_secrets(REPO_DIR)

# ------------------------------------------------------------
# 13. Run local CI exactly before publication
# ------------------------------------------------------------
run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"],
    cwd=REPO_DIR,
)

# Format exported source files.
run(
    [sys.executable, "-m", "ruff", "format", "scripts"],
    cwd=REPO_DIR,
)
run(
    [sys.executable, "-m", "ruff", "check", "--fix", "."],
    cwd=REPO_DIR,
)
run(
    [sys.executable, "-m", "ruff", "format", "--check", "."],
    cwd=REPO_DIR,
)
run(
    [sys.executable, "-m", "pytest", "-q"],
    cwd=REPO_DIR,
)

normalize_text_files(REPO_DIR)
write_checksums(GITHUB_RESULT_ROOT)
normalize_text_files(REPO_DIR)
check_for_secrets(REPO_DIR)

print("\nSUCCESS: local Ruff and pytest checks passed.")

# ------------------------------------------------------------
# 14. Build Hugging Face bundle
# ------------------------------------------------------------
if HF_BUNDLE.exists():
    shutil.rmtree(HF_BUNDLE)

(HF_BUNDLE / "results/sccl").mkdir(parents=True)
(HF_BUNDLE / "results/lebot").mkdir(parents=True)
(HF_BUNDLE / "code").mkdir(parents=True)
(HF_BUNDLE / "config").mkdir(parents=True)

hf_sccl_files = copy_selected(
    SCCL_RESULTS,
    HF_BUNDLE / "results/sccl",
    HF_EXTENSIONS,
)
hf_lebot_files = copy_selected(
    LEBOT_RESULTS,
    HF_BUNDLE / "results/lebot",
    HF_EXTENSIONS,
)

shutil.copy2(
    SCCL_PROTOCOL,
    HF_BUNDLE / "results/sccl/protocol.json",
)
shutil.copy2(
    LEBOT_PROTOCOL,
    HF_BUNDLE / "results/lebot/protocol.json",
)
shutil.copy2(
    CONFIG_DIR / "sccl_lebot_baselines.json",
    HF_BUNDLE / "config/sccl_lebot_baselines.json",
)

for source in sorted(SCRIPT_DIR.glob("*sccl*.py")):
    shutil.copy2(source, HF_BUNDLE / "code" / source.name)

for source in sorted(SCRIPT_DIR.glob("*lebot*.py")):
    shutil.copy2(source, HF_BUNDLE / "code" / source.name)

hf_readme = f"""# SCCL and LeBoT-style USPTO clustering artifacts

This directory contains result files and reproducibility artifacts for the
SCCL and LeBoT-style patent clustering baselines.

- SCCL spherical mean NMI: {spherical_mean:.6f}
- SCCL native mean NMI: {native_mean:.6f}
- LeBoT-style mean NMI: {lebot_mean:.6f}

The LeBoT-style result uses Qwen3-0.6B and is not an exact reproduction of
the original Gemma-2-9B setting.

Raw patent records, claim text, token caches, and checkpoints are excluded.
"""

(HF_BUNDLE / "README.md").write_text(
    hf_readme,
    encoding="utf-8",
)

normalize_text_files(HF_BUNDLE)
write_checksums(HF_BUNDLE)
check_for_secrets(HF_BUNDLE)

print("\nHugging Face bundle files:")
for path in sorted(HF_BUNDLE.rglob("*")):
    if path.is_file():
        print(f"  {path.relative_to(HF_BUNDLE)} ({path.stat().st_size / 2**20:.2f} MB)")

# ------------------------------------------------------------
# 15. Upload to Hugging Face
# ------------------------------------------------------------
run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "huggingface_hub>=0.36.0",
    ]
)

from huggingface_hub import HfApi

hf_token = getpass.getpass("Hugging Face token (hidden): ").strip()
assert hf_token, "Hugging Face token is required."

hf_api = HfApi(token=hf_token)
hf_api.create_repo(
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    private=True,
    exist_ok=True,
)

hf_commit = hf_api.upload_folder(
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    folder_path=str(HF_BUNDLE),
    path_in_repo=HF_PATH_IN_REPO,
    commit_message="Add SCCL and LeBoT-style clustering artifacts",
)

hf_files = hf_api.list_repo_files(
    repo_id=HF_REPO_ID,
    repo_type="dataset",
)

required_hf_paths = [
    f"{HF_PATH_IN_REPO}/results/sccl/sccl_spherical_metrics.csv",
    f"{HF_PATH_IN_REPO}/results/lebot/lebot_qwen06b_metrics.csv",
    f"{HF_PATH_IN_REPO}/results/lebot/lebot_qwen06b_predictions.npz",
    f"{HF_PATH_IN_REPO}/results/lebot/lebot_qwen06b_bot_vectors.npy",
    f"{HF_PATH_IN_REPO}/checksums.sha256",
]

for required_path in required_hf_paths:
    assert required_path in hf_files, f"Missing from HF: {required_path}"

print("SUCCESS: Hugging Face synchronization completed.")

# Remove token from memory as soon as possible.
del hf_token

# ------------------------------------------------------------
# 16. Stage, verify and commit GitHub files
# ------------------------------------------------------------
normalize_text_files(REPO_DIR)
write_checksums(GITHUB_RESULT_ROOT)
normalize_text_files(REPO_DIR)

run(["git", "config", "user.name", "Yongmin Yoo"], cwd=REPO_DIR)
run(
    [
        "git",
        "config",
        "user.email",
        "59948809+Yongmin-Yoo@users.noreply.github.com",
    ],
    cwd=REPO_DIR,
)

run(
    [
        "git",
        "add",
        "configs/sccl_lebot_baselines.json",
        "docs/sccl_lebot_baselines.md",
        "results/additional_clustering_baselines",
        "scripts",
        "tests/test_sccl_lebot_baselines.py",
    ],
    cwd=REPO_DIR,
)

diff_check = subprocess.run(
    ["git", "diff", "--cached", "--check"],
    cwd=REPO_DIR,
    text=True,
    capture_output=True,
)
print(diff_check.stdout)
print(diff_check.stderr)
assert diff_check.returncode == 0, "git diff --cached --check failed"

staged = run(
    ["git", "diff", "--cached", "--name-only"],
    cwd=REPO_DIR,
    capture=True,
).stdout.strip()

assert staged, "No staged files found."
print("Staged files:\n", staged)

run(
    ["git", "commit", "-m", COMMIT_MESSAGE],
    cwd=REPO_DIR,
)

commit_sha = run(
    ["git", "rev-parse", "HEAD"],
    cwd=REPO_DIR,
    capture=True,
).stdout.strip()

# ------------------------------------------------------------
# 17. Push branch securely
# ------------------------------------------------------------
github_token = getpass.getpass("GitHub token (hidden): ").strip()
assert github_token, "GitHub token is required."

askpass_path = Path("/tmp/claimsem_git_askpass.sh")
askpass_path.write_text(
    "#!/bin/sh\n"
    'case "$1" in\n'
    '  *Username*) echo "x-access-token" ;;\n'
    '  *Password*) echo "$GITHUB_TOKEN" ;;\n'
    "esac\n",
    encoding="utf-8",
)
askpass_path.chmod(0o700)

push_env = os.environ.copy()
push_env["GIT_ASKPASS"] = str(askpass_path)
push_env["GIT_TERMINAL_PROMPT"] = "0"
push_env["GITHUB_TOKEN"] = github_token

run(
    ["git", "push", "-u", "origin", BRANCH],
    cwd=REPO_DIR,
    env=push_env,
)

remote_sha = run(
    ["git", "ls-remote", "origin", f"refs/heads/{BRANCH}"],
    cwd=REPO_DIR,
    capture=True,
).stdout.split()[0]

assert remote_sha == commit_sha, (
    f"Remote SHA mismatch: local={commit_sha}, remote={remote_sha}"
)

# ------------------------------------------------------------
# 18. Create or find draft PR
# ------------------------------------------------------------
headers = {
    "Authorization": f"Bearer {github_token}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

pr_api = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/pulls"

existing_response = requests.get(
    pr_api,
    headers=headers,
    params={
        "state": "open",
        "head": f"{GITHUB_OWNER}:{BRANCH}",
    },
    timeout=60,
)
existing_response.raise_for_status()
existing_prs = existing_response.json()

pr_body = f"""## Summary

- adds the SCCL native-head and spherical K-means results;
- adds the Qwen3-0.6B LeBoT-style patent adaptation;
- records protocols, metrics, LaTeX rows, checksums, and regression tests;
- stores large predictions and representations on Hugging Face.

## Mean TEST NMI

- SCCL native head: {native_mean:.6f}
- SCCL + spherical K-means: {spherical_mean:.6f}
- LeBoT-style (Qwen3-0.6B): {lebot_mean:.6f}

## Artifacts

https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/{HF_PATH_IN_REPO}

Raw USPTO records, claim text, token caches, and training checkpoints are
not redistributed.
"""

if existing_prs:
    pr = existing_prs[0]
else:
    create_response = requests.post(
        pr_api,
        headers=headers,
        json={
            "title": "Add SCCL and LeBoT-style clustering baselines",
            "head": BRANCH,
            "base": "main",
            "body": pr_body,
            "draft": True,
        },
        timeout=60,
    )
    create_response.raise_for_status()
    pr = create_response.json()

assert pr["head"]["sha"] == commit_sha, (
    f"PR head mismatch: {pr['head']['sha']} != {commit_sha}"
)

# ------------------------------------------------------------
# 19. Final verification
# ------------------------------------------------------------
status = run(
    ["git", "status", "--porcelain"],
    cwd=REPO_DIR,
    capture=True,
).stdout.strip()

assert not status, f"Repository is not clean:\n{status}"

askpass_path.unlink(missing_ok=True)
del github_token

hf_commit_id = getattr(hf_commit, "oid", None) or str(hf_commit)

print("\n" + "=" * 80)
print("PUBLICATION SUCCESS")
print("=" * 80)
print("Base main SHA :", main_sha)
print("Commit SHA    :", commit_sha)
print(
    "Commit URL    :",
    f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/commit/{commit_sha}",
)
print("PR URL        :", pr["html_url"])
print("PR draft      :", pr["draft"])
print(
    "HF dataset    :",
    f"https://huggingface.co/datasets/{HF_REPO_ID}/tree/main/{HF_PATH_IN_REPO}",
)
print("HF commit     :", hf_commit_id)
print("HF files      :", len(hf_files))
print("SCCL mean NMI :", f"{spherical_mean:.6f}")
print("LeBoT mean NMI:", f"{lebot_mean:.6f}")
print("Raw records   : NOT uploaded")
print("Claim text    : NOT uploaded")
print("Token caches  : NOT uploaded")
print("Checkpoints   : NOT uploaded")
print("\nNext: GitHub CI가 통과하면 PR을 Ready for review로 전환하고 병합하세요.")
