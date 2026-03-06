"""
Run a reproducible evaluation workflow and capture environment metadata.

This script standardizes experiment execution so runs can be compared fairly.
"""

import argparse
import json
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict


def run_cmd(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a command and return the completed process."""
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def get_package_version(pkg_name: str) -> str:
    """Read installed package version safely."""
    try:
        module = __import__(pkg_name)
        return getattr(module, "__version__", "unknown")
    except Exception:
        return "not-installed"


def get_git_commit(repo_root: Path) -> str:
    """Get current git commit hash if available."""
    proc = run_cmd(["git", "rev-parse", "HEAD"], repo_root)
    if proc.returncode == 0:
        return proc.stdout.strip()
    return "unknown"


def get_git_branch(repo_root: Path) -> str:
    """Get current git branch if available."""
    proc = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], repo_root)
    if proc.returncode == 0:
        return proc.stdout.strip()
    return "unknown"


def collect_env_metadata(repo_root: Path, tag: str, pipeline_cmd: list[str]) -> Dict:
    """Collect metadata that affects reproducibility."""
    return {
        "tag": tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": get_git_commit(repo_root),
        "git_branch": get_git_branch(repo_root),
        "package_versions": {
            "numpy": get_package_version("numpy"),
            "pandas": get_package_version("pandas"),
            "scikit_learn": get_package_version("sklearn"),
            "joblib": get_package_version("joblib"),
            "pyarrow": get_package_version("pyarrow"),
        },
        "commands": {
            "pipeline": " ".join(pipeline_cmd),
            "analysis": "scripts/analyze_results.py",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible Mayhem evaluation")
    parser.add_argument(
        "--tag",
        required=True,
        help="Experiment tag (example: exp_norm_v1)",
    )
    parser.add_argument(
        "--mode",
        choices=["full", "eval-only"],
        default="full",
        help="full = retrain + evaluate (without 2k inference), eval-only = evaluate existing artifacts",
    )
    parser.add_argument(
        "--attributes",
        nargs="*",
        default=None,
        help="Optional attribute subset (name phone website address category)",
    )
    parser.add_argument(
        "--pipeline-extra-args",
        nargs="*",
        default=None,
        help="Extra args passed through to scripts/run_algorithm_pipeline.py",
    )
    parser.add_argument(
        "--pipeline-extra-args-str",
        default="",
        help="Quoted extra args string passed through to scripts/run_algorithm_pipeline.py",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "data" / "results"
    reports_dir = results_dir / "experiment_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "full":
        pipeline_cmd = [
            sys.executable,
            "scripts/run_algorithm_pipeline.py",
            "--skip-inference-2k",
            "--skip-consolidation",
        ]
    else:
        pipeline_cmd = [
            sys.executable,
            "scripts/run_algorithm_pipeline.py",
            "--skip-golden",
            "--skip-features",
            "--skip-ml",
            "--skip-inference-2k",
            "--skip-consolidation",
        ]

    if args.attributes:
        pipeline_cmd.extend(["--attributes", *args.attributes])

    if args.pipeline_extra_args:
        pipeline_cmd.extend(args.pipeline_extra_args)

    if args.pipeline_extra_args_str:
        pipeline_cmd.extend(shlex.split(args.pipeline_extra_args_str))

    print("=" * 80)
    print(f"Running experiment tag: {args.tag}")
    print(f"Mode: {args.mode}")
    print(f"Pipeline command: {' '.join(pipeline_cmd)}")
    print("=" * 80)

    pipeline_proc = run_cmd(pipeline_cmd, repo_root)
    pipeline_log_path = reports_dir / f"{args.tag}_pipeline.log"
    pipeline_log_path.write_text(
        pipeline_proc.stdout + ("\nSTDERR:\n" + pipeline_proc.stderr if pipeline_proc.stderr else ""),
        encoding="utf-8",
    )

    if pipeline_proc.returncode != 0:
        print("Pipeline failed. See log:", pipeline_log_path)
        return pipeline_proc.returncode

    analysis_proc = run_cmd([sys.executable, "scripts/analyze_results.py"], repo_root)
    report_path = reports_dir / f"{args.tag}_report.txt"
    report_path.write_text(
        analysis_proc.stdout + ("\nSTDERR:\n" + analysis_proc.stderr if analysis_proc.stderr else ""),
        encoding="utf-8",
    )

    if analysis_proc.returncode != 0:
        print("Analysis failed. See report:", report_path)
        return analysis_proc.returncode

    metadata = collect_env_metadata(repo_root, args.tag, pipeline_cmd)
    metadata_path = reports_dir / f"{args.tag}_meta.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Experiment artifacts saved:")
    print("-", pipeline_log_path)
    print("-", report_path)
    print("-", metadata_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
