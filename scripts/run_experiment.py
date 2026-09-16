#!/usr/bin/env python3
"""Run the paper evaluation settings from the repository root."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "run_challenge_with_restart.py"
SNR_GRID_PATH = REPO_ROOT / "run_noise_challenge_snr_grid.py"
ALL_EPISODES = list(range(24))

SETTINGS: Dict[str, Dict[str, object]] = {
    "no_comm": {
        "experiment_name": "noise_free",
        "run_id": "no_comm",
        "communication": False,
        "token_comm": False,
        "enable_noise": False,
        "prompt_template_path": "tdw_mat/LLM/prompts/plan_nocom.csv",
    },
    "comm_wo_protocol": {
        "experiment_name": "noise_free",
        "run_id": "comm_wo_protocol",
        "communication": True,
        "token_comm": False,
        "enable_noise": False,
        "prompt_template_path": "tdw_mat/LLM/prompts/plan_com.csv",
    },
    "comm_w_protocol": {
        "experiment_name": "noise_free",
        "run_id": "comm_w_protocol",
        "communication": True,
        "token_comm": True,
        "enable_noise": False,
        "prompt_template_path": "tdw_mat/LLM/prompts/plan_com.csv",
    },
}


def bool_flag(value: bool) -> str:
    return "true" if value else "false"


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Reproduce the paper experiment settings."
    )
    parser.add_argument(
        "setting",
        choices=("no_comm", "comm_wo_protocol", "comm_w_protocol", "noise_snr_grid"),
        help="Which paper setting to run.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python interpreter used to launch the experiment.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base output directory.",
    )
    parser.add_argument(
        "--eval-episodes",
        nargs="+",
        type=int,
        default=ALL_EPISODES,
        help="Episode ids. Default is the full 24-episode test set.",
    )
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="SNR values for noise_snr_grid. Ignored for other settings.",
    )
    args, forwarded_args = parser.parse_known_args(argv)
    return args, forwarded_args


def build_clean_command(args: argparse.Namespace, forwarded_args: Sequence[str]) -> List[str]:
    setting = SETTINGS[args.setting]
    command = [
        args.python_exe,
        str(RUNNER_PATH),
        "--cleanup-before-first-run",
        "--output_dir",
        args.output_dir,
        "--experiment_name",
        str(setting["experiment_name"]),
        "--run_id",
        str(setting["run_id"]),
        "--communication",
        bool_flag(bool(setting["communication"])),
        "--token_comm",
        bool_flag(bool(setting["token_comm"])),
        "--enable_noise",
        bool_flag(bool(setting["enable_noise"])),
        "--prompt_template_path",
        str(setting["prompt_template_path"]),
        "--eval_episodes",
        *[str(episode) for episode in args.eval_episodes],
    ]
    command.extend(forwarded_args)
    return command


def build_noise_command(args: argparse.Namespace, forwarded_args: Sequence[str]) -> List[str]:
    command = [
        args.python_exe,
        str(SNR_GRID_PATH),
        "--python-exe",
        args.python_exe,
        "--output-dir",
        args.output_dir,
        "--episodes",
        *[str(episode) for episode in args.eval_episodes],
        "--snrs",
        *[str(snr) for snr in args.snrs],
    ]
    command.extend(forwarded_args)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args, forwarded_args = parse_args(argv or sys.argv[1:])
    if args.setting == "noise_snr_grid":
        command = build_noise_command(args, forwarded_args)
    else:
        command = build_clean_command(args, forwarded_args)

    print(subprocess.list2cmdline(command))
    result = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
