#!/usr/bin/env python3
"""Run noisy challenge sweeps over episodes and SNRs using fixed per-episode protocols."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = REPO_ROOT / "run_challenge_with_restart.py"
SUPPORTED_SNRS = [0, 1, 2, 3, 4]
DEFAULT_SNRS = list(SUPPORTED_SNRS)
DEFAULT_EPISODES = list(range(24))


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description="Sweep noisy challenge evaluation across episodes and SNRs."
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=DEFAULT_EPISODES,
        help="Episode ids to evaluate.",
    )
    parser.add_argument(
        "--snrs",
        nargs="+",
        type=int,
        default=DEFAULT_SNRS,
        help="SNR values to evaluate. Supported values: 0 1 2 3 4.",
    )
    parser.add_argument(
        "--order",
        choices=("snr_first", "episode_first"),
        default="episode_first",
        help="Loop order for the sweep.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python interpreter used to run the restart wrapper.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Base output directory passed to challenge.py.",
    )
    parser.add_argument(
        "--experiment-name",
        default="noise_challenge",
        help="Experiment name passed to challenge.py.",
    )
    args, forwarded_args = parser.parse_known_args(argv)
    return args, forwarded_args


def format_snr(snr: int) -> str:
    return str(int(snr))


def iter_jobs(snrs: Sequence[int], episodes: Sequence[int], order: str) -> Iterable[tuple[int, int]]:
    if order == "episode_first":
        for episode in episodes:
            for snr in snrs:
                yield snr, episode
        return
    for snr in snrs:
        for episode in episodes:
            yield snr, episode


def protocol_path_for_episode(episode: int) -> Path:
    return REPO_ROOT / "results" / "noise_free" / "comm_w_protocol" / str(int(episode)) / "protocol.json"


def build_command(
    python_exe: str,
    output_dir: str,
    experiment_name: str,
    snr: int,
    episode: int,
    forwarded_args: Sequence[str],
) -> List[str]:
    protocol_path = protocol_path_for_episode(episode)
    if not protocol_path.exists():
        raise FileNotFoundError(f"protocol not found for episode {episode}: {protocol_path}")

    command = [
        python_exe,
        str(RUNNER_PATH),
        "--cleanup-before-first-run",
        "--output_dir",
        output_dir,
        "--experiment_name",
        experiment_name,
        "--run_id",
        format_snr(snr),
        "--eval_episodes",
        str(int(episode)),
        "--communication",
        "true",
        "--token_comm",
        "true",
        "--enable_noise",
        "true",
        "--comm_snr_db",
        str(int(snr)),
        "--fixed_protocol_path",
        str(protocol_path),
    ]
    command.extend(forwarded_args)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args, forwarded_args = parse_args(argv or sys.argv[1:])
    if not RUNNER_PATH.exists():
        print(f"[snr-grid] restart runner not found: {RUNNER_PATH}", file=sys.stderr)
        return 2

    supported_snrs = set(SUPPORTED_SNRS)
    for snr in args.snrs:
        if snr not in supported_snrs:
            print(f"[snr-grid] unsupported SNR: {snr}. Supported values: {sorted(supported_snrs)}", file=sys.stderr)
            return 2

    total_jobs = len(args.snrs) * len(args.episodes)
    for job_index, (snr, episode) in enumerate(iter_jobs(args.snrs, args.episodes, args.order), start=1):
        command = build_command(
            python_exe=args.python_exe,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            snr=snr,
            episode=episode,
            forwarded_args=forwarded_args,
        )
        print("=" * 80)
        print(f"[snr-grid] Job {job_index}/{total_jobs}: episode={episode}, snr={snr}")
        print(f"[snr-grid] Command: {subprocess.list2cmdline(command)}")
        print("=" * 80)
        result = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
        if result.returncode != 0:
            print(
                f"[snr-grid] job failed for episode={episode}, snr={snr}, "
                f"exit_code={result.returncode}",
                file=sys.stderr,
            )
            return result.returncode

    print("[snr-grid] all jobs completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
