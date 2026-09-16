#!/usr/bin/env python3
"""Run challenge.py and automatically retry after abnormal exits.

This wrapper is intended for the TDW challenge workflow on Windows:
- Start `tdw_mat/tdw-gym/challenge.py` with the current Python interpreter.
- If the child process exits with code 0, stop immediately.
- If the child process exits abnormally, close the lingering TDW window/process,
  wait for a short delay, and start `challenge.py` again.

Unknown command-line arguments are forwarded to `challenge.py`, so you can use
the same flags that you normally pass to the challenge entrypoint.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent
CHALLENGE_PATH = REPO_ROOT / "tdw_mat" / "tdw-gym" / "challenge.py"
DEFAULT_PORT = 10999


def parse_args(argv: Sequence[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically rerun tdw_mat/tdw-gym/challenge.py after abnormal exits."
        )
    )
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=None,
        help="Maximum number of retries after failures. Omit to retry forever.",
    )
    parser.add_argument(
        "--restart-delay",
        type=float,
        default=5.0,
        help="Seconds to wait after cleanup before restarting challenge.py.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python interpreter used to run challenge.py.",
    )
    parser.add_argument(
        "--cleanup-before-first-run",
        action="store_true",
        help="Kill TDW before the first run as well.",
    )
    parser.add_argument(
        "--cleanup-on-success",
        dest="cleanup_on_success",
        action="store_true",
        help="Kill lingering TDW after a successful run (default).",
    )
    parser.add_argument(
        "--no-cleanup-on-success",
        dest="cleanup_on_success",
        action="store_false",
        help="Do not kill TDW after a successful run.",
    )
    parser.add_argument(
        "--kill-all-tdw",
        dest="kill_all_tdw",
        action="store_true",
        help="Kill all TDW.exe processes after a failed run (default).",
    )
    parser.add_argument(
        "--no-kill-all-tdw",
        dest="kill_all_tdw",
        action="store_false",
        help="Only try to close the TDW window that matches the selected port.",
    )
    parser.set_defaults(kill_all_tdw=True, cleanup_on_success=True)
    wrapper_args, challenge_args = parser.parse_known_args(argv)
    return wrapper_args, challenge_args


def extract_port(challenge_args: Sequence[str]) -> int:
    for index, arg in enumerate(challenge_args):
        if arg == "--port" and index + 1 < len(challenge_args):
            try:
                return int(challenge_args[index + 1])
            except ValueError:
                return DEFAULT_PORT
        if arg.startswith("--port="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return DEFAULT_PORT
    return DEFAULT_PORT


def run_quiet_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
        check=False,
    )


def cleanup_tdw(port: int, kill_all_tdw: bool) -> None:
    if os.name != "nt":
        print("[cleanup] Non-Windows platform detected, skipping TDW taskkill cleanup.")
        return

    commands: List[Sequence[str]] = [
        [
            "taskkill",
            "/F",
            "/T",
            "/FI",
            f"WINDOWTITLE eq TDW {port}",
        ]
    ]
    if kill_all_tdw:
        commands.append(["taskkill", "/F", "/T", "/IM", "TDW.exe"])

    for command in commands:
        result = run_quiet_command(command)
        output = (result.stdout or "").strip()
        joined = subprocess.list2cmdline(list(command))
        if output:
            print(f"[cleanup] {joined}\n{output}")
        else:
            print(f"[cleanup] {joined} -> exit code {result.returncode}")

    time.sleep(2.0)


def build_challenge_command(python_exe: str, challenge_args: Sequence[str]) -> List[str]:
    return [python_exe, str(CHALLENGE_PATH), *challenge_args]


def attempt_limit_reached(attempt: int, max_restarts: int | None) -> bool:
    if max_restarts is None:
        return False
    return attempt >= max_restarts + 1


def log_attempt_header(attempt: int, command: Iterable[str]) -> None:
    command_text = subprocess.list2cmdline(list(command))
    print("=" * 80)
    print(f"[runner] Starting attempt #{attempt}")
    print(f"[runner] Working directory: {REPO_ROOT}")
    print(f"[runner] Command: {command_text}")
    print("=" * 80)


def main(argv: Sequence[str] | None = None) -> int:
    wrapper_args, challenge_args = parse_args(argv or sys.argv[1:])
    if not CHALLENGE_PATH.exists():
        print(f"[runner] challenge.py not found: {CHALLENGE_PATH}", file=sys.stderr)
        return 2

    port = extract_port(challenge_args)
    challenge_command = build_challenge_command(wrapper_args.python_exe, challenge_args)

    if wrapper_args.cleanup_before_first_run:
        print(f"[runner] Cleaning TDW before first run on port {port}.")
        cleanup_tdw(port=port, kill_all_tdw=wrapper_args.kill_all_tdw)

    attempt = 1
    while True:
        log_attempt_header(attempt, challenge_command)
        try:
            result = subprocess.run(
                challenge_command,
                cwd=str(REPO_ROOT),
                check=False,
            )
        except KeyboardInterrupt:
            print("\n[runner] Interrupted by user, stopping wrapper.")
            return 130

        if result.returncode == 0:
            print(f"[runner] challenge.py finished successfully on attempt #{attempt}.")
            if wrapper_args.cleanup_on_success:
                print(f"[runner] Cleaning TDW after successful run on port {port}.")
                cleanup_tdw(port=port, kill_all_tdw=wrapper_args.kill_all_tdw)
            return 0

        print(
            f"[runner] challenge.py exited abnormally with code {result.returncode}. "
            "Cleaning TDW before restart."
        )
        cleanup_tdw(port=port, kill_all_tdw=wrapper_args.kill_all_tdw)

        if attempt_limit_reached(attempt, wrapper_args.max_restarts):
            print("[runner] Reached retry limit, stop restarting.")
            return result.returncode
        attempt += 1

        if wrapper_args.restart_delay > 0:
            print(f"[runner] Waiting {wrapper_args.restart_delay:.1f}s before restart.")
            time.sleep(wrapper_args.restart_delay)


if __name__ == "__main__":
    raise SystemExit(main())
