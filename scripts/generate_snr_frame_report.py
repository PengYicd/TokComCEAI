#!/usr/bin/env python3
"""Generate IEEE-style LaTeX and plots for noise challenge frame-vs-SNR results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_results_root = repo_root / "results" / "noise_challenge"
    default_ter_csv = repo_root / "channel" / "snr_ber_ter.csv"
    default_output_dir = repo_root / "figures"

    parser = argparse.ArgumentParser(
        description="Generate an IEEE-style LaTeX table and SNR-frame plot for noise_challenge results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root,
        help="Root directory containing per-SNR evaluation outputs.",
    )
    parser.add_argument(
        "--ter-csv",
        type=Path,
        default=default_ter_csv,
        help="CSV file containing snr_db and ter columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where generated files are written.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=800,
        help="DPI for the generated figure.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_integer(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{int(round(value)):,}"


def format_average(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}"


def load_ter_by_snr(csv_path: Path) -> Dict[int, float]:
    ter_by_snr: Dict[int, float] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            snr_text = row.get("snr_db")
            ter_text = row.get("ter")
            if not snr_text or not ter_text:
                continue
            snr = int(round(float(snr_text)))
            ter_by_snr[snr] = float(ter_text)
    return ter_by_snr


def extract_frame(payload: dict) -> Optional[int]:
    for key in ("frame", "frames", "steps"):
        value = payload.get(key)
        if value is not None:
            return int(round(value))
    return None


def load_snr_frames(results_root: Path) -> Dict[int, Dict[int, Optional[int]]]:
    snr_to_episode_frames: Dict[int, Dict[int, Optional[int]]] = {}

    snr_dirs = []
    for child in results_root.iterdir():
        if not child.is_dir():
            continue
        try:
            snr = int(child.name)
        except ValueError:
            continue
        snr_dirs.append((snr, child))

    for snr, snr_dir in sorted(snr_dirs, key=lambda item: item[0]):
        episode_frames: Dict[int, Optional[int]] = {}

        eval_result_path = snr_dir / "eval_result.json"
        if eval_result_path.exists():
            eval_result = load_json(eval_result_path)
            for key, payload in eval_result.get("episode_results", {}).items():
                try:
                    episode = int(key)
                except (TypeError, ValueError):
                    continue
                if not (0 <= episode <= 23):
                    continue
                if isinstance(payload, dict):
                    episode_frames[episode] = extract_frame(payload)

        for child in sorted(snr_dir.iterdir(), key=lambda path: path.name):
            if not child.is_dir():
                continue
            try:
                episode = int(child.name)
            except ValueError:
                continue
            if not (0 <= episode <= 23):
                continue
            result_path = child / "result_episode.json"
            if result_path.exists():
                episode_frames[episode] = extract_frame(load_json(result_path))

        snr_to_episode_frames[snr] = episode_frames

    return snr_to_episode_frames


def collect_episode_ids(snr_to_episode_frames: Dict[int, Dict[int, Optional[int]]]) -> List[int]:
    episode_ids = set()
    for episode_frames in snr_to_episode_frames.values():
        episode_ids.update(episode_frames.keys())
    return sorted(episode_ids)


def compute_summary(
    snr_to_episode_frames: Dict[int, Dict[int, Optional[int]]], ter_by_snr: Dict[int, float]
) -> List[dict]:
    summary_rows: List[dict] = []
    for snr in sorted(snr_to_episode_frames):
        frames = [
            frame
            for frame in snr_to_episode_frames[snr].values()
            if frame is not None
        ]
        frame_mean = mean(frames) if frames else None
        frame_std = stdev(frames) if len(frames) >= 2 else 0.0 if frames else None
        summary_rows.append(
            {
                "snr": snr,
                "n_episodes": len(frames),
                "frame_mean": frame_mean,
                "frame_std": frame_std,
                "ter": ter_by_snr.get(snr),
            }
        )
    return summary_rows


def build_latex_table(
    snr_to_episode_frames: Dict[int, Dict[int, Optional[int]]],
    episode_ids: List[int],
) -> str:
    column_spec = "l" + "r" * len(episode_ids) + "rr"
    episode_headers = [f"E{episode}" for episode in episode_ids]
    caption = (
        "Per-SNR frame counts for all available noise-challenge episodes. "
        "Smaller frame counts indicate better efficiency. "
        "Missing runs are marked with -; averages are computed over available episodes only."
    )
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:noise_challenge_snr_frames}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "SNR (dB) & " + " & ".join(episode_headers) + r" & Avg. & Std. \\",
        r"\midrule",
    ]

    for snr in sorted(snr_to_episode_frames):
        episode_frames = snr_to_episode_frames[snr]
        ordered_frames = [episode_frames.get(episode) for episode in episode_ids]
        valid_frames = [frame for frame in ordered_frames if frame is not None]
        row = [str(snr)]
        row.extend(format_integer(frame) for frame in ordered_frames)
        row.append(format_average(mean(valid_frames) if valid_frames else None))
        row.append(
            format_average(stdev(valid_frames) if len(valid_frames) >= 2 else 0.0 if valid_frames else None)
        )
        lines.append(" & ".join(row) + r"\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def write_summary_csv(
    output_path: Path,
    summary_rows: List[dict],
    snr_to_episode_frames: Dict[int, Dict[int, Optional[int]]],
    episode_ids: List[int],
) -> None:
    fieldnames = ["snr_db", "n_episodes", "frame_mean", "frame_std", "ter"]
    fieldnames.extend(f"episode_{episode}_frame" for episode in episode_ids)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            snr = row["snr"]
            csv_row = {
                "snr_db": snr,
                "n_episodes": row["n_episodes"],
                "frame_mean": "" if row["frame_mean"] is None else f"{row['frame_mean']:.6f}",
                "frame_std": "" if row["frame_std"] is None else f"{row['frame_std']:.6f}",
                "ter": "" if row["ter"] is None else f"{row['ter']:.12f}",
            }
            for episode in episode_ids:
                value = snr_to_episode_frames[snr].get(episode)
                csv_row[f"episode_{episode}_frame"] = "" if value is None else int(value)
            writer.writerow(csv_row)


def make_plot(
    output_path: Path,
    summary_rows: List[dict],
    snr_to_episode_frames: Dict[int, Dict[int, Optional[int]]],
    episode_ids: List[int],
    figure_dpi: int,
) -> None:
    valid_rows = [row for row in summary_rows if row["frame_mean"] is not None]
    if not valid_rows:
        raise ValueError("No valid frame statistics are available for plotting.")

    snrs = [row["snr"] for row in valid_rows]
    mean_frames = [row["frame_mean"] for row in valid_rows]
    ters = [row["ter"] for row in valid_rows]

    fig, ax_frame = plt.subplots(figsize=(8, 6))
    episode_cmap = plt.get_cmap("tab20", max(len(episode_ids), 1))
    for color_index, episode in enumerate(episode_ids):
        points = []
        for snr in snrs:
            frame = snr_to_episode_frames.get(snr, {}).get(episode)
            if frame is not None:
                points.append((snr, frame))
        if not points:
            continue

        point_snrs = [point[0] for point in points]
        point_frames = [point[1] for point in points]
        ax_frame.scatter(
            point_snrs,
            point_frames,
            s=28,
            color=episode_cmap(color_index),
            alpha=0.8,
            edgecolors="none",
        )

    ax_frame.plot(
        snrs,
        mean_frames,
        color="black",
        marker="o",
        markersize=5,
        linewidth=2.0,
        zorder=3,
    )
    ax_frame.set_xlabel("SNR (dB)", fontsize=24)
    ax_frame.set_ylabel("Transport Steps (TS)", fontsize=24)
    ax_frame.tick_params(axis='both', which='major', labelsize=20)
    ax_frame.set_xticks(snrs)
    ax_frame.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_frame.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.6)

    ax_top = ax_frame.secondary_xaxis("top")
    ax_top.set_xticks(snrs)
    ter_labels = [f"{ter:.4g}" if ter is not None else "-" for ter in ters]
    ax_top.set_xticklabels(ter_labels, fontsize=20)
    ax_top.set_xlabel("Token Error Rate (TER)", fontsize=24)

    fig.tight_layout()
    fig.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.results_root.exists():
        raise FileNotFoundError(f"Results root not found: {args.results_root}")
    if not args.ter_csv.exists():
        raise FileNotFoundError(f"TER csv not found: {args.ter_csv}")

    snr_to_episode_frames = load_snr_frames(args.results_root)
    if not snr_to_episode_frames:
        raise ValueError(f"No per-SNR results found in {args.results_root}")

    ter_by_snr = load_ter_by_snr(args.ter_csv)
    episode_ids = collect_episode_ids(snr_to_episode_frames)
    summary_rows = compute_summary(snr_to_episode_frames, ter_by_snr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "snr_frame_table.tex"
    summary_csv_path = args.output_dir / "snr_frame_summary.csv"
    figure_path = args.output_dir / "snr_frame_vs_snr.svg"

    table_tex = build_latex_table(snr_to_episode_frames, episode_ids)
    table_path.write_text(table_tex + "\n", encoding="utf-8")
    write_summary_csv(summary_csv_path, summary_rows, snr_to_episode_frames, episode_ids)
    make_plot(
        figure_path,
        summary_rows,
        snr_to_episode_frames,
        episode_ids,
        args.figure_dpi,
    )

    print("% Requires: \\usepackage{booktabs,graphicx}")
    print("% Missing episode results are shown as '-' and excluded from averages/std.")
    print("% Generated from results/noise_challenge")
    print()
    print(f"% Table written to: {table_path.as_posix()}")
    print(f"% Summary written to: {summary_csv_path.as_posix()}")
    print(f"% Figure written to: {figure_path.as_posix()}")
    print()
    print(table_tex)


if __name__ == "__main__":
    main()
