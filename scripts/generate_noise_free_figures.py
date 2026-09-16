#!/usr/bin/env python3
"""Generate IEEE Magazine style figures from noise_free evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

METHODS = [
    ("no_comm", "No Comm"),
    ("comm_wo_protocol", "Comm w/o Prot."),
    ("comm_w_protocol", "Comm w/ Prot."),
]

TASKS = ("food", "stuff")
LEGACY_BITS_PER_TOKEN = 17
ACTUAL_BITS_PER_TOKEN = 18

def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_results_root = repo_root / "results" / "noise_free"
    default_dataset_config = repo_root / "tdw_mat" / "dataset" / "dataset_test" / "test_env.json"
    default_output_dir = repo_root / "figures"

    parser = argparse.ArgumentParser(
        description="Generate figures for noise_free results."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=default_results_root,
        help="Root directory containing no_comm, comm_wo_protocol, and comm_w_protocol.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=default_dataset_config,
        help="Path to dataset_test/test_env.json for scenario ordering.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where generated figures are written.",
    )
    return parser.parse_args()

def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def scenario_label(entry: dict) -> str:
    return f"{entry['scene']}-{entry['layout']}"

def load_eval_results(method_dir: Path) -> Dict[int, dict]:
    episode_results: Dict[int, dict] = {}
    eval_result_path = method_dir / "eval_result.json"
    if eval_result_path.exists():
        eval_result = load_json(eval_result_path)
        for key, value in eval_result.get("episode_results", {}).items():
            try:
                episode_idx = int(key)
            except (TypeError, ValueError):
                continue
            episode_results[episode_idx] = value

    for child in sorted(method_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        try:
            episode_idx = int(child.name)
        except ValueError:
            continue
        result_path = child / "result_episode.json"
        if result_path.exists():
            episode_results[episode_idx] = load_json(result_path)

    return episode_results

def get_comm_wo_protocol_bits_per_token(comm_stats: dict, method_dir: Path, episode_idx: int) -> int:
    agents = comm_stats.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError(f"Missing communication_stats.agents for {method_dir.name} episode {episode_idx}.")

    bits_per_token_values = {
        int(agent["channel_bits_per_token"])
        for agent in agents
        if isinstance(agent, dict) and agent.get("channel_bits_per_token") is not None
    }
    if not bits_per_token_values:
        raise ValueError(f"Missing agent channel_bits_per_token for {method_dir.name} episode {episode_idx}.")
    if len(bits_per_token_values) != 1:
        raise ValueError(f"Inconsistent channel_bits_per_token for {method_dir.name} episode {episode_idx}: {sorted(bits_per_token_values)}")

    bits_per_token = next(iter(bits_per_token_values))
    if bits_per_token != LEGACY_BITS_PER_TOKEN:
        raise ValueError(f"Expected channel_bits_per_token={LEGACY_BITS_PER_TOKEN} for {method_dir.name} episode {episode_idx}, got {bits_per_token}.")
    return bits_per_token

def adjust_comm_wo_protocol_bits(bits: Optional[int], comm_stats: dict, method_dir: Path, episode_idx: int) -> Optional[int]:
    if bits is None:
        return None
    bits_per_token = get_comm_wo_protocol_bits_per_token(comm_stats, method_dir, episode_idx)
    if bits % bits_per_token != 0:
        raise ValueError(f"encoded_total_bits={bits} is not divisible by {bits_per_token} for {method_dir.name} episode {episode_idx}.")
    return bits // bits_per_token * ACTUAL_BITS_PER_TOKEN

def extract_metrics(method_dir: Path, episode_idx: int, episode_payload: Optional[dict]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    steps = None
    tokens = None
    bits = None

    if episode_payload:
        frames_value = episode_payload.get("frame") or episode_payload.get("frames") or episode_payload.get("steps")
        if frames_value is not None:
            steps = int(round(frames_value))
        comm_stats = episode_payload.get("communication_stats", {})
        tokens_value = comm_stats.get("encoded_total_tokens")
        if tokens_value is not None:
            tokens = int(round(tokens_value))
        bits_value = comm_stats.get("encoded_total_bits")
        if bits_value is not None:
            bits = int(round(bits_value))
        if method_dir.name == "comm_wo_protocol":
            bits = adjust_comm_wo_protocol_bits(bits, comm_stats, method_dir, episode_idx)

    if method_dir.name == "no_comm":
        tokens = None
        bits = None

    return steps, tokens, bits

def collect_results(results_root: Path, scenarios: List[dict]) -> Dict[str, Dict[int, dict]]:
    collected: Dict[str, Dict[int, dict]] = {}
    for method_key, _ in METHODS:
        method_dir = results_root / method_key
        episode_payloads = load_eval_results(method_dir) if method_dir.exists() else {}
        method_results: Dict[int, dict] = {}
        for episode_idx, scenario in enumerate(scenarios):
            payload = episode_payloads.get(episode_idx)
            steps, tokens, bits = extract_metrics(method_dir, episode_idx, payload)
            method_results[episode_idx] = {
                "task": scenario["task"],
                "scene_label": scenario_label(scenario),
                "steps": steps,
                "tokens": tokens,
                "bits": bits,
            }
        collected[method_key] = method_results
    return collected

def setup_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 18,
        "axes.titlesize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 16,
        "figure.titlesize": 16,
        "axes.grid": True,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
    })

COLORS = {
    "no_comm": "#B3B3B3",         # Light Gray
    "comm_wo_protocol": "#8DA0CB", # Soft Blue
    "comm_w_protocol": "#FC8D62",  # Soft Orange
}

def plot_bar_charts(task: str, scenarios: List[Tuple[int, dict]], collected: Dict[str, Dict[int, dict]], output_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    ax_ts, ax_bc = axes

    # task_prefix = "F" if task == "food" else "S"
    task_prefix = ''
    labels = [f"{task_prefix}{i+1}" for i in range(len(scenarios))]
    x = np.arange(len(labels))
    width = 0.25

    # TS subplot
    no_comm_ts = [collected["no_comm"][ep_idx]["steps"] or 0 for ep_idx, _ in scenarios]
    wo_prot_ts = [collected["comm_wo_protocol"][ep_idx]["steps"] or 0 for ep_idx, _ in scenarios]
    w_prot_ts = [collected["comm_w_protocol"][ep_idx]["steps"] or 0 for ep_idx, _ in scenarios]

    ax_ts.bar(x - width, no_comm_ts, width, label='No Comm', color=COLORS["no_comm"], edgecolor='black', linewidth=0.5)
    ax_ts.bar(x, wo_prot_ts, width, label='Comm w/o Prot.', color=COLORS["comm_wo_protocol"], edgecolor='black', linewidth=0.5)
    ax_ts.bar(x + width, w_prot_ts, width, label='Comm w/ Prot.', color=COLORS["comm_w_protocol"], edgecolor='black', linewidth=0.5)

    ax_ts.set_ylabel('Transport Steps (TS)')
    # ax_ts.set_title(f'Transport Steps for {task.capitalize()} Scenarios')
    ax_ts.set_xticks(x)
    ax_ts.set_xticklabels(labels)
    ax_ts.legend(loc=1)

    # BC subplot
    wo_prot_bc = [collected["comm_wo_protocol"][ep_idx]["bits"] or 0 for ep_idx, _ in scenarios]
    w_prot_bc = [collected["comm_w_protocol"][ep_idx]["bits"] or 0 for ep_idx, _ in scenarios]

    width_bc = 0.35
    ax_bc.bar(x - width_bc/2, wo_prot_bc, width_bc, label='Comm w/o Prot.', color=COLORS["comm_wo_protocol"], edgecolor='black', linewidth=0.5)
    ax_bc.bar(x + width_bc/2, w_prot_bc, width_bc, label='Comm w/ Prot.', color=COLORS["comm_w_protocol"], edgecolor='black', linewidth=0.5)

    ax_bc.set_ylabel('Bit Consumption (BC)')
    # ax_bc.set_title(f'Bit Consumption for {task.capitalize()} Scenarios')
    ax_bc.set_xticks(x)
    ax_bc.set_xticklabels(labels)
    ax_bc.set_yscale('log')
    ax_bc.legend()

    plt.tight_layout()
    output_path = output_dir / f"{task}_bar_charts.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")

def average(values: Iterable[Optional[float]]) -> float:
    numeric_values = [v for v in values if v is not None]
    return mean(numeric_values) if numeric_values else 0.0

def plot_pareto(scenarios: List[dict], collected: Dict[str, Dict[int, dict]], output_dir: Path):
    # Calculate overall averages
    no_comm_ts_all = []
    wo_prot_ts_all, wo_prot_bc_all = [], []
    w_prot_ts_all, w_prot_bc_all = [], []

    for ep_idx, _ in enumerate(scenarios):
        if collected["no_comm"][ep_idx]["steps"] is not None:
            no_comm_ts_all.append(collected["no_comm"][ep_idx]["steps"])
            
        if collected["comm_wo_protocol"][ep_idx]["steps"] is not None:
            wo_prot_ts_all.append(collected["comm_wo_protocol"][ep_idx]["steps"])
        if collected["comm_wo_protocol"][ep_idx]["bits"] is not None:
            wo_prot_bc_all.append(collected["comm_wo_protocol"][ep_idx]["bits"])
            
        if collected["comm_w_protocol"][ep_idx]["steps"] is not None:
            w_prot_ts_all.append(collected["comm_w_protocol"][ep_idx]["steps"])
        if collected["comm_w_protocol"][ep_idx]["bits"] is not None:
            w_prot_bc_all.append(collected["comm_w_protocol"][ep_idx]["bits"])

    avg_no_comm_ts = average(no_comm_ts_all)
    
    avg_wo_prot_ts = average(wo_prot_ts_all)
    avg_wo_prot_bc = average(wo_prot_bc_all)
    
    avg_w_prot_ts = average(w_prot_ts_all)
    avg_w_prot_bc = average(w_prot_bc_all)

    # Plot Pareto
    fig, ax = plt.subplots(figsize=(8, 6))

    # Upper bound (No Comm)
    ax.axhline(y=avg_no_comm_ts, color='black', linestyle='--', linewidth=2, label=f'No Comm')
    # COLORS["no_comm"]

    # Points
    ax.scatter(avg_wo_prot_bc, avg_wo_prot_ts, color=COLORS["comm_wo_protocol"], s=150, zorder=5, label='Comm w/o Prot.', edgecolor='black')
    ax.scatter(avg_w_prot_bc, avg_w_prot_ts, color=COLORS["comm_w_protocol"], s=150, zorder=5, label='Comm w/ Prot.', edgecolor='black', marker='^')

    # Annotations (Values)
    ax.annotate(f'TS: {avg_wo_prot_ts:.2f}\nBC: {avg_wo_prot_bc:.2f}', 
                (avg_wo_prot_bc, avg_wo_prot_ts), 
                textcoords="offset points", xytext=(15, -10), ha='right', va='top', fontsize=18)
    
    ax.annotate(f'TS: {avg_w_prot_ts:.2f}\nBC: {avg_w_prot_bc:.2f}', 
                (avg_w_prot_bc, avg_w_prot_ts), 
                textcoords="offset points", xytext=(-15, -10), ha='left', va='top', fontsize=18)

    # Reduction Annotations
    wo_prot_ts_reduction = (avg_no_comm_ts - avg_wo_prot_ts) / avg_no_comm_ts * 100
    w_prot_ts_reduction = (avg_no_comm_ts - avg_w_prot_ts) / avg_no_comm_ts * 100
    bc_reduction = (avg_wo_prot_bc - avg_w_prot_bc) / avg_wo_prot_bc * 100

    # Draw lines for TS reduction
    ax.annotate('', xy=(avg_wo_prot_bc, avg_wo_prot_ts), xytext=(avg_wo_prot_bc, avg_no_comm_ts),
                arrowprops=dict(arrowstyle="->", color="gray", linestyle="--", shrinkA=0, shrinkB=6))
    ax.annotate(f"-{wo_prot_ts_reduction:.2f}% TS", xy=(avg_wo_prot_bc, (avg_no_comm_ts + avg_wo_prot_ts) / 2),
                xytext=(-2, 4), textcoords="offset points", va='center', ha='right', color="r", fontsize=18)

    ax.annotate('', xy=(avg_w_prot_bc, avg_w_prot_ts), xytext=(avg_w_prot_bc, avg_no_comm_ts),
                arrowprops=dict(arrowstyle="->", color="gray", linestyle="--", shrinkA=0, shrinkB=6))
    ax.annotate(f"-{w_prot_ts_reduction:.2f}% TS", xy=(avg_w_prot_bc, (avg_no_comm_ts + avg_w_prot_ts) / 2),
                xytext=(5, 1), textcoords="offset points", va='center', ha='left', color="r", fontsize=18)

    # Draw line for BC reduction (between the two points)
    ax.annotate('', xy=(avg_w_prot_bc, avg_w_prot_ts), xytext=(avg_wo_prot_bc, avg_wo_prot_ts),
                arrowprops=dict(arrowstyle="->", color="gray", linestyle="--", shrinkA=8, shrinkB=8))
    mid_bc = 10**((np.log10(avg_w_prot_bc) + np.log10(avg_wo_prot_bc)) / 2)
    mid_ts = (avg_w_prot_ts + avg_wo_prot_ts) / 2
    ax.annotate(f"-{bc_reduction:.2f}% BC", xy=(mid_bc, mid_ts), xytext=(0, 10), textcoords="offset points", 
                va='bottom', ha='center', color="r", fontsize=18)

    ax.set_xscale('log')
    ax.set_xlim(100, 10000)
    
    from matplotlib.ticker import LogLocator
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 1.0))
    
    ax.grid(True, which='major', linestyle='--', alpha=0.6)
    ax.grid(True, which='minor', linestyle=':', alpha=0.4)

    ax.set_xlabel('Average Bit Consumption (BC)')
    ax.set_ylabel('Average Transport Steps (TS)')
    # ax.set_title('Pareto-Style Trade-off (24 Scenarios Average)')
    
    # Optional: adjust limits to make it look nice
    ax.set_ylim(bottom=min(avg_w_prot_ts, avg_wo_prot_ts) * 0.85, top=avg_no_comm_ts * 1.05)

    ax.legend()
    plt.tight_layout()
    output_path = output_dir / "pareto_tradeoff.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    scenarios = load_json(args.dataset_config)
    collected = collect_results(args.results_root, scenarios)

    setup_plot_style()

    for task in TASKS:
        scoped_scenarios = [
            (episode_idx, scenario)
            for episode_idx, scenario in enumerate(scenarios)
            if scenario.get("task") == task
        ]
        plot_bar_charts(task, scoped_scenarios, collected, args.output_dir)

    plot_pareto(scenarios, collected, args.output_dir)

if __name__ == "__main__":
    main()
