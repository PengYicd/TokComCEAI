#!/usr/bin/env python3
"""Generate IEEE-style LaTeX tables from noise_free evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


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
    default_output_dir = default_results_root / "latex_tables"

    parser = argparse.ArgumentParser(
        description="Generate two LaTeX tables for noise_free results."
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
        help="Directory where generated .tex files are written.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def format_integer(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{int(round(value)):,}"


def format_average(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.2f}"


def compute_reduction_percent(baseline: Optional[float], current: Optional[float]) -> Optional[float]:
    if baseline in (None, 0) or current is None:
        return None
    return (baseline - current) / baseline * 100.0


def format_average_with_delta(value: Optional[float], reduction_percent: Optional[float]) -> str:
    formatted_value = format_average(value)
    if reduction_percent is None:
        return formatted_value
    return f"{formatted_value} (-{reduction_percent:.2f}\\%)"


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
        raise ValueError(
            f"Missing communication_stats.agents for {method_dir.name} episode {episode_idx}."
        )

    bits_per_token_values = {
        int(agent["channel_bits_per_token"])
        for agent in agents
        if isinstance(agent, dict) and agent.get("channel_bits_per_token") is not None
    }
    if not bits_per_token_values:
        raise ValueError(
            f"Missing agent channel_bits_per_token for {method_dir.name} episode {episode_idx}."
        )
    if len(bits_per_token_values) != 1:
        raise ValueError(
            f"Inconsistent channel_bits_per_token for {method_dir.name} episode {episode_idx}: "
            f"{sorted(bits_per_token_values)}"
        )

    bits_per_token = next(iter(bits_per_token_values))
    if bits_per_token != LEGACY_BITS_PER_TOKEN:
        raise ValueError(
            f"Expected channel_bits_per_token={LEGACY_BITS_PER_TOKEN} for {method_dir.name} "
            f"episode {episode_idx}, got {bits_per_token}."
        )
    return bits_per_token


def adjust_comm_wo_protocol_bits(bits: Optional[int], comm_stats: dict, method_dir: Path, episode_idx: int) -> Optional[int]:
    if bits is None:
        return None

    bits_per_token = get_comm_wo_protocol_bits_per_token(comm_stats, method_dir, episode_idx)
    if bits % bits_per_token != 0:
        raise ValueError(
            f"encoded_total_bits={bits} is not divisible by {bits_per_token} for "
            f"{method_dir.name} episode {episode_idx}."
        )
    return bits // bits_per_token * ACTUAL_BITS_PER_TOKEN


def extract_metrics(
    method_dir: Path, episode_idx: int, episode_payload: Optional[dict]
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    steps = None
    tokens = None
    bits = None

    if episode_payload:
        frames_value = episode_payload.get("frame")
        if frames_value is None:
            frames_value = episode_payload.get("frames")
        if frames_value is None:
            frames_value = episode_payload.get("steps")
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


def average(values: Iterable[Optional[float]]) -> Optional[float]:
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return mean(numeric_values)


def build_table(task: str, scenarios: List[Tuple[int, dict]], collected: Dict[str, Dict[int, dict]]) -> str:
    task_title = task.capitalize()
    task_prefix = "F" if task == "food" else "S"
    label = f"tab:noise_free_{task}"
    caption = (
        f"Per-scenario step count, transmitted token count, and encoded communication overhead on the {task_title} subset "
        f"under the noise-free setting. Smaller step counts indicate better efficiency. "
        f"Average cells report two decimal places; annotated percentages indicate relative reductions."
    )
    column_spec = "ll" + "r" * (len(scenarios) + 1)
    case_headers = [f"{task_prefix}{local_idx}" for local_idx, _ in enumerate(scenarios, start=1)]

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Method & Metric & " + " & ".join(case_headers) + r" & Avg. \\",
        r"\midrule",
    ]

    no_comm_rows = [collected["no_comm"][episode_idx] for episode_idx, _ in scenarios]
    no_comm_step_cells = [format_integer(row["steps"]) for row in no_comm_rows]
    no_comm_step_avg_value = average(row["steps"] for row in no_comm_rows)
    no_comm_step_avg = format_average(no_comm_step_avg_value)
    lines.append("No Comm & Steps & " + " & ".join(no_comm_step_cells) + f" & {no_comm_step_avg} " + r"\\")
    lines.append(r"\midrule")

    for method_key, method_label in METHODS[1:]:
        rows = [collected[method_key][episode_idx] for episode_idx, _ in scenarios]
        step_cells = [format_integer(row["steps"]) for row in rows]
        token_cells = [format_integer(row["tokens"]) for row in rows]
        bits_cells = [format_integer(row["bits"]) for row in rows]
        step_avg_value = average(row["steps"] for row in rows)
        token_avg_value = average(row["tokens"] for row in rows)
        bits_avg_value = average(row["bits"] for row in rows)
        step_reduction = compute_reduction_percent(no_comm_step_avg_value, step_avg_value)
        step_avg = format_average_with_delta(step_avg_value, step_reduction)
        token_avg = format_average(token_avg_value)

        bits_reduction = None
        if method_key == "comm_w_protocol":
            comm_wo_protocol_rows = [
                collected["comm_wo_protocol"][episode_idx] for episode_idx, _ in scenarios
            ]
            comm_wo_protocol_bits_avg = average(row["bits"] for row in comm_wo_protocol_rows)
            bits_reduction = compute_reduction_percent(comm_wo_protocol_bits_avg, bits_avg_value)
        bits_avg = format_average_with_delta(bits_avg_value, bits_reduction)
        lines.append(
            rf"\multirow{{3}}{{*}}{{{method_label}}} & Steps & "
            + " & ".join(step_cells)
            + f" & {step_avg} "
            + r"\\"
        )
        lines.append(
            " & Tokens & "
            + " & ".join(token_cells)
            + f" & {token_avg} "
            + r"\\"
        )
        lines.append(
            " & Bits & "
            + " & ".join(bits_cells)
            + f" & {bits_avg} "
            + r"\\"
        )
        if method_key != METHODS[-1][0]:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )

    return "\n".join(lines)


def write_output(output_dir: Path, task: str, table_tex: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{task}_noise_free_table.tex"
    output_path.write_text(table_tex + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    scenarios = load_json(args.dataset_config)
    collected = collect_results(args.results_root, scenarios)

    print("% Requires: \\usepackage{booktabs,multirow,graphicx}")
    print("% Steps columns are populated from per-episode frame counts for paper reporting.")
    print("% Generated from results/noise_free")
    print()

    for task in TASKS:
        scoped_scenarios = [
            (episode_idx, scenario)
            for episode_idx, scenario in enumerate(scenarios)
            if scenario.get("task") == task
        ]
        table_tex = build_table(task, scoped_scenarios, collected)
        output_path = write_output(args.output_dir, task, table_tex)
        print(f"% ===== {task.upper()} TABLE: {output_path.as_posix()} =====")
        print(table_tex)
        print()


if __name__ == "__main__":
    main()
