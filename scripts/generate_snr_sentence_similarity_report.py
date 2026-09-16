#!/usr/bin/env python3
"""Generate IEEE-style LaTeX and plots for noise challenge sentence similarity vs SNR results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class LocalSentenceTransformer:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences, batch_size=32):
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            encoded_input = self.tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(sentence_embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_results_root = repo_root / "results" / "noise_challenge"
    default_ter_csv = repo_root / "channel" / "snr_ber_ter.csv"
    default_output_dir = repo_root / "figures"

    parser = argparse.ArgumentParser(
        description="Generate an IEEE-style LaTeX table and SNR-similarity plot for noise_challenge results."
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


def format_sim(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


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


def load_snr_sentence_similarities(results_root: Path, model: LocalSentenceTransformer) -> Dict[int, Dict[int, Optional[float]]]:
    # Pass 1: Collect all pairs
    data: Dict[int, Dict[int, List[Tuple[str, str]]]] = {}
    
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
        episode_data: Dict[int, List[Tuple[str, str]]] = {}
        for child in sorted(snr_dir.iterdir(), key=lambda path: path.name):
            if not child.is_dir():
                continue
            try:
                episode = int(child.name)
            except ValueError:
                continue
            if not (0 <= episode <= 11):
                continue
                
            log_path = child / "transmission_log.csv"
            if not log_path.exists():
                continue
                
            pending_sends = {}
            pairs = []
            with log_path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    direction = row.get("direction")
                    sender = row.get("agent_id") if direction == "send" else row.get("peer_agent_id")
                    receiver = row.get("peer_agent_id") if direction == "send" else row.get("agent_id")
                    
                    if direction == "send":
                        pending_sends[(sender, receiver)] = row
                    elif direction == "receive":
                        send_row = pending_sends.pop((sender, receiver), None)
                        if send_row is not None:
                            sent_text = send_row.get("semantic_message", "").strip()
                            recv_text = row.get("semantic_message", "").strip()
                            pairs.append((sent_text, recv_text))
            
            episode_data[episode] = pairs
        data[snr] = episode_data

    # Pass 2: Batch compute similarities
    snr_to_episode_sims: Dict[int, Dict[int, Optional[float]]] = {}
    
    for snr, episode_data in data.items():
        snr_to_episode_sims[snr] = {}
        for episode, pairs in episode_data.items():
            if not pairs:
                snr_to_episode_sims[snr][episode] = None
                continue
            
            sims = []
            to_compute_s = []
            to_compute_r = []
            to_compute_indices = []
            
            for i, (s_text, r_text) in enumerate(pairs):
                if not r_text or r_text == "[mask]" * len(r_text.split()):
                    # If received text is empty or just masks
                    # Actually user said "如果有 全 [mask] 的发送信号，我们原始是当成 接收端没有收到，即不解码，这时 sentence similarity 记为 0"
                    # But the semantic_message in log for all masks is usually empty string.
                    # We just check if it's empty.
                    if not r_text:
                        sims.append(0.0)
                    else:
                        sims.append(0.0) # placeholder
                        to_compute_s.append(s_text)
                        to_compute_r.append(r_text)
                        to_compute_indices.append(i)
                else:
                    sims.append(0.0) # placeholder
                    to_compute_s.append(s_text)
                    to_compute_r.append(r_text)
                    to_compute_indices.append(i)
                    
            if to_compute_s:
                s_embs = model.encode(to_compute_s, batch_size=32)
                r_embs = model.encode(to_compute_r, batch_size=32)
                
                s_norm = np.linalg.norm(s_embs, axis=1, keepdims=True)
                r_norm = np.linalg.norm(r_embs, axis=1, keepdims=True)
                s_embs_norm = s_embs / np.where(s_norm == 0, 1, s_norm)
                r_embs_norm = r_embs / np.where(r_norm == 0, 1, r_norm)
                cos_sims = np.sum(s_embs_norm * r_embs_norm, axis=1)
                
                for idx, sim in zip(to_compute_indices, cos_sims):
                    # Clamp sim between 0 and 1
                    sim_val = max(0.0, min(1.0, float(sim)))
                    sims[idx] = sim_val
                    
            snr_to_episode_sims[snr][episode] = mean(sims) if sims else None

    return snr_to_episode_sims


def collect_episode_ids(snr_to_episode_sims: Dict[int, Dict[int, Optional[float]]]) -> List[int]:
    episode_ids = set()
    for episode_sims in snr_to_episode_sims.values():
        episode_ids.update(episode_sims.keys())
    return sorted(episode_ids)


def compute_summary(
    snr_to_episode_sims: Dict[int, Dict[int, Optional[float]]], ter_by_snr: Dict[int, float]
) -> List[dict]:
    summary_rows: List[dict] = []
    for snr in sorted(snr_to_episode_sims):
        sims = [
            sim
            for sim in snr_to_episode_sims[snr].values()
            if sim is not None
        ]
        sim_mean = mean(sims) if sims else None
        sim_std = stdev(sims) if len(sims) >= 2 else 0.0 if sims else None
        summary_rows.append(
            {
                "snr": snr,
                "n_episodes": len(sims),
                "sim_mean": sim_mean,
                "sim_std": sim_std,
                "ter": ter_by_snr.get(snr),
            }
        )
    return summary_rows


def build_latex_table(
    snr_to_episode_sims: Dict[int, Dict[int, Optional[float]]],
    episode_ids: List[int],
) -> str:
    column_spec = "l" + "r" * len(episode_ids) + "rr"
    episode_headers = [f"E{episode}" for episode in episode_ids]
    caption = (
        "Per-SNR sentence similarity for all available noise-challenge episodes. "
        "Higher similarity indicates better semantic recovery. "
        "Missing runs are marked with -; averages are computed over available episodes only."
    )
    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:noise_challenge_snr_similarity}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.1}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "SNR (dB) & " + " & ".join(episode_headers) + r" & Avg. & Std. \\",
        r"\midrule",
    ]

    for snr in sorted(snr_to_episode_sims):
        episode_sims = snr_to_episode_sims[snr]
        ordered_sims = [episode_sims.get(episode) for episode in episode_ids]
        valid_sims = [sim for sim in ordered_sims if sim is not None]
        row = [str(snr)]
        row.extend(format_sim(sim) for sim in ordered_sims)
        row.append(format_sim(mean(valid_sims) if valid_sims else None))
        row.append(
            format_sim(stdev(valid_sims) if len(valid_sims) >= 2 else 0.0 if valid_sims else None)
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
    snr_to_episode_sims: Dict[int, Dict[int, Optional[float]]],
    episode_ids: List[int],
) -> None:
    fieldnames = ["snr_db", "n_episodes", "sim_mean", "sim_std", "ter"]
    fieldnames.extend(f"episode_{episode}_sim" for episode in episode_ids)

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            snr = row["snr"]
            csv_row = {
                "snr_db": snr,
                "n_episodes": row["n_episodes"],
                "sim_mean": "" if row["sim_mean"] is None else f"{row['sim_mean']:.6f}",
                "sim_std": "" if row["sim_std"] is None else f"{row['sim_std']:.6f}",
                "ter": "" if row["ter"] is None else f"{row['ter']:.12f}",
            }
            for episode in episode_ids:
                value = snr_to_episode_sims[snr].get(episode)
                csv_row[f"episode_{episode}_sim"] = "" if value is None else f"{value:.6f}"
            writer.writerow(csv_row)


def make_plot(
    output_path: Path,
    summary_rows: List[dict],
    snr_to_episode_sims: Dict[int, Dict[int, Optional[float]]],
    episode_ids: List[int],
    figure_dpi: int,
) -> None:
    valid_rows = [row for row in summary_rows if row["sim_mean"] is not None]
    if not valid_rows:
        print("Warning: No valid similarity statistics are available for plotting.")
        return

    snrs = [row["snr"] for row in valid_rows]
    mean_sims = [row["sim_mean"] for row in valid_rows]
    ters = [row["ter"] for row in valid_rows]

    fig, ax_sim = plt.subplots(figsize=(8, 6))
    episode_cmap = plt.get_cmap("tab20", max(len(episode_ids), 1))
    for color_index, episode in enumerate(episode_ids):
        points = []
        for snr in snrs:
            sim = snr_to_episode_sims.get(snr, {}).get(episode)
            if sim is not None:
                points.append((snr, sim))
        if not points:
            continue

        point_snrs = [point[0] for point in points]
        point_sims = [point[1] for point in points]
        ax_sim.scatter(
            point_snrs,
            point_sims,
            s=28,
            color=episode_cmap(color_index),
            alpha=0.8,
            edgecolors="none",
        )

    ax_sim.plot(
        snrs,
        mean_sims,
        color="black",
        marker="o",
        markersize=5,
        linewidth=2.0,
        zorder=3,
    )
    ax_sim.set_xlabel("SNR (dB)", fontsize=24)
    ax_sim.set_ylabel("Sentence Similarity", fontsize=24)
    ax_sim.tick_params(axis='both', which='major', labelsize=20)
    ax_sim.set_xticks(snrs)
    ax_sim.grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_sim.grid(True, axis="x", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_sim.set_ylim(-0.05, 1.05)

    ax_top = ax_sim.secondary_xaxis("top")
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

    print("Loading LocalSentenceTransformer model 'all-MiniLM-L6-v2'...")
    model = LocalSentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("Processing transmission logs...")
    snr_to_episode_sims = load_snr_sentence_similarities(args.results_root, model)
    if not snr_to_episode_sims:
        raise ValueError(f"No per-SNR results found in {args.results_root}")

    ter_by_snr = load_ter_by_snr(args.ter_csv)
    episode_ids = collect_episode_ids(snr_to_episode_sims)
    summary_rows = compute_summary(snr_to_episode_sims, ter_by_snr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "snr_semantic_similarity_table.tex"
    summary_csv_path = args.output_dir / "snr_semantic_similarity_summary.csv"
    figure_path = args.output_dir / "snr_semantic_similarity_vs_snr.svg"

    table_tex = build_latex_table(snr_to_episode_sims, episode_ids)
    table_path.write_text(table_tex + "\n", encoding="utf-8")
    write_summary_csv(summary_csv_path, summary_rows, snr_to_episode_sims, episode_ids)
    make_plot(
        figure_path,
        summary_rows,
        snr_to_episode_sims,
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
