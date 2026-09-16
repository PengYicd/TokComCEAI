# Token Communication-Assisted Collaborative Embodied AI

This repository contains the code, prompt templates, and paper figures for the manuscript

**Token Communication-Assisted Collaborative Embodied Artificial Intelligence: Concepts, Framework, and Opportunities**

Peng Yi and Ying-Chang Liang

The paper is currently under review at IEEE.

The implementation studies task-adaptive token communication for two embodied agents in the TDW-MAT transport benchmark. The agents can share no messages, free-form natural language, or a compact protocol-based token sequence.

## What is included

```text
channel/                         Token channel, SNR-to-TER mapping, MATLAB simulation
tdw_mat/LLM/prompts/             All paper prompt templates
tdw_mat/LLM/LLM.py               Protocol generation, encode/decode, dialogue memory
tdw_mat/tdw-gym/                 Evaluation entrypoint and LLM agent
scripts/run_experiment.py        Paper reproduction entrypoint
example/protocol.json            Example GFM-generated protocol
figures/                         Paper figures (SVG)
run_challenge_with_restart.py    Windows-friendly TDW restart wrapper
run_noise_challenge_snr_grid.py  Noisy-channel SNR grid
```

`tdw_mat/` is a modified fork of [TDW-MAT / CoELA](https://github.com/UMass-Foundation-Model/Co-LLM-Agents). The token-communication method, noisy channel, prompt templates, and paper scripts are the new code for this manuscript. Please keep the original MIT license notices.

The following local files are **not** part of the intended GitHub release. They remain on disk for the authors, but are ignored by git:

- `transport_challenge_asset_bundles/` (large TDW assets; download separately)
- `results/` (full per-step logs and images)
- `tdw_mat/LLM/logs/`
- `paper_images/`
- `token_comm_experiment_summary.md` (internal writing notes)

## Setup

Python 3.9 is recommended.

```bash
conda create -n tokenceai python=3.9
conda activate tokenceai
pip install -r requirements.txt
pip install -e tdw_mat
```

Install [TDW](https://github.com/threedworld-mit/tdw) according to the official documentation. If you run TDW on a remote Linux server, configure the X server as described in the [TDW install guide](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/install.md).

Download the Transport Challenge asset bundles from the [original TDW-MAT release](https://drive.google.com/file/d/1us2hpJj3_u1Ti_R0OrqVDgUQbdMPUaKN/view?usp=sharing) and unzip them into `transport_challenge_asset_bundles/` at the repository root. Do not commit this folder.

Start Ollama, then optionally set:

```bash
export OLLAMA_HOST=http://localhost:11434
```

On Windows PowerShell:

```powershell
$env:OLLAMA_HOST="http://localhost:11434"
```

A smoke-test scene from the original TDW-MAT code is:

```bash
python tdw_mat/demo/demo_scene.py
```

## Prompt templates

All templates live in [`tdw_mat/LLM/prompts/`](tdw_mat/LLM/prompts).

| Template | Setting |
|---|---|
| `plan_nocom.csv` | No communication |
| `plan_com.csv` | Natural-language communication and protocol-based communication |
| `protocol.txt` | Task-specific codebook generation |
| `encode.txt` | Semantic message to codewords |
| `decode.txt` | Codewords back to natural language, including `[mask]` recovery |
| `memory.txt` | Room-centric partner memory |

## Example protocol

[`example/protocol.json`](example/protocol.json) is a GFM-generated protocol for one transport episode. It contains the task-specific syntax rules, codebook, and encoding examples produced by the protocol-generation prompt, and is representative of the protocols used in the `comm_w_protocol` experiments.

## Figures

The manuscript figures are in [`figures/`](figures):

- `food_bar_charts.svg`
- `stuff_bar_charts.svg`
- `pareto_tradeoff.svg`
- `snr_frame_vs_snr.svg`
- `snr_semantic_similarity_vs_snr.svg`

## Reproduce the paper experiments

All commands below should be run from the repository root. Unknown flags are forwarded to `tdw_mat/tdw-gym/challenge.py`.

Clean-channel baselines, 24 test episodes:

```bash
python scripts/run_experiment.py no_comm
python scripts/run_experiment.py comm_wo_protocol
python scripts/run_experiment.py comm_w_protocol
```

Noisy-channel SNR grid:

```bash
python scripts/run_experiment.py noise_snr_grid
```

For a fair comparison, use the same per-episode protocol in the noise-free `comm_w_protocol` setting and in the noisy-channel SNR grid. Run `comm_w_protocol` first; `noise_snr_grid` then loads `results/noise_free/comm_w_protocol/<episode>/protocol.json`.

A single-episode smoke test:

```bash
python scripts/run_experiment.py comm_w_protocol --eval-episodes 0
```

The three paper communication settings are:

| Setting | Communication | Token protocol | Channel noise |
|---|---|---|---|
| `no_comm` | off | off | off |
| `comm_wo_protocol` | on | off | off |
| `comm_w_protocol` | on | on | off |
| `noise_snr_grid` | on | on | on, SNR in `{0,1,2,3,4}` dB |

Default model flags used in the manuscript:

- `--source gemma`
- `--lm_id 31b`
- `--think true`
- `--max_frames 6000`
- `--data_prefix tdw_mat/dataset/dataset_test/`

To regenerate figures after a local rerun:

```bash
python scripts/generate_noise_free_figures.py --results-root results/noise_free
python scripts/generate_snr_frame_report.py --results-root results/noise_challenge
```

The sentence-similarity SNR figure additionally needs `torch` and `transformers`:

```bash
python scripts/generate_snr_sentence_similarity_report.py --results-root results/noise_challenge
```

## License

This repository is released under the MIT License. It includes modified code from TDW-MAT / CoELA and `transport_challenge_multi_agent`. Please cite those projects if you use the environment, and cite this manuscript if you use the token-communication method.
