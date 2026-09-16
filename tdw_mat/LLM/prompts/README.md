# Prompt Templates

These files are the prompt templates used in the paper experiments.

| File | Used by | Role |
|---|---|---|
| `plan_com.csv` | action planning with communication | Natural-language or protocol-based cooperation |
| `plan_nocom.csv` | action planning without communication | No-communication baseline |
| `protocol.txt` | `LLM.generate_protocol()` | Task-specific codebook generation |
| `encode.txt` | `LLM.encode_message()` | Semantic message to codeword sequence |
| `decode.txt` | `LLM.decode_message()` | Codeword sequence back to natural language |
| `memory.txt` | `LLM.refresh_dialogue_memory()` | Room-centric partner memory |

Placeholders such as `$AGENT_NAME$`, `$PROTOCOL$`, and `$TASK_CONTEXT$` are filled at runtime.

`decode.txt` contains `$NOISY_CHANNEL_NOTE$`. In clean-channel evaluation this placeholder is empty. In noisy-channel evaluation it is replaced by a short instruction that `[mask]` tokens were corrupted by the channel.
