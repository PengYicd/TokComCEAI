import csv
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence


MASK_TOKEN = "[mask]"
SUPPORTED_SNR_DB = {0.0, 1.0, 2.0, 3.0, 4.0}


def load_snr_channel_map(csv_path: str) -> Dict[float, Dict[str, float]]:
    path = Path(csv_path)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    mapping: Dict[float, Dict[str, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"snr_db", "ber", "ter"}
        if reader.fieldnames is None or not required_columns.issubset(set(reader.fieldnames)):
            raise ValueError("csv must contain snr_db, ber, and ter columns")
        for row in reader:
            snr_db = float(row["snr_db"])
            if snr_db not in SUPPORTED_SNR_DB:
                continue
            mapping[snr_db] = {
                "ber": float(row["ber"]),
                "ter": float(row["ter"]),
            }
    if not mapping:
        raise ValueError("snr to ber/ter csv is empty for supported SNRs")
    return mapping


class TokenChannel:
    def __init__(
        self,
        tokenizer_dir: str,
        *,
        snr_db: float,
        snr_to_ter_csv_path: str,
        transmission_log_path: str,
        enable_noise: bool = False,
    ) -> None:
        _ = tokenizer_dir
        self.snr_db = float(snr_db)
        self.enable_noise = bool(enable_noise)
        self._snr_channel_map = load_snr_channel_map(snr_to_ter_csv_path)
        self._rng = random.Random(123)
        self.codebook_tokens = set()
        self.codebook_size = 0
        self._forced_bits_per_token = None
        self.set_output_dir(transmission_log_path)

    def _bits_per_token(self) -> int:
        if self._forced_bits_per_token is not None:
            return int(self._forced_bits_per_token)
        if self.codebook_size <= 1:
            return 0
        return int(math.ceil(math.log2(self.codebook_size)))

    def set_codebook(
        self,
        codewords: Optional[Sequence[str]] = None,
        *,
        bits_per_token: Optional[int] = None,
    ) -> None:
        normalized = []
        seen = set()
        for codeword in codewords or []:
            token = str(codeword).strip()
            if not token or token in seen:
                continue
            seen.add(token)
            normalized.append(token)
        self.codebook_tokens = set(normalized)
        self.codebook_size = len(normalized)
        self._forced_bits_per_token = int(bits_per_token) if bits_per_token is not None else None

    def set_codebook_from_protocol(self, protocol_data: Optional[Dict]) -> None:
        if not isinstance(protocol_data, dict):
            self.set_codebook([])
            return
        codebook = protocol_data.get("codebook", [])
        codewords = []
        if isinstance(codebook, list):
            for entry in codebook:
                if not isinstance(entry, dict):
                    continue
                codeword = str(entry.get("codeword", "")).strip()
                if codeword:
                    codewords.append(codeword)
        self.set_codebook(codewords)

    def _mapped_ber(self, snr_db: Optional[float] = None) -> float:
        snr = self.snr_db if snr_db is None else float(snr_db)
        if snr in self._snr_channel_map:
            return float(self._snr_channel_map[snr]["ber"])

        keys = sorted(self._snr_channel_map.keys())
        if snr <= keys[0]:
            return float(self._snr_channel_map[keys[0]]["ber"])
        if snr >= keys[-1]:
            return float(self._snr_channel_map[keys[-1]]["ber"])

        for left, right in zip(keys[:-1], keys[1:]):
            if left <= snr <= right:
                left_ter = self._snr_channel_map[left]["ber"]
                right_ter = self._snr_channel_map[right]["ber"]
                ratio = (snr - left) / (right - left)
                return float(left_ter + ratio * (right_ter - left_ter))
        return 0.0

    def _mapped_ter(self, snr_db: Optional[float] = None) -> float:
        bits_per_token = self._bits_per_token()
        if bits_per_token <= 0:
            return 0.0
        ber = self._mapped_ber(snr_db)
        return float(1.0 - math.pow(max(0.0, 1.0 - ber), bits_per_token))

    def reset_episode_stats(self) -> None:
        self.last_token_count = 0
        self.total_token_count = 0
        self.total_record_count = 0
        self.last_transmission_record = None
        self.episode_stats = {
            "num_messages": 0,
            "plain_text_messages": 0,
            "protocol_messages": 0,
            "raw_total_tokens": 0,
            "raw_total_bits": 0,
            "encoded_total_tokens": 0,
            "encoded_total_bits": 0,
            "channel_total_tokens": 0,
            "channel_total_bits": 0,
            "token_error_count": 0,
        }

    def set_output_dir(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.transmission_log_path = os.path.join(output_dir, "transmission_log.csv")
        self.reset_episode_stats()
        with open(self.transmission_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "message_id",
                "direction",
                "mode",
                "agent_id",
                "peer_agent_id",
                "step",
                "semantic_message",
                "channel_input_text",
                "channel_output_text",
                "raw_token_count",
                "raw_total_bits",
                "channel_token_count",
                "channel_total_bits",
                "encoded_token_count",
                "encoded_total_bits",
                "codebook_size",
                "channel_codebook_size",
                "channel_bits_per_token",
                "snr_db",
                "mapped_ber",
                "mapped_ter",
                "noise_enabled",
                "token_error_count",
            ])

    def _split_tokens(self, message: str) -> List[str]:
        text = str(message).strip()
        return text.split() if text else []

    def text_to_token_ids(self, message: str) -> List[str]:
        return self._split_tokens(message)

    def token_ids_to_text(self, token_ids: Sequence[str], *, skip_special_tokens: bool = True) -> str:
        _ = skip_special_tokens
        return " ".join(str(token).strip() for token in token_ids if str(token).strip())

    def estimate_text_stats(self, message: str, *, token_ids: Optional[Sequence[str]] = None) -> Dict:
        tokens = [str(token).strip() for token in (list(token_ids) if token_ids is not None else self.text_to_token_ids(message))]
        tokens = [token for token in tokens if token]
        token_count = len(tokens)
        bits_per_token = self._bits_per_token()
        return {
            "message": str(message).strip(),
            "tokens": tokens,
            "token_count": token_count,
            "bits_per_token": bits_per_token,
            "estimated_total_bits": token_count * bits_per_token,
        }

    def apply_token_errors(self, token_ids: Sequence[str], *, snr_db: Optional[float] = None) -> List[str]:
        if not self.enable_noise:
            return [str(token) for token in token_ids]

        ter = self._mapped_ter(snr_db)
        out: List[str] = []
        for token in token_ids:
            token_text = str(token).strip()
            if not token_text:
                continue
            if token_text == MASK_TOKEN or self._rng.random() >= ter:
                out.append(token_text)
                continue
            out.append(MASK_TOKEN)
        return out

    def all_tokens_masked(self, token_ids: Sequence[str]) -> bool:
        tokens = [str(token).strip() for token in token_ids if str(token).strip()]
        return bool(tokens) and all(token == MASK_TOKEN for token in tokens)

    def record_transmission(
        self,
        channel_input_text: str,
        *,
        channel_output_text: str,
        tx_token_ids: Sequence[str],
        rx_token_ids: Sequence[str],
        snr_db: Optional[float],
        mode: str,
        semantic_stats: Dict,
        protocol_stats: Optional[Dict],
        agent_id: Optional[int],
        step: Optional[int],
        direction: str = "send",
        peer_agent_id: Optional[int] = None,
        include_in_stats: bool = True,
    ) -> Dict:
        channel_stats = self.estimate_text_stats(channel_input_text, token_ids=tx_token_ids)
        encoded_stats = protocol_stats or channel_stats
        token_error_count = sum(
            1 for tx, rx in zip(tx_token_ids, rx_token_ids)
            if str(tx).strip() != str(rx).strip()
        )

        self.total_record_count += 1
        if include_in_stats:
            self.last_token_count = channel_stats["token_count"]
            self.total_token_count += channel_stats["token_count"]
            self.episode_stats["num_messages"] += 1
            self.episode_stats["plain_text_messages"] += 0 if protocol_stats else 1
            self.episode_stats["protocol_messages"] += 1 if protocol_stats else 0
            self.episode_stats["raw_total_tokens"] += int(semantic_stats.get("token_count", 0) or 0)
            self.episode_stats["raw_total_bits"] += int(semantic_stats.get("estimated_total_bits", 0) or 0)
            self.episode_stats["encoded_total_tokens"] += int(encoded_stats.get("token_count", 0) or 0)
            self.episode_stats["encoded_total_bits"] += int(encoded_stats.get("estimated_total_bits", 0) or 0)
            self.episode_stats["channel_total_tokens"] += int(channel_stats.get("token_count", 0) or 0)
            self.episode_stats["channel_total_bits"] += int(channel_stats.get("estimated_total_bits", 0) or 0)
            self.episode_stats["token_error_count"] += token_error_count

        record = {
            "message_id": self.total_record_count,
            "direction": str(direction),
            "mode": mode,
            "agent_id": agent_id,
            "peer_agent_id": peer_agent_id,
            "step": step,
            "semantic_message": semantic_stats.get("message", channel_input_text),
            "channel_input_text": str(channel_input_text),
            "channel_output_text": str(channel_output_text),
            "raw_token_count": int(semantic_stats.get("token_count", 0) or 0),
            "raw_total_bits": int(semantic_stats.get("estimated_total_bits", 0) or 0),
            "channel_token_count": int(channel_stats.get("token_count", 0) or 0),
            "channel_total_bits": int(channel_stats.get("estimated_total_bits", 0) or 0),
            "encoded_token_count": int(encoded_stats.get("token_count", 0) or 0),
            "encoded_total_bits": int(encoded_stats.get("estimated_total_bits", 0) or 0),
            "codebook_size": encoded_stats.get("codebook_size"),
            "channel_codebook_size": self.codebook_size,
            "channel_bits_per_token": self._bits_per_token(),
            "snr_db": self.snr_db if snr_db is None else float(snr_db),
            "mapped_ber": self._mapped_ber(snr_db),
            "mapped_ter": self._mapped_ter(snr_db),
            "noise_enabled": self.enable_noise,
            "token_error_count": token_error_count,
        }
        self.last_transmission_record = record

        with open(self.transmission_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                record["message_id"],
                record["direction"],
                record["mode"],
                record["agent_id"],
                record["peer_agent_id"],
                record["step"],
                record["semantic_message"],
                record["channel_input_text"],
                record["channel_output_text"],
                record["raw_token_count"],
                record["raw_total_bits"],
                record["channel_token_count"],
                record["channel_total_bits"],
                record["encoded_token_count"],
                record["encoded_total_bits"],
                record["codebook_size"],
                record["channel_codebook_size"],
                record["channel_bits_per_token"],
                record["snr_db"],
                record["mapped_ber"],
                record["mapped_ter"],
                record["noise_enabled"],
                record["token_error_count"],
            ])
        return record

    def get_episode_stats(self) -> Dict:
        num_messages = int(self.episode_stats["num_messages"])
        raw_total_tokens = int(self.episode_stats["raw_total_tokens"])
        raw_total_bits = int(self.episode_stats["raw_total_bits"])
        encoded_total_tokens = int(self.episode_stats["encoded_total_tokens"])
        encoded_total_bits = int(self.episode_stats["encoded_total_bits"])
        channel_total_tokens = int(self.episode_stats["channel_total_tokens"])
        token_error_count = int(self.episode_stats["token_error_count"])
        return {
            "num_messages": num_messages,
            "plain_text_messages": int(self.episode_stats["plain_text_messages"]),
            "protocol_messages": int(self.episode_stats["protocol_messages"]),
            "raw_total_tokens": raw_total_tokens,
            "raw_total_bits": raw_total_bits,
            "encoded_total_tokens": encoded_total_tokens,
            "encoded_total_bits": encoded_total_bits,
            "channel_total_tokens": int(self.episode_stats["channel_total_tokens"]),
            "channel_total_bits": int(self.episode_stats["channel_total_bits"]),
            "avg_raw_tokens_per_message": (raw_total_tokens / num_messages) if num_messages else 0.0,
            "avg_encoded_tokens_per_message": (encoded_total_tokens / num_messages) if num_messages else 0.0,
            "avg_raw_bits_per_message": (raw_total_bits / num_messages) if num_messages else 0.0,
            "avg_encoded_bits_per_message": (encoded_total_bits / num_messages) if num_messages else 0.0,
            "token_error_count": token_error_count,
            "observed_ter": (token_error_count / channel_total_tokens) if channel_total_tokens else 0.0,
            "compression_ratio_tokens": (raw_total_tokens / encoded_total_tokens) if encoded_total_tokens else None,
            "compression_ratio_bits": (raw_total_bits / encoded_total_bits) if encoded_total_bits else None,
            "channel_tokenizer_vocab_size": self.codebook_size,
            "channel_codebook_size": self.codebook_size,
            "channel_bits_per_token": self._bits_per_token(),
            "noise_enabled": self.enable_noise,
            "default_snr_db": self.snr_db,
            "mapped_ber_at_default_snr": self._mapped_ber(),
            "mapped_ter_at_default_snr": self._mapped_ter(),
        }
