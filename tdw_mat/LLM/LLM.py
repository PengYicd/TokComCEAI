import os, random, re, json, math
from datetime import datetime
import pandas as pd
import backoff

from .gemma import gemma_generate

class LLM:
    def __init__(self,
                 source, 
                 lm_id,
                 think,
                 prompt_template_path,
                 communication,
                 token_comm,
                 agent_id
                 ):
        self.rooms_explored = None
        self.goal_desc = None
        self.agent_id = agent_id
        self.agent_name = "Alice" if agent_id == 0 else "Bob"
        self.oppo_name = "Alice" if agent_id == 1 else "Bob"
        self.rooms = []
        self.prompt_template_path = prompt_template_path
        
        df = pd.read_csv(self.prompt_template_path)
        templates = {}
        if 'type' in df.columns and 'prompt' in df.columns:
            for _, row in df.iterrows():
                t = str(row.get('type', '')).strip()
                p = row.get('prompt', None)
                if isinstance(p, str) and t:
                    templates[t] = p
        if not templates and 'prompt' in df.columns:
            prompts = list(df['prompt'])
            templates['base'] = prompts[0]
            if len(prompts) > 2:
                templates['enc'] = prompts[2]
            if len(prompts) > 3:
                templates['dec'] = prompts[3]

        def _tmpl(name: str):
            p = templates.get(name)
            if not isinstance(p, str):
                return None
            return p.replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)

        self.prompt_template = _tmpl('base') or ""
        self._prompt_dir = os.path.join(os.path.dirname(__file__), "prompts")
        self.protocol_prompt_template = self._load_prompt_text("protocol.txt")
        self.encode_prompt_template = self._load_prompt_text("encode.txt")
        self.decode_prompt_template = self._load_prompt_text("decode.txt")
        self.memory_prompt_template = self._load_prompt_text("memory.txt")

        self.communication = communication
        self.token_comm = token_comm
        self.source = source
        self.lm_id = lm_id
        self.think = think
        self.total_cost = 0
        self.max_parse_retries = 5
        self.dialogue_history = []
        self.last_semantic_message = None
        self.last_encoded_message = None
        self.last_encode_result = None
        self.last_decode_result = None
        self.protocol = None
        self.protocol_data = None
        self.protocol_codebook = {}
        self.protocol_codewords = []
        self.protocol_delimiter = " "

        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(log_dir, exist_ok=True)
        self._io_log_path = os.path.join(log_dir, f"llm_io_agent{self.agent_id}.jsonl")
        self._clear_io_log()

        def lm_engine(source):
            @backoff.on_exception(backoff.expo, Exception)
            def gemma_generate_with_backoff(prompt):
                text, usage = gemma_generate(
                    prompt,
                    model=self.lm_id,
                    think=self.think
                )
                return text.strip(), int(usage)

            def _generate(prompt):
                if source == "gemma":
                    return gemma_generate_with_backoff(prompt)
                raise ValueError("invalid source")

            return _generate

        self.generator = lm_engine(self.source)

        self.current_room = None
        self.object_list = None
        self.holding_objects = None
        self.obj_per_room = None
        self.protocol_task_inventory = {}
        self.protocol_output_dir = None

    def _load_prompt_text(self, filename: str) -> str:
        path = os.path.join(getattr(self, "_prompt_dir", os.path.join(os.path.dirname(__file__), "prompts")), filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"prompt template not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _render_prompt(self, template: str, mapping: dict) -> str:
        text = template
        for key, value in mapping.items():
            text = text.replace(key, str(value))
        return text

    def reset(self, rooms_name, goal_objects, protocol_task_inventory=None, output_dir=None):
        self.rooms = rooms_name
        self.goal_desc = self.goal2description(goal_objects)
        self.dialogue_history = []
        self.protocol = None
        self.protocol_data = None
        self.protocol_codebook = {}
        self.protocol_codewords = []
        self.protocol_delimiter = " "
        self.last_semantic_message = None
        self.last_encoded_message = None
        self.last_encode_result = None
        self.last_decode_result = None
        self.protocol_task_inventory = self._normalize_protocol_task_inventory(protocol_task_inventory)
        self.protocol_output_dir = output_dir
        self._clear_io_log()

    def _clear_io_log(self) -> None:
        with open(self._io_log_path, "w", encoding="utf-8") as f:
            f.write("")

    def _normalize_model_output(self, output) -> str:
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                return ""
            return self._normalize_model_output(output[0])
        return str(output).strip()

    def _usage_total(self, usage) -> int:
        try:
            return int(usage)
        except Exception:
            return 0

    def _extract_tagged_content(self, text: str, tag: str):
        matches = re.findall(rf'<\|{re.escape(tag)}\|>(.*?)<\|{re.escape(tag)}\|>', text, re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def _strip_json_fence(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        return stripped.strip()

    def _bits_needed(self, vocab_size: int) -> int:
        size = int(vocab_size)
        if size <= 1:
            return 0
        return int(math.ceil(math.log2(size)))

    def _parse_protocol_json(self, protocol_str: str) -> dict:
        payload = self._strip_json_fence(protocol_str)
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise ValueError("protocol must be a JSON object")

        codebook = data.get("codebook")
        if not isinstance(codebook, list) or len(codebook) == 0:
            raise ValueError("protocol.codebook must be a non-empty list")

        def _normalize_text_list(value, field_name: str):
            if value is None:
                return []
            if not isinstance(value, list):
                raise ValueError(f"protocol.{field_name} must be a list of strings")
            items = []
            for item in value:
                text = str(item).strip()
                if text:
                    items.append(text)
            return items

        syntax_rules = _normalize_text_list(data.get("syntax rules"), "syntax rules")
        if len(syntax_rules) == 0:
            raise ValueError("protocol.syntax rules must be a non-empty list")

        normalized_codebook = []
        seen = set()
        for entry in codebook:
            if not isinstance(entry, dict):
                raise ValueError("each codebook entry must be a JSON object")
            codeword = str(entry.get("codeword", "")).strip()
            meaning = str(entry.get("meaning", "")).strip()
            category = str(entry.get("category", "")).strip() 
            if not codeword or not meaning:
                raise ValueError("each codebook entry must contain non-empty codeword and meaning")
            if any(ch.isspace() for ch in codeword):
                raise ValueError(f"codeword '{codeword}' must be a single token without whitespace")
            if codeword in seen:
                raise ValueError(f"duplicate codeword '{codeword}' found in codebook")
            seen.add(codeword)
            normalized_codebook.append({
                "codeword": codeword,
                "meaning": meaning,
                "category": category,
            })

        examples = data.get("example")
        if not isinstance(examples, list) or len(examples) == 0:
            raise ValueError("protocol.example must be a non-empty list")

        normalized_examples = []
        for entry in examples:
            if not isinstance(entry, dict):
                raise ValueError("each example entry must be a JSON object")
            encoded_message = str(entry.get("encoded message", "")).strip()
            original_message = str(entry.get("original message", "")).strip()
            if not encoded_message or not original_message:
                raise ValueError("each example entry must contain non-empty encoded message and original message")
            invalid_tokens = [tok for tok in encoded_message.split() if tok not in seen]
            if invalid_tokens:
                raise ValueError(f"example contains out-of-codebook tokens: {invalid_tokens}")
            normalized_examples.append({
                "encoded message": encoded_message,
                "original message": original_message,
            })

        normalized = {
            "syntax rules": syntax_rules,
            "codebook": normalized_codebook,
            "example": normalized_examples,
        }
        return normalized

    def set_protocol(self, protocol_payload) -> str:
        if isinstance(protocol_payload, dict):
            protocol_data = self._parse_protocol_json(json.dumps(protocol_payload, ensure_ascii=False))
        elif isinstance(protocol_payload, str):
            protocol_data = self._parse_protocol_json(protocol_payload)
        else:
            raise TypeError("protocol_payload must be a dict or JSON string")

        self.protocol_data = protocol_data
        self.protocol = json.dumps(protocol_data, indent=2, ensure_ascii=False)
        self.protocol_delimiter = " "
        self.protocol_codebook = {
            entry["codeword"]: entry for entry in protocol_data["codebook"]
        }
        self.protocol_codewords = list(self.protocol_codebook.keys())
        return self.protocol

    def _normalize_protocol_task_inventory(self, protocol_task_inventory) -> dict:
        if not isinstance(protocol_task_inventory, dict):
            return {}

        normalized = {}
        for room_name, info in protocol_task_inventory.items():
            room = str(room_name).strip()
            if not room:
                continue
            if not isinstance(info, dict):
                info = {}

            def _normalize_entries(values):
                items = []
                seen = set()
                if not isinstance(values, list):
                    return items
                for value in values:
                    item = str(value).strip()
                    if not item or item in seen:
                        continue
                    seen.add(item)
                    items.append(item)
                return items

            normalized[room] = {
                "targets": _normalize_entries(info.get("targets", [])),
                "containers": _normalize_entries(info.get("containers", [])),
            }
        return normalized

    def _collect_protocol_entities(self, field_name: str):
        entries = []
        seen = set()
        for room_info in self.protocol_task_inventory.values():
            for item in room_info.get(field_name, []):
                text = str(item).strip()
                if not text or text in seen:
                    continue
                seen.add(text)
                entries.append(text)
        return sorted(entries)

    def _format_protocol_task_context(self) -> str:
        objects = self._collect_protocol_entities("targets")
        containers = self._collect_protocol_entities("containers")
        lines = [
            "Task:",
            f"Goal: {self.goal_desc}",
            f"Rooms: {', '.join(self.rooms)}",
            f"Target objects: {', '.join(objects)}",
            f"Containers: {', '.join(containers)}",
            "Entity format: rooms, targets, and containers use <name> (id).",
            "Identity rule: same name but different id means a different entity.",
            "Available action types: go to a room, explore the current room, grasp a target object, grasp a container, put an object into a carried container, transport carried objects to the bed, and send a message.",
            "High-value coordination information: room claim/search, room empty/cleared, found items, taken items, remaining items, and next destination.",
        ]

        return "\n".join(lines)

    def _request_tagged_response(self, prompt: str, tag: str, validate_fn=None):
        chat_prompt = [{"role": "user", "content": prompt}]
        output = ""
        last_error = f"{tag} not found in output"

        for attempt in range(self.max_parse_retries):
            raw_output, usage = self.generator(chat_prompt)
            output = self._normalize_model_output(raw_output)
            self.total_cost += self._usage_total(usage)

            try:
                payload = self._extract_tagged_content(output, tag)
                if payload is None:
                    raise ValueError(f"{tag} not found in output")
                result = validate_fn(payload) if validate_fn is not None else payload.strip()
                return result, output, usage, attempt
            except Exception as exc:
                last_error = str(exc)
                print(last_error)

        raise ValueError(last_error)

    def _validate_protocol_payload(self, protocol_payload: str) -> str:
        self.set_protocol(protocol_payload)
        return self.protocol

    def _save_protocol_json(self) -> None:
        if not self.protocol or not self.protocol_output_dir:
            return
        os.makedirs(self.protocol_output_dir, exist_ok=True)
        protocol_path = os.path.join(self.protocol_output_dir, "protocol.json")
        with open(protocol_path, "w", encoding="utf-8") as f:
            f.write(self.protocol)
            if not self.protocol.endswith("\n"):
                f.write("\n")

    def _validate_encoded_message(self, encoded_message: str) -> dict:
        message = str(encoded_message).strip()
        stats = self.estimate_protocol_stats(message)
        if stats["token_count"] == 0:
            raise ValueError("encoded message is empty")
        if stats["invalid_tokens"]:
            raise ValueError(
                f"encoded message contains out-of-codebook tokens: {stats['invalid_tokens']}"
            )
        return dict(stats)

    def _validate_decoded_message(self, decoded_message: str) -> str:
        message = str(decoded_message).strip()
        if not message:
            raise ValueError("decoded message is empty")
        return message

    def _tokenize_protocol_message(self, message: str):
        text = str(message).strip()
        if text == "":
            return []
        if self.protocol_delimiter.isspace():
            return text.split()
        return [tok.strip() for tok in text.split(self.protocol_delimiter) if tok.strip()]

    def estimate_free_text_stats(self, message: str, token_ids=None, vocab_size=None) -> dict:
        text = str(message).strip()
        if token_ids is not None:
            token_count = len(list(token_ids))
            estimation_method = "external_tokenizer"
        else:
            token_count = len(text.split()) if text else 0
            estimation_method = "whitespace_fallback"

        bits_per_token = None
        estimated_total_bits = None
        if vocab_size is not None:
            bits_per_token = self._bits_needed(int(vocab_size))
            estimated_total_bits = token_count * bits_per_token

        return {
            "message": text,
            "token_count": token_count,
            "bits_per_token": bits_per_token,
            "estimated_total_bits": estimated_total_bits,
            "vocab_size": vocab_size,
            "estimation_method": estimation_method,
        }

    def estimate_protocol_stats(self, encoded_message: str, *, allow_mask_token: bool = False) -> dict:
        if not self.protocol_codebook:
            raise ValueError("protocol not defined")

        tokens = self._tokenize_protocol_message(encoded_message)
        invalid_tokens = [
            tok for tok in tokens
            if tok not in self.protocol_codebook and not (allow_mask_token and tok == "[mask]")
        ]
        codebook_size = len(self.protocol_codebook)
        bits_per_token = self._bits_needed(codebook_size)
        mask_token_count = sum(1 for tok in tokens if tok == "[mask]")

        return {
            "message": str(encoded_message).strip(),
            "tokens": tokens,
            "token_count": len(tokens),
            "invalid_tokens": invalid_tokens,
            "mask_token_count": mask_token_count,
            "all_tokens_masked": bool(tokens) and mask_token_count == len(tokens),
            "codebook_size": codebook_size,
            "bits_per_token": bits_per_token,
            "estimated_total_bits": len(tokens) * bits_per_token,
            "delimiter": self.protocol_delimiter,
        }

    def generate_protocol(self) -> str:
        prompt = self._render_prompt(
            self.protocol_prompt_template,
            {"$TASK_CONTEXT$": self._format_protocol_task_context()},
        )
        protocol_text, output, usage, attempt = self._request_tagged_response(
            prompt,
            "protocol",
            validate_fn=self._validate_protocol_payload,
        )
        self._save_protocol_json()
        self._append_io_log({
            "stage": "generate_protocol",
            "attempt": attempt,
            "prompt": prompt,
            "outputs": output,
            "usage": usage,
            "extracted_protocol": protocol_text,
            "protocol_data": self.protocol_data,
        })
        return protocol_text

    def _append_io_log(self, record: dict) -> None:
        try:
            record = dict(record)
            record["ts"] = datetime.now().isoformat(timespec="seconds")
            record["agent_id"] = self.agent_id
            record["agent_name"] = self.agent_name
            with open(self._io_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str))
                f.write("\n")
        except Exception:
            return


    def goal2description(self, goals):  # {predicate: count}
        s = "Transport "
        for object_name, count in goals.items():
            s += f"{count} {object_name}{'s' if count > 1 else ''}, "
        s = s[:-2] + f" to the bed."
        return s


    def parse_answer(self, available_actions, text, warn: bool = False):
        text = text.strip()
        flags = 'AC'
        
        # 1. Try to extract action and message from tags
        action_match = re.search(r'<\|action\|>(.*?)<\|action\|>', text, re.DOTALL)
        message_match = re.search(r'<\|message\|>(.*?)<\|message\|>', text, re.DOTALL)
        extracted_message = message_match.group(1).strip() if message_match else ""

        def get_result(idx, current_flags):
            act = available_actions[idx]
            if act == "send a message" or act.startswith("send a message:"):
                if extracted_message == "":
                    raise ValueError("send a message selected but <|message|> is missing or empty")
                return f"send a message: {extracted_message}", current_flags
            return act, current_flags

        if action_match:
            parsed_action = action_match.group(1).strip().lower()
            for i in range(len(available_actions)):
                action_str = available_actions[i]
                check_str = "send a message" if action_str.startswith("send a message:") else action_str
                if check_str.lower() == parsed_action or check_str.lower() in parsed_action:
                    try:
                        return get_result(i, flags)
                    except ValueError:
                        break

        if warn:
            raise ValueError("WARNING! No available action parsed!!!")
        flags = "failed to parse"
        chosen_idx = random.randint(0, len(available_actions) - 1)
        return get_result(chosen_idx, flags)


    def progress2text(self, current_step, satisfied):
        s = f"I've taken {current_step} steps. "

        sss = {}
        for room, obj_list in self.obj_per_room.items():
            sr = ""
            s_obj = ""
            s_con = ""
            objs = obj_list[0]
            cons = obj_list[1]
            if len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"a target object <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in objs])
                    s_obj += f"target objects " + ss

            if len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"a container <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in cons])
                    s_con = f"containers " + ss
            if s_obj == "" and s_con == "":
                sr += 'nothing'
            elif s_obj != "" and s_con != "":
                sr += s_obj + ', and ' + s_con
            elif s_obj != "" and s_con == "":
                sr += s_obj
            elif s_obj == "" and s_con != "":
                sr += s_con
            elif s_obj != "" and s_con != "":
                sr += s_obj + ', ' + s_con
            else:
                sr += s_obj + s_con
            sss[room] = sr

        if len(satisfied) == 0:
            pass
        else:
            s += f"The target objects I have personally transported to the bed or confirmed at the bed are "
            unique_satisfied = []
            for x in satisfied:
                if x not in unique_satisfied:
                    unique_satisfied.append(x)
            satisfied_targets = [x for x in unique_satisfied if x['type'] == 0]
            if len(satisfied_targets) == 0:
                s += 'nothing'
            else:
                s += ', '.join([f"<{x['name']}> ({x['id']})" for x in satisfied_targets])
            s += '. '

        s_hold = ["", ""]
        for i, obj in enumerate(self.holding_objects):
            if obj['type'] == 0:
                s_hold[i] = f"a target object <{obj['name']}> ({obj['id']}). "
            elif obj['type'] == 1:
                ss = ""
                cnt = 0
                for j, o in enumerate(obj['contained']):
                    if o is None:
                        break
                    cnt += 1
                    ss += f"<{obj['contained_name'][j]}> ({o}), "
                if cnt == 0:
                    ss = 'nothing'
                else:
                    ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it. "

        if self.holding_objects[0]["type"] == 0 and self.holding_objects[1]['type'] == 0:
            s += f"I'm holding two target objects <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) and <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}). "
        elif s_hold[0] == "" and s_hold[1] == "":
            s += "I'm holding nothing. "
        elif s_hold[0] != "" and s_hold[1] != "":
            s += f"I'm holding {s_hold[0][:-2]}, and {s_hold[1]}"
        else:
            s += f"I'm holding {s_hold[0]}{s_hold[1]}"

        # If we have already found objects in a room, it shouldn't still read as "none".
        rooms_explored = self.rooms_explored or {}

        def _pred_room(room):
            pred_room = rooms_explored.get(room, 'none')
            if pred_room == 'none' and sss[room] != 'nothing':
                return 'part'
            return pred_room

        # print(self.current_room, self.obj_per_room)
        pred_room = _pred_room(self.current_room)
        if pred_room != 'all' and sss[self.current_room] == 'nothing':
            s += f"I'm in the {self.current_room}, where I've explored {pred_room} of it. "
        else:
            s += f"I'm in the {self.current_room}, where I've explored {pred_room} of it and found {sss[self.current_room]}. "
        for room in self.rooms:
            if room == self.current_room:
                continue
            #s += f"I've explored {self.rooms_explored[room] if room in self.rooms_explored else 'None'} of the {room}, and I found {sss[room]} there. "
            pred_room = _pred_room(room)
            if pred_room != 'all' and sss[room] == 'nothing':
                s += f"I've explored {pred_room} of the {room}. "
            else:
                s += f"I've explored {pred_room} of the {room}, and I found {sss[room]} there. "

        return s

    def _format_available_plans(self, available_plans):
        plans = ""
        for i, plan in enumerate(available_plans):
            option = chr(ord('A') + i)
            plans += f"{option}: {plan}\n"
        return plans


    def get_available_plans(self):
        """
        go to room {}
        explore current room {}
        go grasp target object / container {}
        holding both container and object: put obj into the container
        holding any goal objects: transport holding objects to the bed
        send a message""
        """
        available_plans = []
        if self.communication:
            available_plans.append(f"send a message")
        if self.holding_objects[0]['type'] is None or self.holding_objects[1]['type'] is None:
            for obj in self.object_list[0]:
                available_plans.append(f"go grasp target object <{obj['name']}> ({obj['id']})")
            if not (self.holding_objects[0]['type'] == 1 or self.holding_objects[1]['type'] == 1):
                for obj in self.object_list[1]:
                    available_plans.append(f"go grasp container <{obj['name']}> ({obj['id']})")
        else:
            if self.holding_objects[0]['type'] == 1 and self.holding_objects[0]['contained'][-1] is None and self.holding_objects[1]['type'] == 0:
                available_plans.append(f"put <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}) into the container <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']})")
            elif self.holding_objects[1]['type'] == 1 and self.holding_objects[1]['contained'][-1] is None and self.holding_objects[0]['type'] == 0:
                available_plans.append(f"put <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) into the container <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']})")
        can_transport_to_goal = bool(getattr(self, "goal_position_known", False) or len(self.object_list[2]) != 0)
        if any(obj['type'] is not None for obj in self.holding_objects) and can_transport_to_goal:
            available_plans.append(f"transport objects I'm holding to the bed")
        for room in self.rooms:
            if room == self.current_room or room is None or room == 'None':
                continue
            available_plans.append(f"go to {room}")
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            available_plans.append(f"explore current room {self.current_room}")

        return self._format_available_plans(available_plans), len(available_plans), available_plans


    def encode_message(self, message: str) -> dict:
        if not self.protocol_data:
            raise ValueError("protocol not defined")

        semantic_message = str(message).strip()
        protocol_str = json.dumps(self.protocol_data, indent=2, ensure_ascii=False)
        prompt = self._render_prompt(
            self.encode_prompt_template,
            {
                "$PROTOCOL$": protocol_str,
                "$ORIGINAL_MESSAGE$": semantic_message,
            },
        )
        stats, output, usage, attempt = self._request_tagged_response(
            prompt,
            "message",
            validate_fn=self._validate_encoded_message,
        )
        result = dict(stats)
        result.update({
            "semantic_message": semantic_message,
            "message": result["message"],
        })
        self.last_semantic_message = semantic_message
        self.last_encoded_message = result["message"]
        self.last_encode_result = result
        self._append_io_log({
            "stage": "encode",
            "attempt": attempt,
            "prompt": prompt,
            "outputs": output,
            "usage": usage,
            "result": result,
        })
        return result

    def decode_message(self, message, *, context: str = "", noise_channel: bool = False) -> dict:
        if not self.protocol_data:
            raise ValueError("protocol not defined")

        encoded_message = message.get("message", "") if isinstance(message, dict) else str(message)
        protocol_str = json.dumps(self.protocol_data, indent=2, ensure_ascii=False)
        encoded_text = str(encoded_message).strip()
        has_mask_token = "[mask]" in encoded_text.split()
        context_text = str(context).strip() or "None"
        if noise_channel or has_mask_token:
            noisy_note = (
                f"Context:\n{context_text}\n\n"
                "Noisy-channel note: any received token marked as [mask] means that protocol token was corrupted by the channel. "
                "You may use the provided context and the protocol structure to infer the most likely masked token(s), then decode the full intended message.\n"
            )
        else:
            noisy_note = ""
        prompt = self._render_prompt(
            self.decode_prompt_template,
            {
                "$PROTOCOL$": protocol_str,
                "$ENCODED_MESSAGE$": encoded_text,
                "$NOISY_CHANNEL_NOTE$": noisy_note,
            },
        )
        decoded_message, output, usage, attempt = self._request_tagged_response(
            prompt,
            "message",
            validate_fn=self._validate_decoded_message,
        )
        result = {
            "message": decoded_message,
            "encoded message": encoded_text,
            "protocol stats": self.estimate_protocol_stats(encoded_text, allow_mask_token=True),
        }
        self.last_decode_result = result
        self._append_io_log({
            "stage": "decode",
            "attempt": attempt,
            "prompt": prompt,
            "outputs": output,
            "usage": usage,
            "result": result,
        })
        return result

    def _normalize_partner_memory_payload(self, payload, room_names=None) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("partner memory must be a JSON object")

        valid_rooms = [str(room).strip() for room in (room_names or []) if str(room).strip()]
        valid_states = {"unexplored", "empty", "cleared", "objectives"}
        valid_agents = {"Alice", "Bob"}

        def _normalize_room_key(room_text: str) -> str:
            text = str(room_text).strip().lower()
            text = text.replace("<", "").replace(">", "")
            text = re.sub(r"\s+", "", text)
            return text

        room_aliases = {}
        for room in valid_rooms:
            room_aliases[_normalize_room_key(room)] = room

        def _dedup_list(values):
            normalized = []
            seen = set()
            if not isinstance(values, list):
                return normalized
            for value in values:
                item = str(value).strip()
                if not item or item in seen:
                    continue
                seen.add(item)
                normalized.append(item)
            return normalized

        normalized_rooms = {}
        raw_rooms = payload.get("rooms", {})
        if isinstance(raw_rooms, dict):
            for room, info in raw_rooms.items():
                room_name = str(room).strip()
                if not room_name:
                    continue
                canonical_room_name = room_aliases.get(_normalize_room_key(room_name), room_name)
                if valid_rooms and canonical_room_name not in valid_rooms:
                    continue
                if not isinstance(info, dict):
                    continue
                objectives = _dedup_list(info.get("objectives", []))
                state = str(info.get("state", "")).strip().lower()
                if objectives and state in ("", "unexplored"):
                    state = "objectives"
                if state not in valid_states:
                    state = "unexplored"
                confirmed_by = []
                raw_confirmed_by = info.get("confirmed_by", [])
                if isinstance(raw_confirmed_by, str):
                    raw_confirmed_by = [raw_confirmed_by]
                if isinstance(raw_confirmed_by, list):
                    seen_agents = set()
                    for agent_name in raw_confirmed_by:
                        normalized_agent = str(agent_name).strip()
                        if normalized_agent not in valid_agents or normalized_agent in seen_agents:
                            continue
                        seen_agents.add(normalized_agent)
                        confirmed_by.append(normalized_agent)
                normalized_rooms[canonical_room_name] = {
                    "state": state,
                    "objectives": objectives,
                    "confirmed_by": confirmed_by,
                }

        partner_taken_objects = _dedup_list(payload.get("partner_taken_objects", []))

        return {
            "rooms": normalized_rooms,
            "partner_taken_objects": partner_taken_objects,
        }

    def refresh_dialogue_memory(
        self,
        existing_memory,
        dialogue_history,
        room_names=None,
    ) -> dict:
        room_names = [str(room).strip() for room in (room_names or []) if str(room).strip()]
        dialogue_history = [str(item).strip() for item in (dialogue_history or []) if str(item).strip()]

        normalized_existing = self._normalize_partner_memory_payload(
            existing_memory or {},
            room_names=room_names,
        )
        if not dialogue_history:
            return {"memory": normalized_existing}

        existing_memory_json = json.dumps(normalized_existing, indent=2, ensure_ascii=False)
        recent_dialogue_text = "\n".join(dialogue_history) if dialogue_history else "None"
        room_list_text = ", ".join(room_names) if room_names else "None"

        prompt = self._render_prompt(
            self.memory_prompt_template,
            {
                "$VALID_ROOM_NAMES$": room_list_text,
                "$PREVIOUS_MEMORY$": existing_memory_json,
                "$NEW_DIALOGUE$": recent_dialogue_text,
            },
        )

        chat_prompt = [{"role": "user", "content": prompt}]
        output = ""
        last_error = "memory not found in output"

        for attempt in range(self.max_parse_retries):
            raw_output, usage = self.generator(chat_prompt)
            output = self._normalize_model_output(raw_output)
            self.total_cost += self._usage_total(usage)

            try:
                payload = self._extract_tagged_content(output, "memory")
                if payload is None:
                    raise ValueError("memory not found in output")

                normalized_memory = self._normalize_partner_memory_payload(
                    json.loads(self._strip_json_fence(payload)),
                    room_names=room_names,
                )
                self._append_io_log({
                    "stage": "refresh_dialogue_memory",
                    "attempt": attempt,
                    "prompt": prompt,
                    "outputs": output,
                    "usage": usage,
                    "existing_memory": normalized_existing,
                    "dialogue_history": dialogue_history,
                    "result": normalized_memory,
                })
                return {"memory": normalized_memory}
            except Exception as exc:
                last_error = str(exc)
                self._append_io_log({
                    "stage": "refresh_dialogue_memory_retry",
                    "attempt": attempt,
                    "prompt": prompt,
                    "outputs": output,
                    "usage": usage,
                    "error": last_error,
                })
                retry_instruction = (
                    "Your previous memory refresh output was invalid. "
                    f"Fix this error: {last_error}. "
                    "Return ONLY one valid JSON object inside <|memory|> and <|memory|> tags."
                )
                chat_prompt.append({"role": "assistant", "content": output})
                chat_prompt.append({"role": "user", "content": retry_instruction})

        raise ValueError(f"failed to refresh dialogue memory: {last_error}")


    def run(self, current_step, current_room, rooms_explored, holding_objects, satisfied, object_list, obj_per_room, action_history, opponent_grabbed_objects = None, opponent_last_room = None, replanning_context: str = "", allow_message_action: bool = True, partner_memory_desc: str = "", goal_position_known: bool = False):
        info = {}
        print("current_step", current_step)
        self.current_room = current_room
        self.rooms_explored = rooms_explored
        self.holding_objects = holding_objects
        self.object_list = object_list
        self.obj_per_room = obj_per_room
        self.goal_position_known = goal_position_known
        progress_desc = self.progress2text(current_step, satisfied)
        action_history_desc = ", ".join(action_history[-5:] if len(action_history) > 5 else action_history)
        dialogue_history = self.dialogue_history
        dialogue_history_desc = '\n'.join(dialogue_history[-5:] if len(dialogue_history) > 5 else dialogue_history)
        partner_memory_desc = str(partner_memory_desc).strip()

        prompt = self.prompt_template.replace('$GOAL$', self.goal_desc)
        prompt = prompt.replace('$PROGRESS$', progress_desc)
        prompt = prompt.replace('$ACTION_HISTORY$', action_history_desc)

        if self.communication:
            prompt = prompt.replace(
                '$PARTNER_MEMORY$',
                partner_memory_desc if partner_memory_desc else "No stable dialogue memory yet."
            )
            prompt = prompt.replace('$DIALOGUE_HISTORY$', dialogue_history_desc)
        if len(replanning_context.strip()) == 0:
            prompt = prompt.replace('$REPLANNING_CONTEXT$', '')
        else:
            prompt = prompt.replace('$REPLANNING_CONTEXT$', f"\n{replanning_context.strip()}\n")

        available_plans, num, available_plans_list = self.get_available_plans()
        if not allow_message_action:
            available_plans_list = [plan for plan in available_plans_list if plan != "send a message"]
            available_plans = self._format_available_plans(available_plans_list)
            num = len(available_plans_list)
        if num == 0 or (self.communication and num == 1):
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num,
                     "plan": None})
            return plan, info

        prompt = prompt.replace('$AVAILABLE_ACTIONS$', available_plans)

        chat_prompt = [{"role": "user", "content": prompt}]
        output, usage = self.generator(chat_prompt)
        self.total_cost += self._usage_total(usage)
        
        for attempt in range(self.max_parse_retries):
            plan, flags = self.parse_answer(available_plans_list, output, warn=(attempt == self.max_parse_retries - 1))

            if flags == 'AC':
                self._append_io_log({
                    "stage": "plan_final",
                    "current_step": current_step,
                    "current_room": current_room,
                    "available_actions_list": available_plans_list,
                    "prompt": prompt,
                    "model_output": output,
                    "plan": plan,
                    "replanning_context": replanning_context,
                    "flags": flags,
                    "attempt": attempt,
                    "usage": usage,
                })
                break
            
            # prompt = prompt + "\nPlease ensure you output your chosen action wrapped EXACTLY in <|action|> tags. Examples: <|action|>go grasp target object <xxx> (xxx)<|action|>"
            # if self.communication:
            #     prompt = prompt + "\nIf you choose 'send a message', you MUST also include your message wrapped in <|message|> tags. Examples: <|action|>send a message<|action|> <|message|>xxx<|message|>"
            chat_prompt = [{"role": "user", "content": prompt}]
            output, usage = self.generator(chat_prompt)
            self.total_cost += self._usage_total(usage)

        info.update({"num_available_actions": num,
                     "parse_exception": flags,
                     "plan": plan,
                     "replanning_context": replanning_context,
                     "total_cost": self.total_cost})
        return plan, info
