from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union
import os
import requests


PromptT = Union[str, Sequence[Mapping[str, Any]]]


def _prompt_to_messages(prompt: PromptT) -> List[Dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]

    messages: List[Dict[str, str]] = []
    for msg in prompt:
        role = str(msg.get("role", "user") or "user")
        content = msg.get("content", "")
        if content is None:
            content = ""
        messages.append({"role": role, "content": str(content)})
    return messages


def gemma_generate(
    prompt: PromptT,
    model: str = "latest", # "latest" , "31b"
    host: Optional[str] = None,
    think: bool = False,
) -> Tuple[str, int]:
    if not host:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    endpoint = "/api/chat" if think else "/api/generate"
    url = host.rstrip("/") + endpoint
    model = "gemma4:" + model

    messages = _prompt_to_messages(prompt)

    if think:
        think_prefix = "<|think|> "
        if messages and messages[0].get("role") == "system":
            content = messages[0].get("content", "")
            if not str(content).startswith(think_prefix):
                messages[0]["content"] = think_prefix + str(content)
        else:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": "<|think|> ",
                },
            )

    options = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    if think:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
    else:
        prompt_text = "\n".join(
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages
        )
        payload = {
            "model": model,
            "prompt": prompt_text,
            "stream": False,
            "options": options,
        }

    resp = requests.post(url, json=payload, timeout=1000)
    if resp.status_code == 200:
        data = resp.json()
        prompt_tokens = int(data.get("prompt_eval_count") or 0)
        completion_tokens = int(data.get("eval_count") or 0)
        total_tokens = prompt_tokens + completion_tokens

        if total_tokens == 0:
            context = data.get("context")
            if isinstance(context, list):
                total_tokens = len(context)

        if think:
            message = data.get("message")
            if not isinstance(message, dict):
                raise ValueError("Ollama response missing 'message' object")
            text = message.get("content")
            if not isinstance(text, str):
                raise ValueError("Ollama response missing 'message.content' text")
            return text, total_tokens

        text = data.get("response")
        if not isinstance(text, str):
            raise ValueError("Ollama response missing 'response' text")
        return text, total_tokens
    else:
        raise ValueError(f"Ollama response status code {resp.status_code}")

if __name__ == "__main__":
    prompt = "Hi, are you ready for the multi-agent task?"
    text, usage = gemma_generate(prompt, think=True)
    print(text)
    print("usage:", usage)
