from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx

from .base import ReasoningBackend
from ..execution import check_cancelled


def _detect_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".gif":
        return "image/gif"
    if suffix in (".tif", ".tiff"):
        return "image/tiff"
    return "image/png"


class OpenAICompatBackend(ReasoningBackend):
    def __init__(self, settings):
        self.base_url = settings.api.base_url.rstrip("/")
        self.model = settings.api.api_model
        self.api_key_env = settings.api.api_key_env
        self.vision_enabled = settings.api.vision_enabled
        self.timeout_seconds = settings.api.timeout_seconds
        self._tested = False

    def _flatten(self, messages: list) -> list[dict]:
        out = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                out.append({"role": message["role"], "content": content})
                continue
            parts = []
            for chunk in content:
                if chunk.get("type") == "text":
                    parts.append({"type": "text", "text": chunk["text"]})
                elif chunk.get("type") == "image" and chunk.get("path"):
                    if not self.vision_enabled:
                        raise RuntimeError("Remote vision capability is disabled. Configure a vision-capable provider/model and set api.vision_enabled=true.")
                    img_path = Path(chunk["path"])
                    import io
                    from PIL import Image
                    from ...utils.dicom import load_study_image
                    array, _ = load_study_image(img_path)
                    buffer = io.BytesIO()
                    Image.fromarray(array).convert("RGB").save(buffer, format="PNG")
                    data, mime = buffer.getvalue(), "image/png"
                    uri = f"data:{mime};base64," + base64.b64encode(data).decode()
                    parts.append({"type": "image_url", "image_url": {"url": uri}})
            out.append({"role": message["role"], "content": parts})
        return out

    def generate(self, messages: list, on_token=None, task: str = "report", response_schema: dict | None = None) -> str:
        check_cancelled()
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key: environment variable '{self.api_key_env}' is not set. "
                f"Set {self.api_key_env} to use remote reasoner '{self.model}' or switch models.reasoner in config."
            )
        headers = {"Authorization": f"Bearer {api_key}"}
        body = {"model": self.model, "messages": self._flatten(messages), "stream": True}
        if response_schema is not None:
            # Fail explicitly if a provider cannot honor this request; never
            # silently rerun a failed schema request with looser constraints.
            body["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "ClinicalReport", "schema": response_schema, "strict": True,
            }}
        text_parts = []
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                with client.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=headers) as response:
                    if response.status_code >= 400:
                        response.read()
                        response.raise_for_status()
                    for line in response.iter_lines():
                        check_cancelled()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError("Malformed JSON in provider event stream") from exc
                        if "error" in event:
                            raise RuntimeError(f"Provider stream error: {event['error']}")
                        choices = event.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}).get("content")
                        if isinstance(delta, str) and delta:
                            text_parts.append(delta)
                            if on_token:
                                on_token(delta)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
            raise RuntimeError(
                f"Remote reasoning provider returned HTTP {status}: {detail}. "
                f"Verify provider credentials and ensure model '{self.model}' supports vision/multimodal input."
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Failed to communicate with remote reasoning provider at '{self.base_url}': {exc}"
            ) from exc
        check_cancelled()
        text = "".join(text_parts)
        if not text.strip():
            raise RuntimeError("Remote provider returned an empty response")
        self._tested = True
        return text
