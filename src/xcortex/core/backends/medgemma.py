from __future__ import annotations

import queue
import hashlib
import json
import threading
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from .base import ReasoningBackend
from ..execution import ExecutionCancelled, check_cancelled, current_cancel_event


class _FinalAnswerFilter:
    """Drop MedGemma thought blocks before any text reaches a consumer.

    Google's MedGemma quick-start uses ``<unused94>thought\n...<unused95>``.
    Decoder chunks can split either delimiter anywhere, so retain only a possible
    delimiter suffix between calls. Thought contents are discarded, not buffered.
    """

    _OPEN = "<unused94>"
    _CLOSE = "<unused95>"

    def __init__(self):
        self._pending = ""
        self._depth = 0
        self.saw_thought = False

    def feed(self, text: str) -> str:
        self._pending += text
        visible = []
        while self._pending:
            positions = [(self._pending.find(marker), marker) for marker in (self._OPEN, self._CLOSE)]
            positions = [(index, marker) for index, marker in positions if index >= 0]
            if positions:
                index, marker = min(positions)
                if not self._depth:
                    visible.append(self._pending[:index])
                self._pending = self._pending[index + len(marker):]
                if marker == self._OPEN:
                    self._depth += 1
                    self.saw_thought = True
                elif self._depth:
                    self._depth -= 1
                else:
                    raise RuntimeError("MedGemma returned an unmatched thinking delimiter; retry generation")
                continue

            # A prefix such as '<unus' might become a delimiter in the next
            # chunk. Never stream it speculatively and try to retract it later.
            suffix_size = max(
                (size for marker in (self._OPEN, self._CLOSE)
                 for size in range(1, len(marker)) if self._pending.endswith(marker[:size])),
                default=0,
            )
            ready = self._pending[:-suffix_size] if suffix_size else self._pending
            if not self._depth:
                visible.append(ready)
            self._pending = self._pending[-suffix_size:] if suffix_size else ""
            break
        return "".join(visible)

    def finish(self) -> None:
        if self._depth or self._pending:
            raise RuntimeError(
                "MedGemma returned an unfinished thinking block or delimiter; "
                "increase models.max_new_tokens or simplify the prompt and retry"
            )


class MedGemmaBackend(ReasoningBackend):
    def __init__(self, settings):
        self.settings = settings
        self.model_id = settings.models.medgemma_id
        self._resolve = settings.resolve_model
        self.dtype_name = settings.models.dtype
        self.device_name = settings.models.device
        self._model: Any = None
        self._processor: Any = None
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._tested = False
        self.last_run: dict[str, Any] = {}
        self.load_metadata: dict[str, Any] = {}
        self._input_trace: dict[str, Any] = {}

    def wait_until_idle(self):
        """Keep server admission occupied until cancelled native work has ended."""
        with self._inference_lock:
            pass

    def release_model(self):
        """End the reasoner's CPU residency before the next collection stage."""
        import gc

        with self._inference_lock:
            with self._lock:
                self._model = None
                self._processor = None
                self._input_trace = {}
            gc.collect()

    def _resolve_device(self):
        import torch
        return torch.device(self.device_name if self.device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    def _resolve_dtype(self, device):
        import torch
        table = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
        if self.dtype_name not in table:
            raise ValueError(f"Unsupported dtype: {self.dtype_name}")
        dtype = table[self.dtype_name]
        # BF16 keeps the CPU weights near 8.6 GB, including on CPUs where BF16
        # operations are emulated. FP32 would double their memory footprint.
        return torch.float32 if device.type == "cpu" and dtype == torch.float16 else dtype

    def _ensure_loaded(self, record: dict | None = None):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            check_cancelled()
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
            device = self._resolve_device()
            dtype = self._resolve_dtype(device)
            quantization = getattr(self.settings.models, "cpu_quantization", "none")
            if quantization not in {"none", "dynamic_int8"}:
                raise ValueError(f"Unsupported CPU quantization: {quantization}")
            if quantization != "none" and device.type != "cpu":
                raise ValueError("cpu_quantization requires device=cpu")
            resolved = self._resolve(self.model_id)
            load_dtype = torch.bfloat16 if quantization == "dynamic_int8" else dtype
            kwargs = {"torch_dtype": load_dtype, "low_cpu_mem_usage": True, "local_files_only": self.settings.models.local_files_only}
            if device.type == "cpu":
                torch.set_num_threads(max(1, self.settings.models.cpu_threads))
                offload = Path(self.settings.paths.weights) / "offload"
                offload.mkdir(parents=True, exist_ok=True)
                if quantization == "dynamic_int8":
                    # Load mapped BF16 checkpoint tensors, then pack one Linear
                    # at a time. Never materialize a full FP32 model copy.
                    kwargs.update(device_map="cpu")
                else:
                    kwargs.update(device_map="auto", max_memory={"cpu": f"{max(1, self.settings.models.cpu_memory_gib)}GiB"}, offload_folder=str(offload))
            else:
                kwargs["device_map"] = str(device)
            try:
                processor = AutoProcessor.from_pretrained(resolved, local_files_only=self.settings.models.local_files_only)
                quantization_metadata: dict[str, Any] = {"mode": "none"}
                if quantization == "dynamic_int8":
                    from .cpu_quantization import load_cpu_int8_model
                    def load_progress(done: int, total: int):
                        if record is not None:
                            record["load_progress"] = {"packed_linear_layers": done, "total_linear_layers": total}
                    model, quantization_metadata = load_cpu_int8_model(
                        resolved, local_files_only=self.settings.models.local_files_only,
                        on_progress=load_progress,
                    )
                else:
                    model = AutoModelForImageTextToText.from_pretrained(resolved, **kwargs)
                model.eval()
                self._processor, self._model = processor, model
                dtype_counts: Counter[str] = Counter()
                for parameter in model.parameters():
                    dtype_counts[str(parameter.dtype)] += parameter.numel()
                self.load_metadata = {
                    "resolved_model": str(resolved), "requested_dtype": self.dtype_name,
                    "checkpoint_load_dtype": "streamed_per_tensor_float32" if quantization == "dynamic_int8" else str(load_dtype),
                    "parameter_dtype_counts": dict(dtype_counts),
                    "quantization": quantization_metadata,
                    "device_map": getattr(model, "hf_device_map", {"": str(device)}),
                    "attention_implementation": getattr(model.config, "_attn_implementation", None),
                    "cpu_threads": torch.get_num_threads(),
                    "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                    "torch_version": torch.__version__,
                }
                check_cancelled()
            except ExecutionCancelled:
                raise
            except Exception as exc:
                raise RuntimeError(f"Failed to initialize MedGemma from '{resolved}': {exc}. Check the local checkpoint and CPU execution settings.") from exc

    def _build_inputs(self, messages: list):
        from PIL import Image
        image_paths, text_messages = [], []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                text_messages.append({"role": message["role"], "content": content})
                continue
            parts = []
            for chunk in content:
                if chunk.get("type") == "image" and chunk.get("path"):
                    image_paths.append(chunk["path"])
                    parts.append({"type": "image"})
                elif chunk.get("type") == "text":
                    parts.append({"type": "text", "text": chunk["text"]})
            text_messages.append({"role": message["role"], "content": parts})
        prompt = self._processor.apply_chat_template(text_messages, add_generation_prompt=True, tokenize=False)
        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        result = self._processor(text=prompt, **({"images": images} if images else {}),
                                 return_tensors="pt", add_special_tokens=False)
        self._input_trace = {
            "rendered_prompt": prompt, "image_sources": [{
                "path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "size": list(image.size),
            } for path, image in zip(image_paths, images)],
            "input_ids": result["input_ids"][0].tolist(),
            "input_token_count": int(result["input_ids"].shape[-1]),
            "pixel_values_shape": list(result["pixel_values"].shape) if "pixel_values" in result else [],
        }
        return result

    def generate(self, messages: list, on_token=None, task: str = "report", response_schema: dict | None = None) -> str:
        from .telemetry import ResourceMonitor

        started = time.perf_counter()
        record: dict[str, Any] = {
            "model_id": self.model_id, "started_at": datetime.now(timezone.utc).isoformat(),
            "task": task, "generation_count": 0, "messages": deepcopy(messages),
            "generation_settings": {"max_new_tokens": self.settings.models.max_new_tokens, "do_sample": False},
            "response_schema": response_schema, "timings": {}, "status": "preparing",
        }
        self.last_run = record
        monitor = ResourceMonitor().start()
        try:
            text = self._generate_once(messages, on_token, task, response_schema, record)
            record.update(status="completed", raw_response=text)
            return text
        except BaseException as exc:
            record.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            if isinstance(exc, TimeoutError):
                record["finish_reason"] = "timeout"
            elif isinstance(exc, ExecutionCancelled):
                record["finish_reason"] = "cancelled"
            raise
        finally:
            record["timings"]["total_seconds"] = time.perf_counter() - started
            record["memory"] = monitor.stop()

    def _generate_once(self, messages: list, on_token, task: str, response_schema: dict | None, record: dict) -> str:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
        check_cancelled()
        cancel = current_cancel_event()
        stop = threading.Event()
        deadline = time.monotonic() + self.settings.models.generation_timeout_seconds
        while not self._inference_lock.acquire(timeout=0.1):
            check_cancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError("MedGemma is still completing a previous inference")
        try:
            check_cancelled()
            load_start = time.perf_counter()
            record["phase"] = "loading"
            self._ensure_loaded(record)
            record["timings"]["load_seconds"] = time.perf_counter() - load_start
            record["model_runtime"] = deepcopy(self.load_metadata)
            # Loading is timed separately; the native generation allowance starts
            # once the model is ready. The lock also protects model residency.
            deadline = time.monotonic() + self.settings.models.generation_timeout_seconds
            prepared = time.perf_counter()
            record["phase"] = "input_preparation"
            inputs = self._build_inputs(messages).to(self._model.device)
            record["timings"]["input_preparation_seconds"] = time.perf_counter() - prepared
            record["input"] = deepcopy(self._input_trace)
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            report_constraint = None
            grammar_start = time.perf_counter()
            record["phase"] = "grammar_preparation"
            if task == "report" and getattr(self.settings.models, "structured_reports", True):
                from ...report.constraints import build_report_constraint
                report_constraint = build_report_constraint(tokenizer, response_schema) if response_schema is not None else build_report_constraint(tokenizer)
            record["timings"]["grammar_seconds"] = time.perf_counter() - grammar_start
            check_cancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError("MedGemma preparation exceeded its configured timeout")
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=0.25)
        except BaseException:
            self._inference_lock.release()
            raise

        class StopRequested(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
                requested = stop.is_set() or (cancel is not None and cancel.is_set()) or time.monotonic() >= deadline
                is_complete = getattr(report_constraint, "is_complete", None)
                if not requested and callable(is_complete) and input_ids.shape[0] == 1:
                    if is_complete(input_ids[0]):
                        # Complete means the whole constrained JSON object, never
                        # just an early answer_choice or a repaired partial stream.
                        record["structured_json_complete"] = True
                        requested = True
                return cast(torch.BoolTensor, torch.full(
                    (input_ids.shape[0],), requested, dtype=torch.bool, device=input_ids.device,
                ))

        errors, finished = [], threading.Event()
        kwargs: dict[str, Any] = dict(**inputs, streamer=streamer, max_new_tokens=self.settings.models.max_new_tokens,
                      do_sample=False, stopping_criteria=StoppingCriteriaList([StopRequested()]))
        if report_constraint is not None:
            kwargs["prefix_allowed_tokens_fn"] = report_constraint
        generation_start = time.perf_counter()
        record["phase"] = "generation"
        original_put = getattr(streamer, "put", None)
        if callable(original_put):
            prompt_seen = False

            def timed_put(value):
                nonlocal prompt_seen
                if prompt_seen:
                    record["timings"].setdefault("first_generated_token_seconds", time.perf_counter() - generation_start)
                else:
                    prompt_seen = True
                original_put(value)

            streamer.put = timed_put

        def worker():
            try:
                with torch.inference_mode():
                    record["generation_count"] = 1
                    output = self._model.generate(**kwargs)
                    if isinstance(output, torch.Tensor) and "input_ids" in inputs:
                        generated = output[0, inputs["input_ids"].shape[-1]:].tolist()
                        record["output_ids"] = generated
                        record["output_token_count"] = len(generated)
                        eos = getattr(getattr(self._model, "generation_config", None), "eos_token_id", [])
                        eos = eos if isinstance(eos, (tuple, list)) else [eos]
                        record["finish_reason"] = "eos" if generated and generated[-1] in eos else (
                            "structured_json_complete" if record.get("structured_json_complete") else (
                                "max_new_tokens" if len(generated) >= self.settings.models.max_new_tokens else "stopping_criterion"))
            except BaseException as exc:
                errors.append(exc)
                streamer.end()
            finally:
                # Native prefill cannot be killed safely. Keep the lock until it
                # really ends, even if the waiting request has timed out.
                self._inference_lock.release()
                finished.set()

        thread = threading.Thread(target=worker, daemon=True, name="xcortex-medgemma")
        try:
            thread.start()
        except BaseException:
            self._inference_lock.release()
            raise
        parts, answer_filter = [], _FinalAnswerFilter()
        try:
            while True:
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError("MedGemma generation exceeded its configured timeout")
                if errors:
                    raise RuntimeError(f"MedGemma generation failed: {errors[0]}") from errors[0]
                try:
                    token = next(streamer)
                except queue.Empty:
                    if finished.is_set():
                        break
                    continue
                except StopIteration:
                    break
                visible = answer_filter.feed(token)
                if visible:
                    record["timings"].setdefault("first_visible_text_seconds", time.perf_counter() - generation_start)
                    parts.append(visible)
                    if on_token:
                        on_token(visible)
            # EOS can be enqueued before generate returns. Do not publish a
            # successful result until post-generation errors have been checked.
            while not finished.wait(timeout=0.05):
                check_cancelled()
                if time.monotonic() >= deadline:
                    raise TimeoutError("MedGemma generation exceeded its configured timeout")
            check_cancelled()
            if errors:
                raise RuntimeError(f"MedGemma generation failed: {errors[0]}") from errors[0]
            answer_filter.finish()
            text = "".join(parts)
            record["timings"]["generation_seconds"] = time.perf_counter() - generation_start
            if not text.strip():
                if answer_filter.saw_thought:
                    raise RuntimeError(
                        "MedGemma completed thinking without a final answer; "
                        "increase models.max_new_tokens or simplify the prompt and retry"
                    )
                raise RuntimeError("MedGemma returned an empty response")
            self._tested = True
            return text
        finally:
            record["streamed_response"] = "".join(parts)
            generation_seconds = time.perf_counter() - generation_start
            record["timings"]["generation_seconds"] = generation_seconds
            first = record["timings"].get("first_generated_token_seconds")
            if first is not None:
                record["timings"]["decode_after_first_token_seconds"] = max(0, generation_seconds - first)
                count = record.get("output_token_count", 0)
                if count > 1 and generation_seconds > first:
                    record["decode_tokens_per_second"] = (count - 1) / (generation_seconds - first)
            stop.set()
            thread.join(timeout=1.0)
