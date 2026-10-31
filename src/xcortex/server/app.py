from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import os
import queue
import re
import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from ..config import Settings
from ..core.agent import Agent
from ..core.messages import Study
from ..model_files import inspect_checkpoint, inspect_model
from ..report.pdf import render_clinical_pdf
from ..utils.dicom import SUPPORTED_SUFFIXES, load_study_image

UI_DIR = Path(__file__).resolve().parents[1] / "ui" / "static"
SAFE_NAME = re.compile(r"[A-Za-z0-9_.\-]+")
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB limit
SSE_QUEUE_TIMEOUT = 120.0  # seconds
SSE_REQUEST_TIMEOUT = 1800.0
SSE_EVENT_QUEUE_SIZE = 256
MAX_INFERENCE_WORKERS = 1
REGION_ALIASES = {
    "chest": "chest", "thorax": "chest", "cxr": "chest", "lung": "chest", "lungs": "chest",
    "wrist": "wrist", "wrists": "wrist", "carpus": "wrist",
    "elbow": "elbow", "elbows": "elbow", "finger": "finger", "fingers": "finger",
    "hand": "hand", "hands": "hand", "forearm": "forearm", "forearms": "forearm",
    "humerus": "humerus", "upper arm": "humerus", "shoulder": "shoulder", "shoulders": "shoulder",
    "musculoskeletal": "musculoskeletal", "ankle": "ankle", "knee": "knee", "hip": "hip",
    "foot": "foot", "feet": "foot", "tibia": "tibia", "femur": "femur", "spine": "spine",
    "pelvis": "pelvis", "lumbar": "lumbar", "cervical": "cervical", "thoracic": "thoracic",
}


class SessionBusy(RuntimeError):
    pass


class InferenceCancelled(RuntimeError):
    pass


class AnalyzeRequest(BaseModel):
    session_id: str
    history: str = ""
    region: str | None = Field(default=None, max_length=64)


class AskRequest(BaseModel):
    session_id: str
    question: str


class SessionStore:
    """Atomic storage for one server process; independent server workers are unsupported."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._claims: dict[str, str] = {}
        self.sessions: dict[str, dict] = {}
        self.recovery = {"status": "clean", "interrupted_sessions": []}
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                records = data["sessions"]
                if not isinstance(records, dict) or any(
                    not isinstance(key, str)
                    or not isinstance(record, dict)
                    or record.get("id") != key
                    for key, record in records.items()
                ):
                    raise ValueError("invalid session records")
                self.sessions = records
            except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError, ValueError) as exc:
                quarantine = path.with_name(f"{path.name}.corrupt.{time.time_ns()}.{uuid.uuid4().hex[:8]}")
                # A failed rename aborts startup. Never overwrite the only copy of corrupt data.
                path.replace(quarantine)
                self.recovery = {
                    "status": "quarantined",
                    "quarantine_path": str(quarantine),
                    "message": f"Invalid session data preserved: {exc}",
                    "interrupted_sessions": [],
                }
        interrupted = []
        for session_id, record in self.sessions.items():
            if record.get("status") == "running" or record.get("active_operation"):
                operation = record.pop("active_operation", {})
                kind = operation.get("kind", "analyze") if isinstance(operation, dict) else "analyze"
                record.update(
                    status="ready" if kind == "ask" and record.get("report") else "interrupted",
                    last_operation_status="interrupted",
                    last_error="The server stopped before inference completed. Submit the request again.",
                )
                interrupted.append(session_id)
        if interrupted:
            self.recovery = {"status": "recovered", "interrupted_sessions": interrupted}
            self._flush()

    def create(self, session_id: str, record: dict) -> None:
        with self._lock:
            if session_id in self.sessions:
                raise ValueError("session already exists")
            self.sessions[session_id] = copy.deepcopy(record)
            try:
                self._flush()
            except Exception:
                self.sessions.pop(session_id)
                raise

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            record = self.sessions.get(session_id)
            return copy.deepcopy(record) if record is not None else None

    def update(self, session_id: str, **fields) -> None:
        with self._lock:
            record = self.sessions.get(session_id)
            if record is None:
                return
            previous = copy.deepcopy(record)
            record.update(copy.deepcopy(fields))
            try:
                self._flush()
            except Exception:
                self.sessions[session_id] = previous
                raise

    def busy(self, session_id: str) -> bool:
        with self._lock:
            record = self.sessions.get(session_id) or {}
            return session_id in self._claims or record.get("status") == "running"

    def claim(self, session_id: str, kind: str, *, turn: dict | None = None, **fields) -> tuple[str, dict]:
        """Claim analysis and follow-ups under the same lock, including the user turn."""
        with self._lock:
            if session_id not in self.sessions:
                raise KeyError(session_id)
            if self.busy(session_id):
                raise SessionBusy("Inference is already in progress for this session.")
            token = uuid.uuid4().hex
            previous = copy.deepcopy(self.sessions[session_id])
            record = self.sessions[session_id]
            record.update(copy.deepcopy(fields))
            if turn is not None:
                record.setdefault("turns", []).append(copy.deepcopy(turn))
            record["active_operation"] = {"id": token, "kind": kind, "started_at": time.time()}
            record["last_operation_status"] = "running"
            record["last_error"] = None
            self._claims[session_id] = token
            try:
                self._flush()
            except Exception:
                self.sessions[session_id] = previous
                self._claims.pop(session_id, None)
                raise
            return token, copy.deepcopy(record)

    def update_claim(self, session_id: str, token: str, **fields) -> bool:
        with self._lock:
            if self._claims.get(session_id) != token:
                return False
            self.update(session_id, **fields)
            return True

    def finish(self, session_id: str, token: str, *, assistant_turn: dict | None = None, **fields) -> bool:
        with self._lock:
            if self._claims.get(session_id) != token:
                return False
            previous = copy.deepcopy(self.sessions[session_id])
            record = self.sessions[session_id]
            record.update(copy.deepcopy(fields))
            if assistant_turn is not None:
                record.setdefault("turns", []).append(copy.deepcopy(assistant_turn))
            record.pop("active_operation", None)
            try:
                self._flush()
            except Exception:
                self.sessions[session_id] = previous
                raise
            self._claims.pop(session_id)
            return True

    def listing(self) -> list[dict]:
        with self._lock:
            items = [
                {
                    "id": record["id"],
                    "title": record.get("title", "study"),
                    "ts": record.get("ts", 0),
                    "done": record.get("status") == "ready" or bool(record.get("report")),
                    "status": record.get("status", "ready" if record.get("report") else "pending"),
                    "region": ((record.get("report") or {}).get("study") or {}).get("region")
                    or record.get("meta", {}).get("region"),
                }
                for record in self.sessions.values()
            ]
            return sorted(items, key=lambda r: r["ts"], reverse=True)

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = self.path.parent / f"{self.path.name}.tmp.{uuid.uuid4().hex[:8]}"
        try:
            tmp_file.write_text(
                json.dumps({"sessions": self.sessions}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_file.replace(self.path)
        except Exception:
            if tmp_file.is_file():
                tmp_file.unlink(missing_ok=True)
            raise


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


@dataclass
class InferenceRun:
    session_id: str
    token: str
    kind: str
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=SSE_EVENT_QUEUE_SIZE))
    cancel_event: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: float = field(default_factory=time.monotonic)
    last_event: float = field(default_factory=time.monotonic)
    error: str | None = None
    outcome: str = "running"


def _dependencies_present(*names: str) -> bool:
    try:
        return all(importlib.util.find_spec(name) is not None for name in names)
    except (ValueError, ModuleNotFoundError):
        return False


def _canonical_region(value: str, *, strict: bool = True) -> str:
    normalized = re.sub(r"[\s_-]+", " ", value.strip().lower())
    if normalized in REGION_ALIASES:
        return REGION_ALIASES[normalized]
    if strict:
        raise HTTPException(status_code=400, detail="Unknown anatomical region. Supply a supported body part such as chest, wrist, or shoulder.")
    return normalized


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.load()
    settings.ensure_dirs()

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            shutdown()

    app = FastAPI(title="X-Cortex", lifespan=lifespan)
    app.state.settings = settings
    app.state.agent = Agent(settings)
    app.state.sessions = SessionStore(Path(settings.paths.state))
    # SessionStore and model instances are intentionally shared by one server process.
    app.state.admission_lock = threading.Lock()
    app.state.inference_slots = threading.BoundedSemaphore(
        max(1, int(getattr(settings.agent, "max_concurrent_runs", MAX_INFERENCE_WORKERS)))
    )
    app.state.active_runs = {}
    app.state.runtime_lock = threading.Lock()
    app.state.tested_tools = set()
    app.state.last_successful_analysis = None
    app.state.last_inference_error = None

    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(UI_DIR / "index.html")

    @app.get("/api/health")
    def health():
        mode = settings.app.mode
        details = {
            "status": "ok",
            "mode": mode,
            "storage": {"process_support": "single-process", "recovery": app.state.sessions.recovery},
            "active_runs": len(app.state.active_runs),
            "last_inference_error": app.state.last_inference_error,
        }
        if mode == "mock":
            details["ready"] = True
            details["readiness_scope"] = "synthetic mock workflow"
            return details

        def hf_status(repo_id: str, *, configured: bool, loaded: bool, tested: bool) -> dict:
            resolved = settings.model_path(repo_id)
            status = inspect_model(resolved)
            dependencies = ["torch", "transformers", "accelerate", "sentencepiece"]
            if repo_id == settings.models.medgemma_id and settings.models.structured_reports:
                dependencies.append("lmformatenforcer")
            return {
                "configured": configured,
                "dependencies_installed": _dependencies_present(*dependencies),
                **status,
                "loaded": loaded,
                "tested": tested,
            }

        registry = app.state.agent.registry
        tools = {tool.name: tool for tool in registry}
        tested = app.state.tested_tools.copy()
        segmentation = tools.get("segmentation")
        rag = tools.get("rag")
        triage = tools.get("triage")
        siglip_loaded = any(
            getattr(getattr(tool, "_encoder", None), "_model", None) is not None
            for tool in (triage, rag)
            if tool is not None
        )
        uses_siglip = settings.models.triage_backend == "zeroshot" or rag is not None
        components = {
            "medsiglip": hf_status(
                settings.models.siglip_id,
                configured=uses_siglip,
                loaded=siglip_loaded,
                tested=(settings.models.triage_backend == "zeroshot" and "triage" in tested) or "rag" in tested,
            ),
            "sam2": {
                "configured": segmentation is not None,
                "dependencies_installed": _dependencies_present("torch", "sam2"),
                **inspect_checkpoint(settings.models.sam2_checkpoint),
                "loaded": getattr(segmentation, "_generator", None) is not None,
                "tested": "segmentation" in tested,
            },
        }
        backend = app.state.agent.backend
        reasoner = settings.models.reasoner
        if reasoner == "medgemma":
            components["medgemma"] = hf_status(
                settings.models.medgemma_id,
                configured=True,
                loaded=getattr(backend, "_model", None) is not None,
                tested=bool(getattr(backend, "_tested", False)),
            )
        elif reasoner == "openai":
            components["openai"] = {
                "configured": True,
                "dependencies_installed": _dependencies_present("httpx"),
                "installed": bool(os.environ.get(settings.api.api_key_env, "").strip()),
                "loaded": app.state.last_successful_analysis is not None,
                "tested": app.state.last_successful_analysis is not None,
                "vision_capability_tested": app.state.last_successful_analysis is not None,
                "reason": "API credentials alone do not verify model vision capability",
            }
        if settings.models.triage_backend == "anomalyclip":
            from ..models.anomalyclip.artifacts import inspect_anomalyclip
            status = inspect_anomalyclip(
                Path(settings.paths.weights), checkpoint=settings.models.anomalyclip_checkpoint,
                backbone=settings.models.anomalyclip_backbone,
            )
            components["detector"] = {
                "configured": True,
                "dependencies_installed": _dependencies_present("torch", "torchvision", "scipy"),
                **status,
                "loaded": getattr(getattr(triage, "_runtime", None), "model", None) is not None,
                "tested": bool(getattr(triage, "_tested", False)),
            }
        elif settings.models.triage_backend != "zeroshot":
            checkpoint = (
                getattr(triage, "checkpoint_path", settings.models.triage_checkpoint) if settings.models.triage_backend == "chestxray"
                else settings.models.triage_checkpoint
            )
            components["detector"] = {
                "configured": True,
                "dependencies_installed": _dependencies_present(
                    "torch", "torchvision",
                    *( ["torchxrayvision"] if settings.models.triage_backend == "chestxray" else [] ),
                ),
                **inspect_checkpoint(checkpoint),
                "loaded": getattr(triage, "_model", None) is not None,
                "tested": "triage" in tested,
            }
        if rag is not None:
            collection = getattr(getattr(rag, "_store", None), "_collection", None)
            try:
                reference_count = collection.count() if collection is not None else None
            except Exception:
                reference_count = None
            components["retrieval"] = {
                "configured": True,
                "dependencies_installed": _dependencies_present("chromadb"),
                "installed": Path(settings.paths.db).is_dir(),
                "loaded": collection is not None,
                "tested": "rag" in tested and bool(reference_count),
                "reference_count": reference_count,
            }
        for component in components.values():
            component["ready"] = all(
                component.get(key, False) for key in ("dependencies_installed", "installed", "loaded", "tested")
            )
        details.update(
            reasoner=reasoner,
            components=components,
            ready=bool(app.state.last_successful_analysis)
            and not app.state.last_inference_error
            and all(component["ready"] for component in components.values() if component["configured"]),
            readiness_scope="successful inference in this server process; clinical performance is not established",
            last_successful_analysis=app.state.last_successful_analysis,
            sam2_ready=components["sam2"]["ready"],
        )
        if reasoner in components:
            details[f"{reasoner}_ready"] = components[reasoner]["ready"]
        return details

    @app.get("/api/config")
    def config():
        return {
            "mode": settings.app.mode,
            "tools": [{"id": tool.name, "label": tool.label} for tool in app.state.agent.registry],
            "models": {
                "reasoner": settings.models.reasoner,
                "medgemma": settings.models.medgemma_id,
                "siglip": settings.models.siglip_id,
                "sam2_variant": settings.models.sam2_variant,
                "triage_threshold": settings.models.triage_threshold,
                    "triage_backend": settings.models.triage_backend,
                    "default_region": getattr(settings.models, "default_region", ""),
            },
            "rag_top_k": settings.rag.top_k,
        }

    @app.post("/api/upload")
    def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=400, detail="unsupported file type")

        content = file.file.read(MAX_UPLOAD_SIZE + 1)
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail=f"File exceeds maximum size of {MAX_UPLOAD_SIZE // (1024 * 1024)} MB")

        session_id = uuid.uuid4().hex[:12]
        uploads = Path(settings.paths.data) / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        target = uploads / f"{session_id}{suffix}"
        target.write_bytes(content)

        try:
            image, meta = load_study_image(target)
        except Exception as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"unreadable image: {exc}")

        run_dir = Path(settings.paths.artifacts) / session_id
        run_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(run_dir / "preview.png")
        preview_url = f"/a/{session_id}/preview.png"

        app.state.sessions.create(
            session_id,
            {
                "id": session_id,
                "title": file.filename or "study",
                "ts": time.time(),
                "status": "pending",
                "image_path": str(target),
                "preview_url": preview_url,
                "meta": meta,
                "history": "",
                "turns": [],
            },
        )
        return {"session_id": session_id, "preview_url": preview_url, "meta": meta}

    def admit(session_id: str, kind: str, *, region: str | None = None, **fields) -> tuple[InferenceRun, dict]:
        with app.state.admission_lock:
            record = app.state.sessions.get(session_id)
            if record is None:
                raise HTTPException(status_code=404, detail="unknown session")
            if app.state.sessions.busy(session_id):
                raise HTTPException(status_code=409, detail="Inference is already in progress for this session.")
            if kind == "ask" and (record.get("status") != "ready" or not record.get("report")):
                raise HTTPException(status_code=409, detail="Complete analysis before asking a follow-up question.")
            if kind == "analyze":
                meta = copy.deepcopy(record.get("meta") or {})
                supplied = _canonical_region(region) if region is not None else None
                existing = str(meta.get("region") or "").strip()
                if supplied and existing and supplied != _canonical_region(existing, strict=False):
                    raise HTTPException(status_code=400, detail="Supplied region conflicts with the image metadata. Verify the study and its body part.")
                if not existing:
                    configured = str(getattr(settings.models, "default_region", "") or "").strip()
                    if supplied:
                        meta.update(region=supplied, region_source="user")
                    elif configured:
                        meta.update(region=_canonical_region(configured), region_source="configured")
                fields["meta"] = meta
            if not app.state.inference_slots.acquire(blocking=False):
                raise HTTPException(status_code=503, detail="Inference capacity is busy. Retry after the active request completes.")
            try:
                token, session = app.state.sessions.claim(session_id, kind, **fields)
            except Exception:
                app.state.inference_slots.release()
                raise
            run = InferenceRun(session_id, token, kind)
            app.state.active_runs[token] = run
            return run, session

    def cancel(run: InferenceRun, outcome: str, message: str) -> None:
        with run.lock:
            if run.finished.is_set() or run.cancel_event.is_set() or run.outcome != "running":
                return
            run.cancel_event.set()
            run.outcome, run.error = outcome, message
            app.state.last_inference_error = message
            app.state.sessions.update_claim(
                run.session_id,
                run.token,
                status=outcome if run.kind == "analyze" else "ready",
                last_operation_status=outcome,
                last_error=message,
            )
            # Retain the claim and admission slot until the worker really exits. Torch
            # calls cannot safely be killed from another Python thread.

    def stream(run: InferenceRun, invoke) -> StreamingResponse:
        request_timeout = max(0.01, float(getattr(settings.agent, "request_timeout_seconds", SSE_REQUEST_TIMEOUT)))
        idle_timeout = max(0.01, float(getattr(settings.agent, "stream_idle_timeout_seconds", SSE_QUEUE_TIMEOUT)))

        def check_active():
            if run.cancel_event.is_set() or run.finished.is_set() or run.outcome != "running":
                raise InferenceCancelled("Inference was cancelled")
            if time.monotonic() - run.started >= request_timeout:
                cancel(run, "timed_out", "Inference request timed out")
                raise InferenceCancelled("Inference request timed out")

        def on_activity():
            check_active()
            run.last_event = time.monotonic()

        def on_event(kind, data):
            check_active()
            if kind == "stage" and isinstance(data, dict) and data.get("status") == "done":
                with app.state.runtime_lock:
                    app.state.tested_tools.add(data.get("tool"))
            while not run.cancel_event.is_set():
                try:
                    run.events.put((kind, data), timeout=0.05)
                    run.last_event = time.monotonic()
                    return
                except queue.Full:
                    if time.monotonic() - run.started >= request_timeout:
                        cancel(run, "timed_out", "Inference request timed out")
                    continue
            raise InferenceCancelled("Inference was cancelled")

        def worker():
            fields = {}
            assistant_turn = None
            try:
                result = invoke(on_event, run.cancel_event, on_activity)
                with run.lock:
                    if not run.cancel_event.is_set() and run.kind == "analyze":
                        fields = {
                            "status": "ready", "last_operation_status": "completed", "last_error": None,
                            "report": result["report"], "verification": result["verification"], "evidence": result["evidence"],
                            "context_manifest": result.get("context_manifest", []),
                            "workflow": result.get("workflow", {}),
                            "generation_metadata": result.get("generation_metadata", {}),
                            "raw_response": result.get("raw_response"),
                        }
                    elif not run.cancel_event.is_set():
                        answer = str(result or "").strip()
                        if not answer:
                            raise RuntimeError("The reasoner returned an empty answer")
                        fields = {"last_operation_status": "completed", "last_error": None}
                        assistant_turn = {"role": "assistant", "content": answer, "ts": time.time()}
            except Exception as exc:
                with run.lock:
                    if not run.cancel_event.is_set():
                        run.error = str(exc)
                        run.outcome = "failed"
                        app.state.last_inference_error = run.error
                        fields = {
                            "status": "failed" if run.kind == "analyze" else "ready",
                            "last_operation_status": "failed", "last_error": run.error,
                        }
                        try:
                            app.state.sessions.update_claim(run.session_id, run.token, **fields)
                        except Exception as persistence_error:
                            run.error = f"Unable to persist inference outcome: {persistence_error}"
                            fields["last_error"] = run.error
                            app.state.last_inference_error = run.error
            finally:
                # A backend may stop waiting while native Torch prefill is still
                # executing. Keep both claims until that native call really ends.
                # This wait deliberately holds neither the run lock nor the store lock.
                native_idle = True
                try:
                    wait_until_idle = getattr(app.state.agent.backend, "wait_until_idle", None)
                    if callable(wait_until_idle):
                        wait_until_idle()
                except Exception as idle_error:
                    native_idle = False
                    run.error = f"Unable to confirm inference stopped; restart the server: {idle_error}"
                    run.outcome = "failed"
                    app.state.last_inference_error = run.error
                    app.state.sessions.update_claim(
                        run.session_id, run.token,
                        status="failed" if run.kind == "analyze" else "ready",
                        last_operation_status="failed", last_error=run.error,
                    )
                if native_idle:
                    with run.lock:
                        if run.cancel_event.is_set():
                            fields, assistant_turn = {}, None
                        try:
                            app.state.sessions.finish(
                                run.session_id, run.token, assistant_turn=assistant_turn, **fields,
                            )
                        except Exception as persistence_error:
                            run.error = f"Unable to persist inference outcome: {persistence_error}"
                            run.outcome = "failed"
                            app.state.last_inference_error = run.error
                        else:
                            if not run.cancel_event.is_set() and not run.error:
                                run.outcome = "completed"
                                if run.kind == "analyze":
                                    app.state.last_successful_analysis = time.time()
                                    app.state.last_inference_error = None
                    with app.state.admission_lock:
                        app.state.active_runs.pop(run.token, None)
                        app.state.inference_slots.release()
                run.finished.set()

        try:
            threading.Thread(target=worker, name=f"xcortex-{run.kind}-{run.session_id}", daemon=True).start()
        except Exception as exc:
            app.state.sessions.finish(
                run.session_id, run.token,
                status="failed" if run.kind == "analyze" else "ready",
                last_operation_status="failed", last_error=str(exc),
            )
            with app.state.admission_lock:
                app.state.active_runs.pop(run.token, None)
                app.state.inference_slots.release()
            raise HTTPException(status_code=503, detail="Unable to start inference worker") from exc

        async def generator():
            heartbeat = time.monotonic()
            try:
                yield sse("start", {"session_id": run.session_id})
                while not run.finished.is_set() or not run.events.empty():
                    now = time.monotonic()
                    if not run.finished.is_set() and (
                        now - run.started >= request_timeout or now - run.last_event >= idle_timeout
                    ):
                        cancel(run, "timed_out", "Inference request timed out")
                    if run.cancel_event.is_set():
                        break
                    try:
                        kind, data = run.events.get_nowait()
                    except queue.Empty:
                        if run.outcome == "failed":
                            break
                        if now - heartbeat >= 15:
                            yield ": keep-alive\n\n"
                            heartbeat = now
                        await asyncio.sleep(0.025)
                    else:
                        yield sse(kind, data)
                if run.error:
                    yield sse("error", {"message": run.error, "code": run.outcome})
                yield sse("done", {"status": run.outcome})
            finally:
                if not run.finished.is_set():
                    cancel(run, "cancelled", "The client disconnected before inference completed")

        return StreamingResponse(
            generator(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/analyze")
    def analyze(body: AnalyzeRequest):
        run, session = admit(
            body.session_id, "analyze", status="running", history=body.history,
            turns=[], report=None, verification=None, evidence=[], region=body.region,
        )
        study = Study(
            id=body.session_id,
            image_path=session["image_path"],
            history=body.history,
            metadata=session.get("meta", {}),
        )
        run_dir = Path(settings.paths.artifacts) / body.session_id
        return stream(run, lambda on_event, cancel_event, on_activity: app.state.agent.analyze(
            study, run_dir, on_event, cancel_event=cancel_event, on_activity=on_activity,
        ))

    @app.post("/api/ask")
    def ask(body: AskRequest):
        if not body.question.strip():
            raise HTTPException(status_code=400, detail="Question must not be empty")
        run, session = admit(
            body.session_id, "ask",
            turn={"role": "user", "content": body.question, "ts": time.time()},
        )
        return stream(run, lambda on_event, cancel_event, on_activity: app.state.agent.ask(
            session.get("evidence") or [], body.question, on_event,
            history=session.get("history") or "", report=session.get("report"),
            prior_turns=(session.get("turns") or [])[:-1], cancel_event=cancel_event,
        ))

    def shutdown():
        with app.state.admission_lock:
            runs = list(app.state.active_runs.values())
        for run in runs:
            cancel(run, "cancelled", "The server is shutting down")

    @app.get("/api/sessions")
    def sessions():
        return {"sessions": app.state.sessions.listing()}

    @app.get("/api/sessions/{session_id}")
    def session_detail(session_id: str):
        record = app.state.sessions.get(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown session")
        return record

    @app.get("/api/sessions/{session_id}/pdf")
    def session_pdf(session_id: str):
        if not SAFE_NAME.fullmatch(session_id):
            raise HTTPException(status_code=400, detail="invalid session id")
        record = app.state.sessions.get(session_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown session")
        report = record.get("report")
        if not report:
            raise HTTPException(status_code=400, detail="session has no generated clinical report yet")
        artifacts_dir = Path(settings.paths.artifacts) / session_id
        pdf_path = artifacts_dir / "clinical_report.pdf"
        try:
            render_clinical_pdf(
                report=report,
                metadata={**record.get("meta", {}), "study_id": session_id, "history": record.get("history", "")},
                evidence=record.get("evidence", []),
                artifacts_dir=artifacts_dir,
                output_path=pdf_path,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to render clinical PDF: {exc}") from exc
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"xcortex_report_{session_id}.pdf",
        )

    @app.get("/a/{session_id}/{name}")
    def artifact(session_id: str, name: str):
        if not SAFE_NAME.fullmatch(session_id) or not SAFE_NAME.fullmatch(name):
            raise HTTPException(status_code=404, detail="invalid artifact path")
        base_dir = Path(settings.paths.artifacts).resolve()
        target_path = (base_dir / session_id / name).resolve()
        try:
            target_path.relative_to(base_dir)
        except ValueError:
            raise HTTPException(status_code=404, detail="path traversal denied")
        if not target_path.is_file():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(target_path)

    return app
