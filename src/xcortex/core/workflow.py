"""Fixed LangGraph workflow: collect, assemble images, reason once, format.

The graph has no conditional edges, retries, checkpoint replay or tool-call loop.
The existing application owns session persistence and native inference admission.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
import re
from threading import Event
from typing import Any, Callable, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .execution import cancellation_scope, check_cancelled
from .messages import EvidenceBundle, Study
from ..report.prompts import MCQ_SYSTEM_PROMPT, SYSTEM_PROMPT, build_user_prompt
from ..report.schema import report_schema_for_options, validate_mcq_options, validate_report_for_options
from ..utils.jsonutil import extract_json


class WorkflowError(RuntimeError):
    pass


class ContextImage(BaseModel):
    """Actual image input with a separate public identity/provenance record."""

    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    role: Literal["query", "auxiliary", "reference"]
    path: str = Field(min_length=1, exclude=True)
    url: str
    source_evidence_id: str | None = None
    reference_case_id: str | None = None
    description: str


class ContextPacket(BaseModel):
    model_config = ConfigDict(extra="forbid")
    images: list[ContextImage] = Field(min_length=1)
    reference_count: int = Field(ge=0)

    def manifest(self) -> list[dict[str, Any]]:
        return [image.model_dump() for image in self.images]


@dataclass(frozen=True)
class AnalysisRuntime:
    study: Study
    run_dir: Path
    on_event: Callable[[str, Any], None]
    cancel_event: Event | None = None
    on_activity: Callable[[], None] | None = None


class AnalysisState(TypedDict, total=False):
    bundle: EvidenceBundle
    packet: ContextPacket
    messages: list[dict[str, Any]]
    raw_text: str
    response_schema: dict[str, Any]
    mcq_options: dict[str, str] | None
    generation_metadata: dict[str, Any]
    result: dict[str, Any]


def assemble_packet(bundle: EvidenceBundle, run_dir: Path) -> ContextPacket:
    """Attach every supplied reference/overlay with unambiguous image IDs."""
    if not bundle.normalized_image_path:
        raise WorkflowError("pipeline did not produce a normalized image for the reasoner")
    images = [ContextImage(
        id="Q0", role="query", path=bundle.normalized_image_path,
        url=f"/a/{bundle.run_id}/normalized.png",
        description="Original current-study radiograph without auxiliary annotations; independently inspect this image.",
    )]
    references = 0
    for item in bundle.items:
        matches = item.payload.get("matches", []) if item.tool in {"rag", "retrieval"} else []
        reference_artifacts = [a for a in item.artifacts if a.kind == "reference"]
        if matches and len(reference_artifacts) != len(matches):
            raise WorkflowError(
                f"retrieval image packet is incomplete: {len(matches)} returned cases but "
                f"{len(reference_artifacts)} attached images"
            )
        for artifact in item.artifacts:
            if artifact.kind not in {"overlay", "reference"}:
                continue
            if artifact.kind == "overlay":
                auxiliary_count = sum(image.role == "auxiliary" for image in images)
                images.append(ContextImage(
                    id=f"Q{auxiliary_count + 1}", role="auxiliary",
                    path=str(run_dir / artifact.name), url=artifact.url,
                    source_evidence_id=item.id,
                    description=f"Annotated copy of Q0 from {item.tool}; fallible auxiliary output, not an independent study or confirmed diagnosis.",
                ))
            else:
                source_index = reference_artifacts.index(artifact)
                index_match = re.fullmatch(r"ref_(\d+)\.[A-Za-z0-9]+", artifact.name)
                if index_match:
                    source_index = int(index_match.group(1))
                match = matches[source_index] if source_index < len(matches) else {}
                references += 1
                images.append(ContextImage(
                    id=f"R{references}", role="reference",
                    path=str(run_dir / artifact.name), url=artifact.url,
                    source_evidence_id=item.id,
                    reference_case_id=str(match.get("case_id", "unknown")),
                    description="Retrieved reference case from another patient. Its diagnosis and measurements do not belong to Q0.",
                ))
    return ContextPacket(images=images, reference_count=references)


class AnalysisWorkflow:
    """Compiled once; callbacks and cancellation are per-invocation context."""

    def __init__(self, agent: Any):
        self.agent = agent
        graph = StateGraph(AnalysisState, context_schema=AnalysisRuntime)
        graph.add_node("collect", self.collect)
        graph.add_node("assemble", self.assemble)
        graph.add_node("reason", self.reason)
        graph.add_node("format", self.format)
        graph.add_edge(START, "collect")
        graph.add_edge("collect", "assemble")
        graph.add_edge("assemble", "reason")
        graph.add_edge("reason", "format")
        graph.add_edge("format", END)
        self.graph = graph.compile()

    def run(self, context: AnalysisRuntime) -> dict[str, Any]:
        # Clinical images, reference text and prompts stay local. This application
        # does not implicitly adopt machine-wide LangSmith tracing environment
        # variables; a future remote-observability feature needs explicit design.
        with tracing_context(enabled=False):
            state = self.graph.invoke({}, context=context)
        return state["result"]

    def collect(self, state: AnalysisState, runtime: Runtime[AnalysisRuntime]) -> AnalysisState:
        context = runtime.context
        # LangGraph may run nodes on another thread; bind the existing thread-local
        # cooperative cancellation scope explicitly in every node.
        with cancellation_scope(context.cancel_event):
            try:
                validate_mcq_options(context.study.metadata.get("options"))
            except ValueError as exc:
                raise WorkflowError(f"Invalid MCQ task options: {exc}") from exc
            if getattr(self.agent.settings.models, "cpu_release_auxiliary_models", False):
                release = getattr(self.agent.backend, "release_model", None)
                if release is not None:
                    release()
            bundle = self.agent.pipeline.run(context.study, context.run_dir, context.on_event)
            check_cancelled()
            return {"bundle": bundle}

    def assemble(self, state: AnalysisState, runtime: Runtime[AnalysisRuntime]) -> AnalysisState:
        context = runtime.context
        with cancellation_scope(context.cancel_event):
            bundle = state.get("bundle")
            if bundle is None:
                raise WorkflowError("the collect node did not supply an evidence bundle")
            packet = assemble_packet(bundle, context.run_dir)
            options = validate_mcq_options(context.study.metadata.get("options"))
            schema = report_schema_for_options(options)
            content = []
            for image in packet.images:
                content.extend([
                    {"type": "text", "text": f"{image.id}: {image.description} Source: {image.source_evidence_id or 'query image'}. Reference case: {image.reference_case_id or 'not a reference'}."},
                    {"type": "image", "path": image.path},
                ])
            content.append({"type": "text", "text": build_user_prompt(context.study, bundle, packet.manifest(), schema)})
            return {"packet": packet, "response_schema": schema, "mcq_options": options, "messages": [
                {"role": "system", "content": MCQ_SYSTEM_PROMPT if options is not None else SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]}

    def reason(self, state: AnalysisState, runtime: Runtime[AnalysisRuntime]) -> AnalysisState:
        context = runtime.context
        with cancellation_scope(context.cancel_event):
            def draft_activity(_token):
                check_cancelled()
                if context.on_activity is not None:
                    context.on_activity()

            messages = state.get("messages")
            if not messages:
                raise WorkflowError("the assemble node did not supply a multimodal input")
            kwargs: dict[str, Any] = {"on_token": draft_activity, "task": "report"}
            if state.get("mcq_options") is not None:
                schema = state.get("response_schema")
                if schema is None:
                    raise WorkflowError("the MCQ request is missing its required response schema")
                kwargs["response_schema"] = schema
            raw_text = self.agent.backend.generate(messages, **kwargs)
            check_cancelled()
            context.run_dir.mkdir(parents=True, exist_ok=True)
            (context.run_dir / "reasoner-response.txt").write_text(raw_text, encoding="utf-8")
            return {
                "raw_text": raw_text,
                "generation_metadata": deepcopy(getattr(self.agent.backend, "last_run", {})),
            }

    def format(self, state: AnalysisState, runtime: Runtime[AnalysisRuntime]) -> AnalysisState:
        context = runtime.context
        with cancellation_scope(context.cancel_event):
            raw_text = state.get("raw_text")
            if raw_text is None:
                raise WorkflowError("the reason node did not supply its single response")
            data = extract_json(raw_text)
            if data is None:
                raise WorkflowError("reasoner returned invalid JSON after its single permitted generation; no automatic retry was made")
            try:
                report = validate_report_for_options(data, state.get("mcq_options"))
            except (ValidationError, ValueError) as exc:
                raise WorkflowError("reasoner output failed the report schema after its single permitted generation; no automatic retry was made") from exc
            bundle, packet = state.get("bundle"), state.get("packet")
            if bundle is None or packet is None:
                raise WorkflowError("the collected evidence or image manifest is missing")
            verification = self.agent.verifier.verify(report, bundle, packet)
            result = {
                "report": report.model_dump(),
                "verification": verification.model_dump(),
                "evidence": [{
                    "id": item.id, "tool": item.tool, "summary": item.summary,
                    "payload": item.payload,
                    "artifacts": [{"name": a.name, "url": a.url, "kind": a.kind} for a in item.artifacts],
                } for item in bundle.items],
                "context_manifest": packet.manifest(),
                "workflow": {"engine": "langgraph", "nodes": ["collect", "assemble", "reason", "format"], "generation_count": 1},
                "raw_response": raw_text,
                "generation_metadata": state.get("generation_metadata", {}),
                "response_schema": state.get("response_schema"),
            }
            check_cancelled()
            context.on_event("token", {"text": json.dumps(result["report"], ensure_ascii=False)})
            check_cancelled()
            context.on_event("report", result)
            return {"result": result}
