from __future__ import annotations

from pathlib import Path
from threading import Event

from .backends import build_backend
from .execution import cancellation_scope, check_cancelled, current_cancel_event
from .messages import Study
from .pipeline import AnalysisPipeline
from .workflow import AnalysisRuntime, AnalysisWorkflow, WorkflowError
from ..report.prompts import (
    ASK_SYSTEM_PROMPT,
    build_ask_prompt,
)
from ..report.verifier import ReportAuditor
from ..tools.registry import build_registry


class AgentError(RuntimeError):
    pass


class Agent:
    def __init__(self, settings):
        self.settings = settings
        self.registry = build_registry(settings)
        self.backend = build_backend(settings)
        self.verifier = ReportAuditor()
        self.pipeline = AnalysisPipeline(self.registry, settings)
        self.workflow = AnalysisWorkflow(self)

    def analyze(
        self, study: Study, run_dir: Path, on_event=lambda kind, data: None,
        cancel_event: Event | None = None,
        on_activity=None,
    ) -> dict:
        with cancellation_scope(cancel_event):
            return self._analyze(study, run_dir, on_event, on_activity)

    def _analyze(self, study: Study, run_dir: Path, on_event, on_activity=None) -> dict:
        check_cancelled()
        context = AnalysisRuntime(
            study=study, run_dir=Path(run_dir), on_event=on_event,
            cancel_event=current_cancel_event(), on_activity=on_activity,
        )
        try:
            return self.workflow.run(context)
        except WorkflowError as exc:
            raise AgentError(str(exc)) from exc

    def ask(
        self,
        session_evidence: list[dict],
        question: str,
        on_event=lambda kind, data: None,
        history: str = "",
        report: dict | None = None,
        prior_turns: list[dict] | None = None,
        cancel_event: Event | None = None,
    ) -> str:
        with cancellation_scope(cancel_event):
            check_cancelled()
            return self._ask(session_evidence, question, on_event, history, report, prior_turns)

    def _ask(self, session_evidence, question, on_event, history, report, prior_turns) -> str:
        prompt_text = build_ask_prompt(
            session_evidence,
            question,
            history=history,
            report=report,
            prior_turns=prior_turns,
        )
        messages = [
            {"role": "system", "content": ASK_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
        ]
        answer = self.backend.generate(
            messages, on_token=lambda token: on_event("token", {"text": token}), task="chat"
        )
        check_cancelled()
        return answer
