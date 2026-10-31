from __future__ import annotations

import abc


class ReasoningBackend(abc.ABC):
    @abc.abstractmethod
    def generate(self, messages: list, on_token=None, task: str = "report", response_schema: dict | None = None) -> str:
        raise NotImplementedError
