from __future__ import annotations

import abc
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


@dataclass
class ToolOutput:
    summary: str
    payload: dict
    images: list[tuple[str, Image.Image | np.ndarray]] = field(default_factory=list)


class BaseTool(abc.ABC):
    name: str = "tool"
    label: str = "Tool"

    def __init__(self, settings):
        self.settings = settings

    @abc.abstractmethod
    def run(self, image: np.ndarray, meta: dict, history: str) -> ToolOutput:
        raise NotImplementedError
