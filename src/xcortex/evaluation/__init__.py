"""Reproducible benchmark preparation; labels stay outside model inputs."""

from .chestagentbench import (
    CHESTAGENTBENCH_REVISION,
    ChestAgentQuestion,
    build_question_content,
    fetch_chestagentbench,
    prepare_chestagentbench,
    score_choice,
    score_report_choice,
)
from .runner import run_benchmark

__all__ = [
    "CHESTAGENTBENCH_REVISION", "ChestAgentQuestion", "build_question_content",
    "fetch_chestagentbench", "prepare_chestagentbench", "score_choice", "score_report_choice",
    "run_benchmark",
]
