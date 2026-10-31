from xcortex.core.agent import Agent
from xcortex.core.messages import Study


def test_mock_analysis_streams_grounded_report(settings, study_png, tmp_path):
    agent = Agent(settings)
    events = []
    study = Study(id="t1", image_path=str(study_png), history="72yo F, fall on wrist")
    result = agent.analyze(study, tmp_path / "artifacts" / "t1", lambda kind, data: events.append((kind, data)))

    kinds = [k for k, _ in events]
    assert kinds.count("stage") >= 6
    assert "token" in kinds
    assert kinds[-1] == "report"

    assert result["report"]["triage_summary"]["abnormality_probability"] == 0.71
    assert result["verification"]["ok"] is True
    assert {e["id"] for e in result["evidence"]} == {"TRI-1", "SEG-1", "RAG-1"}

    overlay = (tmp_path / "artifacts" / "t1" / "overlay.png").is_file()
    refs = all((tmp_path / "artifacts" / "t1" / f"ref_{i}.png").is_file() for i in range(3))
    assert overlay and refs


def test_ask_uses_chat_task(settings):
    agent = Agent(settings)
    tokens = []
    answer = agent.ask([{"id": "TRI-1", "payload": {"abnormality_probability": 0.71}}], "summarize", lambda k, d: tokens.append(d))
    assert "0.71" in answer
    assert tokens
