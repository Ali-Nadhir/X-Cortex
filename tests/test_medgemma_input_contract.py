"""The actual local processor must preserve image/question/task boundaries."""
from pathlib import Path

import pytest

from xcortex.config import Settings
from xcortex.core.backends.medgemma import MedGemmaBackend


def test_local_processor_has_one_bos_and_retains_question_and_image(study_png):
    from transformers import AutoProcessor

    settings = Settings()
    model_path = settings.model_path(settings.models.medgemma_id)
    if not (model_path / "processor_config.json").is_file():
        pytest.skip("Local MedGemma processor not installed; no checkpoint download in tests")
    backend = MedGemmaBackend(settings)
    backend._processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    question = "Which option applies? A) first supplied choice B) second supplied choice"
    inputs = backend._build_inputs([
        {"role": "system", "content": "Interpret the supplied image."},
        {"role": "user", "content": [
            {"type": "image", "path": str(study_png)}, {"type": "text", "text": question},
        ]},
    ])
    tokenizer = backend._processor.tokenizer
    ids = inputs["input_ids"][0].tolist()
    assert ids[0] == tokenizer.bos_token_id
    assert ids[1] != tokenizer.bos_token_id
    assert question in tokenizer.decode(ids)
    assert inputs["pixel_values"].shape[0] == 1
    assert backend._input_trace["image_sources"][0]["path"] == str(Path(study_png).resolve())


def test_int8_profile_is_explicit_and_does_not_change_default_precision():
    baseline = Settings.load("configs/chestagentbench.yaml")
    optimized = Settings.load("configs/cpu-int8.yaml")
    assert baseline.models.cpu_quantization == "none"
    assert baseline.models.dtype == "bfloat16"
    assert optimized.models.cpu_quantization == "dynamic_int8"
    assert optimized.models.cpu_release_auxiliary_models
    assert optimized.rag.top_k == baseline.rag.top_k == 5
