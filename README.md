# X-Cortex

**Agentic AI for evidence-augmented radiograph reasoning.**

[Interactive demo](https://x-cortex-demo.chxx8878.chatgpt.site) · [Repository](https://github.com/Ali-Nadhir/X-Cortex)

X-Cortex is a research framework that brings image retrieval, auxiliary visual evidence, and multimodal reasoning into a single coordinated workflow. Its central research question is how reference cases and localized visual cues can support an agent's interpretation of a radiograph.

## Research approach

The agent follows a fixed LangGraph workflow: collect evidence, assemble the multimodal context, reason once, and format the result. Evidence collection precedes the final reasoning step, making the information supplied to the model explicit and reproducible.

1. **Retrieve reference cases.** MedSigLIP encodes the whole query image and retrieves up to five eligible reference images.
2. **Collect visual evidence.** AnomalyCLIP produces an image-level score and a heatmap overlay to highlight candidate regions.
3. **Assemble context.** The original radiograph, overlay, reference images, clinical context, and evidence ledger form one multimodal input.
4. **Reason and report.** MedGemma 1.5 produces a structured interpretation in a single generation. The reasoner can assess the supporting evidence independently.

The workflow preserves the original image as the primary source. Retrieved cases provide comparison context, while auxiliary visual cues guide attention. User-initiated follow-up questions continue the case conversation.

## Architecture

~~~text
Radiograph + clinical context
             |
        Collect evidence
        /              \
MedSigLIP retrieval   AnomalyCLIP visual cues
        \              /
      Assemble multimodal context
             |
      MedGemma single-pass reasoning
             |
      Structured report + case conversation
~~~

## Engineering components

- **LangGraph orchestration:** explicit stages and a predictable execution sequence.
- **Evidence-aware reporting:** Pydantic schemas, structured decoding, source identifiers, and deterministic metadata checks.
- **Reproducible experiments:** input hashes, case eligibility checks, isolated run artifacts, and model/configuration provenance.
- **Evaluation tooling:** ChestAgentBench preparation and execution support, case-disjoint manifests, and separate tracking of answer completion and correctness.
- **Interactive workspace:** image and context intake, structured findings, follow-up conversation, and saved case history.

The repository includes the Python research implementation and a deployed interface for exploring image review and case conversation. Research evaluation is ongoing.

## Development

Requirements: Python 3.11 is recommended. Model weights and research datasets are obtained separately.

~~~bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m pytest -q
pyright
~~~

For model execution, install the model dependencies and select the appropriate configuration:

~~~bash
python -m pip install -e ".[models,dev]"
xcortex --config configs/chestagentbench.yaml download medgemma siglip anomalyclip
xcortex --config configs/chestagentbench.yaml index --manifest path/to/references.csv
xcortex --config configs/chestagentbench.yaml serve
~~~

The reference manifest defines the eligible image collection. Configure model access, reference paths, hardware, and generation settings for the experiment before running the full workflow.

## Repository structure

| Directory | Purpose |
| --- | --- |
| src/xcortex/core | Agent workflow and reasoning backends |
| src/xcortex/rag | Image encoding, indexing, and retrieval |
| src/xcortex/tools | Auxiliary visual-evidence tools |
| src/xcortex/report | Schemas, prompts, structured decoding, and report formatting |
| src/xcortex/evaluation | Dataset preparation, execution, and provenance |
| deployment/portfolio | Deployed interface and presentation layer |
| configs | Model and experiment configurations |
| tests | Workflow, reporting, retrieval, runtime, and evaluation checks |

## License

[Apache 2.0](LICENSE). Vendored components and bundled fonts retain their accompanying license notices. Model weights and datasets follow their respective licenses.

