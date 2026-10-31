from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xcortex", description="X-Cortex agentic radiograph reasoning")
    parser.add_argument("--config", help="configuration YAML (defaults to XCORTEX_CONFIG or configs/default.yaml)")
    sub = parser.add_subparsers(dest="command")

    app_p = sub.add_parser("app", help="desktop application")
    app_p.add_argument("--browser", action="store_true")
    app_p.add_argument("--mock", action="store_true")
    app_p.add_argument("--port", type=int)

    serve_p = sub.add_parser("serve", help="headless API server")
    serve_p.add_argument("--host")
    serve_p.add_argument("--port", type=int)
    serve_p.add_argument("--mock", action="store_true")

    index_p = sub.add_parser("index", help="build the reference vector store")
    index_p.add_argument("--manifest", required=True)
    index_p.add_argument("--root")
    index_p.add_argument("--batch", type=int, default=16)

    predict_p = sub.add_parser("predict", help="single-study analysis")
    predict_p.add_argument("--image", required=True)
    predict_p.add_argument("--history", default="")
    predict_p.add_argument("--region", default="")
    predict_p.add_argument("--out")
    predict_p.add_argument("--mock", action="store_true")

    download_p = sub.add_parser("download", help="fetch official model weights into the project weights folder")
    download_p.add_argument("models", nargs="*", choices=["sam2", "siglip", "medgemma", "detector", "anomalyclip", "all"])

    smoke_p = sub.add_parser("smoke", help="run real CPU model checks; missing models are reported as blocked")
    smoke_p.add_argument("--image", required=True)
    smoke_p.add_argument("--region", default="chest")
    smoke_p.add_argument("--components", nargs="+", choices=["detector", "segmentation", "siglip", "reasoner"], default=["detector", "reasoner"])
    smoke_p.add_argument("--out", default="data/smoke/results.json")
    smoke_p.add_argument("--max-new-tokens", type=int, default=64)
    smoke_p.add_argument("--full-report", action="store_true")

    benchmark_p = sub.add_parser("benchmark", help="prepare ChestAgentBench inputs without running evaluation")
    benchmark_sub = benchmark_p.add_subparsers(dest="benchmark_command", required=True)
    fetch_p = benchmark_sub.add_parser("fetch", help="fetch pinned official ChestAgentBench assets")
    fetch_p.add_argument("--destination", default="data/benchmarks/chestagentbench")
    fetch_p.add_argument("--include-figures", action="store_true")
    prep_p = benchmark_sub.add_parser("prepare", help="create case-disjoint preparation manifests")
    prep_p.add_argument("--metadata", required=True)
    prep_p.add_argument("--output", required=True)
    prep_p.add_argument("--figures-root")
    prep_p.add_argument("--development-cases", type=int, default=5)
    prep_p.add_argument("--reference-cases", type=int, default=5)
    prep_p.add_argument("--seed", type=int, default=20261027)

    run_p = benchmark_sub.add_parser("run", help="execute benchmark evaluation")
    run_p.add_argument("--questions", default="data/benchmarks/chestagentbench/preparations/pilot-v2-seed20261027/development.questions.jsonl")
    run_p.add_argument("--answers", default="data/benchmarks/chestagentbench/preparations/pilot-v2-seed20261027/private_answer_keys/development.answers.jsonl")
    run_p.add_argument("--audit", default="data/benchmarks/chestagentbench/preparations/pilot-v2-seed20261027/image_audit.audited.jsonl")
    run_p.add_argument("--figures-root", default="data/benchmarks/chestagentbench/raw")
    run_p.add_argument("--results", default="data/benchmarks/chestagentbench/results/pilot_dev_results.jsonl")
    run_p.add_argument("--limit", type=int, default=None)
    run_p.add_argument("--case-id", default=None)
    run_p.add_argument("--question-id", default=None)
    run_p.add_argument("--no-pdf", action="store_true")

    pdf_p = sub.add_parser("pdf", help="render clinical report PDF for a session")
    pdf_p.add_argument("session_id", help="session ID to render PDF for")
    pdf_p.add_argument("--out", help="output path for the rendered PDF")

    return parser


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = Settings.load(args.config)

    if args.command == "benchmark":
        from .evaluation import fetch_chestagentbench, prepare_chestagentbench
        if args.benchmark_command == "fetch":
            result = fetch_chestagentbench(args.destination, include_figures=args.include_figures)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.benchmark_command == "prepare":
            result = prepare_chestagentbench(
                args.metadata, args.output, figures_root=args.figures_root,
                development_cases=args.development_cases, reference_cases=args.reference_cases,
                seed=args.seed,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.benchmark_command == "run":
            from .evaluation import run_benchmark
            result = run_benchmark(
                settings,
                questions_path=args.questions,
                answers_path=args.answers,
                audit_path=args.audit,
                figures_root=args.figures_root,
                results_path=args.results,
                limit=args.limit,
                case_id=args.case_id,
                question_id=args.question_id,
                export_pdf=not args.no_pdf,
            )
        return

    if args.command == "pdf":
        from .report.pdf import render_clinical_pdf
        from .server.app import SessionStore
        store = SessionStore(Path(settings.paths.state))
        record = store.get(args.session_id)
        if not record:
            parser.exit(1, f"Session '{args.session_id}' not found\n")
        report = record.get("report")
        if not report:
            parser.exit(1, f"Session '{args.session_id}' has no generated report\n")
        artifacts_dir = Path(settings.paths.artifacts) / args.session_id
        out_path = args.out or str(artifacts_dir / f"xcortex_report_{args.session_id}.pdf")
        render_clinical_pdf(
            report=report,
            metadata={**record.get("meta", {}), "study_id": args.session_id, "history": record.get("history", "")},
            evidence=record.get("evidence", []),
            artifacts_dir=artifacts_dir,
            output_path=out_path,
        )
        print(f"Clinical report PDF rendered: {out_path}")
        return

    if args.command == "smoke":
        from .smoke import run_smoke
        result = run_smoke(settings, args.image, args.components, args.out, region=args.region, max_new_tokens=args.max_new_tokens, full_report=args.full_report)
        print(f"CPU smoke {result['status']}: {Path(args.out).resolve()}")
        if result["status"] != "passed":
            raise SystemExit(2)
        return

    if args.command in (None, "app"):
        if getattr(args, "mock", False):
            settings.app.mode = "mock"
        if getattr(args, "port", None):
            settings.app.port = args.port
        from .ui.launcher import launch

        try:
            launch(settings, browser=getattr(args, "browser", False))
        except ImportError:
            parser.error("desktop shell requires: pip install xcortex[desktop] (or use --browser)")
        return

    if args.command == "serve":
        if args.mock:
            settings.app.mode = "mock"
        host = args.host or settings.app.host
        port = args.port or settings.app.port
        import uvicorn

        from .server.app import create_app

        uvicorn.run(create_app(settings), host=host, port=port)
        return

    if args.command == "index":
        from .rag.indexer import run_indexing

        run_indexing(settings, args.manifest, root=args.root, batch=args.batch)
        return

    if args.command == "download":
        from .downloads import run_downloads

        which = args.models or ["all"]
        if "all" in which:
            which = ["detector", "anomalyclip", "sam2", "siglip", "medgemma"]
        try:
            run_downloads(settings, which)
        except (RuntimeError, ValueError, OSError) as exc:
            parser.exit(1, f"Model setup incomplete: {exc}\n")
        return

    if args.command == "predict":
        if args.mock:
            settings.app.mode = "mock"
        settings.ensure_dirs()
        from .core.agent import Agent
        from .core.messages import Study

        agent = Agent(settings)
        region = args.region or settings.models.default_region
        study = Study(id="cli", image_path=args.image, history=args.history, metadata={"region": region} if region else {})
        run_dir = Path(settings.paths.artifacts) / "cli"
        result = agent.analyze(study, run_dir, lambda kind, data: None)
        output = json.dumps(result["report"], ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(output, encoding="utf-8")
            print(f"report written to {args.out}")
        else:
            print(output)
        return


if __name__ == "__main__":
    main()
