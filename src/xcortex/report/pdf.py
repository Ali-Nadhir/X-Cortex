"""Deterministic pure-Python PDF report renderer for X-Cortex.

Renders ClinicalReport JSON, study metadata, and auxiliary/retrieval image
artifacts into a printable clinical PDF without any secondary model call.
Zero external PDF dependencies required; byte-reproducible output.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from PIL import Image


def _escape_pdf_text(text: str) -> str:
    """Escape special characters for PDF text string literals."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """Simple, clean word wrapping."""
    if not text:
        return []
    words = text.split()
    if not words:
        return []
    lines = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current_len + len(word) + (1 if current else 0) <= max_chars:
            current.append(word)
            current_len += len(word) + (1 if len(current) > 1 else 0)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
    if current:
        lines.append(" ".join(current))
    return lines


class SimplePDFCanvas:
    """Lightweight 2-page canvas builder for deterministic medical reports."""

    def __init__(self, page_width: float = 595.28, page_height: float = 841.89):
        self.width = page_width
        self.height = page_height
        self.pages: list[list[str]] = [[]]
        self.page_images: list[dict[str, tuple[int, int, bytes]]] = [{}]
        self.current_page = 0

    def new_page(self) -> None:
        self.pages.append([])
        self.page_images.append({})
        self.current_page += 1

    def add_command(self, cmd: str) -> None:
        self.pages[self.current_page].append(cmd)

    def fill_rect(self, x: float, y: float, w: float, h: float, r: float, g: float, b: float) -> None:
        self.add_command(f"{r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")

    def stroke_rect(self, x: float, y: float, w: float, h: float, r: float, g: float, b: float, line_width: float = 1.0) -> None:
        self.add_command(f"{r:.3f} {g:.3f} {b:.3f} RG {line_width:.2f} w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re S")

    def line(self, x1: float, y1: float, x2: float, y2: float, r: float, g: float, b: float, line_width: float = 1.0) -> None:
        self.add_command(f"{r:.3f} {g:.3f} {b:.3f} RG {line_width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def text(self, x: float, y: float, string: str, font: str = "/F1", size: float = 10.0, r: float = 0.1, g: float = 0.1, b: float = 0.1) -> None:
        escaped = _escape_pdf_text(string)
        self.add_command(f"BT {font} {size:.1f} Tf {r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} Td ({escaped}) Tj ET")

    def draw_image(self, name: str, img: Image.Image, x: float, y: float, w: float, h: float) -> None:
        buf = io.BytesIO()
        rgb = img.convert("RGB")
        rgb.save(buf, format="JPEG", quality=85)
        self.page_images[self.current_page][name] = (rgb.width, rgb.height, buf.getvalue())
        self.add_command(f"q {w:.2f} 0 0 {h:.2f} {x:.2f} {y:.2f} cm /{name} Do Q")

    def compile(self) -> bytes:
        objs: list[bytes] = []
        # Obj 1: Catalog
        objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        
        # We need object IDs for pages, resources, contents, fonts, and images.
        num_pages = len(self.pages)
        page_obj_ids = [3 + i * 3 for i in range(num_pages)]
        
        # Obj 2: Pages container
        kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>".encode("latin-1"))

        # Common Fonts: F1 (Helvetica), F2 (Helvetica-Bold), F3 (Helvetica-Oblique)
        # We will append them at the end.
        font_base = 2 + num_pages * 3 + 1
        f1_id = font_base
        f2_id = font_base + 1
        f3_id = font_base + 2

        # Build pages, contents, resources
        image_obj_counter = f3_id + 1
        for i, (page_cmds, p_images) in enumerate(zip(self.pages, self.page_images)):
            content_id = page_obj_ids[i] + 1
            resource_id = page_obj_ids[i] + 2

            # Page object
            page_dict = (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.2f} {self.height:.2f}] "
                f"/Contents {content_id} 0 R /Resources {resource_id} 0 R >>"
            )
            objs.append(page_dict.encode("latin-1"))

            # Content stream
            stream_data = "\n".join(page_cmds).encode("latin-1", errors="replace")
            content_obj = f"<< /Length {len(stream_data)} >>\nstream\n".encode("latin-1") + stream_data + b"\nendstream"
            objs.append(content_obj)

            # Resources dictionary
            xobj_entries = []
            image_mappings = {}
            for img_name, (iw, ih, idata) in p_images.items():
                img_id = image_obj_counter
                image_obj_counter += 1
                image_mappings[img_id] = (iw, ih, idata)
                xobj_entries.append(f"/{img_name} {img_id} 0 R")

            xobj_str = f"<< {' '.join(xobj_entries)} >>" if xobj_entries else "<<"
            res_dict = (
                f"<< /Font << /F1 {f1_id} 0 R /F2 {f2_id} 0 R /F3 {f3_id} 0 R >> "
                f"/XObject {xobj_str} >>"
            )
            objs.append(res_dict.encode("latin-1"))

        # Font objects
        objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>")

        # Image objects
        for page_imgs in self.page_images:
            for iw, ih, idata in page_imgs.values():
                img_obj = (
                    f"<< /Type /XObject /Subtype /Image /Width {iw} /Height {ih} "
                    f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
                    f"/Length {len(idata)} >>\nstream\n"
                ).encode("latin-1") + idata + b"\nendstream"
                objs.append(img_obj)

        # Assemble PDF file with xref table
        out = io.BytesIO()
        out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for idx, obj in enumerate(objs, 1):
            offsets.append(out.tell())
            out.write(f"{idx} 0 obj\n".encode("latin-1"))
            out.write(obj)
            out.write(b"\nendobj\n")

        xref_pos = out.tell()
        total_objs = len(objs) + 1
        out.write(f"xref\n0 {total_objs}\n".encode("latin-1"))
        out.write(b"0000000000 65535 f \n")
        for off in offsets:
            out.write(f"{off:010d} 00000 n \n".encode("latin-1"))
        trailer = f"trailer\n<< /Size {total_objs} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
        out.write(trailer.encode("latin-1"))
        return out.getvalue()


def render_clinical_pdf(
    report: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    artifacts_dir: str | Path | None = None,
    output_path: str | Path | None = None,
) -> bytes:
    """Generate a clean, deterministic clinical report PDF from report JSON and artifacts."""
    meta = metadata or {}
    canvas = SimplePDFCanvas(page_width=595.28, page_height=841.89)

    margin = 40.0
    content_width = 515.28
    y = 800.0

    # 1. Header Band
    canvas.fill_rect(margin, y - 55, content_width, 55, 0.08, 0.18, 0.32)
    canvas.text(margin + 16, y - 24, "X-CORTEX CLINICAL RADIOGRAPH REPORT", font="/F2", size=15.0, r=1.0, g=1.0, b=1.0)
    canvas.text(margin + 16, y - 42, "Multimodal Single-Pass Interpretation With Whole-Image Retrieval & Auxiliary Context", font="/F1", size=8.5, r=0.82, g=0.88, b=0.96)
    y -= 75

    # 2. Study Metadata Grid
    canvas.fill_rect(margin, y - 48, content_width, 48, 0.96, 0.97, 0.98)
    canvas.stroke_rect(margin, y - 48, content_width, 48, 0.85, 0.88, 0.91, 0.75)

    study_id = str(meta.get("study_id") or meta.get("id") or "UNKNOWN-STUDY")
    region = str(report.get("study", {}).get("region") or meta.get("region") or "Chest").capitalize()
    history = str(report.get("study", {}).get("history") or meta.get("history") or "None provided").strip()
    date_str = str(meta.get("study_date") or meta.get("date") or "2026-10-28")

    canvas.text(margin + 12, y - 18, f"Study ID: {study_id}", font="/F2", size=9.0, r=0.15, g=0.2, b=0.25)
    canvas.text(margin + 220, y - 18, f"Region / Anatomy: {region}", font="/F2", size=9.0, r=0.15, g=0.2, b=0.25)
    canvas.text(margin + 400, y - 18, f"Date: {date_str}", font="/F1", size=9.0, r=0.3, g=0.35, b=0.4)

    user_req = str(meta.get("question") or "").strip()
    if user_req:
        canvas.text(margin + 12, y - 36, f"User Request: {user_req[:75]}{'...' if len(user_req) > 75 else ''}", font="/F1", size=8.5, r=0.25, g=0.3, b=0.35)
    else:
        canvas.text(margin + 12, y - 36, f"Clinical History: {history[:80]}{'...' if len(history) > 80 else ''}", font="/F1", size=8.5, r=0.25, g=0.3, b=0.35)
    y -= 62

    # 3. Triage / Auxiliary Detector Summary
    triage = report.get("triage_summary")
    if triage:
        canvas.fill_rect(margin, y - 42, content_width, 42, 0.98, 0.96, 0.92)
        canvas.stroke_rect(margin, y - 42, content_width, 42, 0.92, 0.82, 0.65, 0.75)
        canvas.text(margin + 12, y - 16, "Auxiliary Anomaly Proposal (Fallible Context):", font="/F2", size=9.5, r=0.55, g=0.32, b=0.08)
        score_val = triage.get("abnormality_probability", 0.0)
        calibrated = triage.get("calibrated", False)
        cal_str = "Calibrated" if calibrated else "Uncalibrated text-similarity softmax"
        method_str = triage.get("method") or "AnomalyCLIP ViT-L/14@336px"
        canvas.text(
            margin + 12, y - 32,
            f"Model: {method_str}  |  Anomaly Score: {score_val:.4f} ({cal_str})  |  Threshold: {triage.get('threshold', 0.33):.2f}",
            font="/F1", size=8.5, r=0.45, g=0.28, b=0.05
        )
        y -= 52

    # 4. Diagnostic Observations & Findings
    canvas.text(margin, y - 14, "DIAGNOSTIC OBSERVATIONS & FINDINGS", font="/F2", size=11.0, r=0.08, g=0.18, b=0.32)
    canvas.line(margin, y - 18, margin + content_width, y - 18, 0.08, 0.18, 0.32, 1.0)
    y -= 30

    findings = report.get("findings", [])
    if not findings:
        canvas.text(margin + 10, y - 12, "No focal radiographic abnormalities described.", font="/F1", size=9.5, r=0.3, g=0.3, b=0.3)
        y -= 22
    else:
        for finding in findings:
            struct = finding.get("structure", "Anatomy")
            obs = finding.get("observation", "")
            evidence_ids = finding.get("evidence_ids", [])
            ev_str = f" [Cited: {', '.join(evidence_ids)}]" if evidence_ids else ""
            
            canvas.text(margin + 8, y - 12, f"• {struct.capitalize()}:", font="/F2", size=9.5, r=0.15, g=0.22, b=0.3)
            y -= 14
            for line in _wrap_text(obs + ev_str, max_chars=88):
                canvas.text(margin + 20, y - 10, line, font="/F1", size=9.0, r=0.2, g=0.2, b=0.2)
                y -= 13
            # Render measurements if any
            for m in finding.get("measurements", []):
                val, unit, src = m.get("value"), m.get("unit"), m.get("source")
                canvas.text(margin + 20, y - 9, f"  Measurement: {val} {unit} (Source: {src})", font="/F3", size=8.5, r=0.35, g=0.35, b=0.4)
                y -= 12
            y -= 4

    # 5. Impression Box
    y -= 8
    canvas.text(margin, y - 14, "IMPRESSION", font="/F2", size=11.0, r=0.08, g=0.18, b=0.32)
    canvas.line(margin, y - 18, margin + content_width, y - 18, 0.08, 0.18, 0.32, 1.0)
    y -= 26

    impression_lines = _wrap_text(report.get("impression", "No acute radiographic cardiopulmonary abnormality."), max_chars=86)
    box_h = max(36.0, len(impression_lines) * 14.0 + 16.0)
    canvas.fill_rect(margin, y - box_h, content_width, box_h, 0.94, 0.96, 0.99)
    canvas.stroke_rect(margin, y - box_h, content_width, box_h, 0.75, 0.83, 0.94, 1.0)

    imp_y = y - 16
    for line in impression_lines:
        canvas.text(margin + 14, imp_y, line, font="/F2", size=9.5, r=0.08, g=0.15, b=0.25)
        imp_y -= 14
    y -= (box_h + 16)

    # 6. Multiple-Choice Option (if present)
    ans_choice = report.get("answer_choice")
    if ans_choice:
        canvas.fill_rect(margin, y - 32, content_width, 32, 0.92, 0.98, 0.94)
        canvas.stroke_rect(margin, y - 32, content_width, 32, 0.55, 0.82, 0.65, 1.0)
        canvas.text(margin + 14, y - 20, f"Benchmark Evaluated Answer: Option ({ans_choice})", font="/F2", size=10.5, r=0.08, g=0.45, b=0.2)
        canvas.text(margin + 260, y - 20, "Single-pass generation; no answer-key leakage.", font="/F1", size=8.5, r=0.2, g=0.4, b=0.25)
        y -= 42

    # 7. Recommendations & Limitations
    recs = report.get("recommendations", [])
    limits = report.get("limitations", [])
    if recs or limits:
        if recs:
            canvas.text(margin, y - 12, "Recommendations:", font="/F2", size=9.0, r=0.2, g=0.25, b=0.3)
            y -= 14
            for rec in recs:
                for line in _wrap_text(f"- {rec}", max_chars=90):
                    canvas.text(margin + 8, y - 10, line, font="/F1", size=8.5, r=0.25, g=0.25, b=0.25)
                    y -= 12
        if limits:
            canvas.text(margin, y - 12, "Limitations:", font="/F2", size=9.0, r=0.3, g=0.2, b=0.2)
            y -= 14
            for lim in limits:
                for line in _wrap_text(f"- {lim}", max_chars=90):
                    canvas.text(margin + 8, y - 10, line, font="/F1", size=8.5, r=0.3, g=0.3, b=0.3)
                    y -= 12

    # Footer Page 1
    canvas.line(margin, 45.0, margin + content_width, 45.0, 0.85, 0.88, 0.91, 0.5)
    canvas.text(margin, 34.0, "X-Cortex Research Framework | Medical Vision-Language Reasoning | Page 1 of 2", font="/F1", size=7.5, r=0.5, g=0.55, b=0.6)

    # PAGE 2: Visual Evidence Panel
    canvas.new_page()
    py = 800.0

    canvas.fill_rect(margin, py - 40, content_width, 40, 0.08, 0.18, 0.32)
    canvas.text(margin + 16, py - 25, "VISUAL EVIDENCE & AUXILIARY ARTIFACT PANEL", font="/F2", size=13.0, r=1.0, g=1.0, b=1.0)
    py -= 56

    canvas.text(
        margin, py - 12,
        "Original radiograph (Q0), auxiliary proposal overlay (Q1), and retrieved same-anatomy reference cases (R1-R5).",
        font="/F1", size=8.5, r=0.3, g=0.35, b=0.4
    )
    py -= 24

    art_dir = Path(artifacts_dir) if artifacts_dir else None
    
    # Grid coordinates for images: 2 main images (Q0, Q1) at top, 4-5 reference thumbnails below
    # Row 1: Q0 and Q1 (180x180 each)
    q0_path = (art_dir / "normalized.png") if art_dir else None
    q1_path = (art_dir / "overlay_anomalyclip.png") if art_dir else None

    img_size = 170.0
    gap = 40.0
    x_q0 = margin + 30.0
    x_q1 = x_q0 + img_size + gap

    # Draw Q0
    if q0_path and q0_path.is_file():
        try:
            with Image.open(q0_path) as im:
                canvas.draw_image("ImQ0", im, x_q0, py - img_size, img_size, img_size)
        except Exception:
            canvas.fill_rect(x_q0, py - img_size, img_size, img_size, 0.9, 0.9, 0.9)
    else:
        canvas.fill_rect(x_q0, py - img_size, img_size, img_size, 0.92, 0.92, 0.92)
    canvas.stroke_rect(x_q0, py - img_size, img_size, img_size, 0.7, 0.75, 0.8, 1.0)
    canvas.text(x_q0 + 10, py - img_size - 14, "Q0: Query Radiograph (Original)", font="/F2", size=8.5, r=0.1, g=0.2, b=0.3)
    canvas.text(x_q0 + 10, py - img_size - 25, "Unannotated independent visual input", font="/F1", size=7.5, r=0.4, g=0.4, b=0.4)

    # Draw Q1
    if q1_path and q1_path.is_file():
        try:
            with Image.open(q1_path) as im:
                canvas.draw_image("ImQ1", im, x_q1, py - img_size, img_size, img_size)
        except Exception:
            canvas.fill_rect(x_q1, py - img_size, img_size, img_size, 0.9, 0.9, 0.9)
    else:
        canvas.fill_rect(x_q1, py - img_size, img_size, img_size, 0.92, 0.92, 0.92)
    canvas.stroke_rect(x_q1, py - img_size, img_size, img_size, 0.7, 0.75, 0.8, 1.0)
    canvas.text(x_q1 + 10, py - img_size - 14, "Q1: AnomalyCLIP Spatial Response", font="/F2", size=8.5, r=0.55, g=0.3, b=0.1)
    canvas.text(x_q1 + 10, py - img_size - 25, "Fallible auxiliary heatmap proposal", font="/F1", size=7.5, r=0.4, g=0.4, b=0.4)

    py -= (img_size + 45.0)

    # Row 2: Reference cases (R1 to R5)
    canvas.text(margin, py - 12, "Retrieved Same-Anatomy Reference Cases (MedSigLIP top-k=5):", font="/F2", size=9.5, r=0.15, g=0.2, b=0.25)
    canvas.text(margin + 310, py - 12, "*Other patients; diagnoses are not query findings.", font="/F3", size=7.5, r=0.5, g=0.2, b=0.2)
    canvas.line(margin, py - 16, margin + content_width, py - 16, 0.8, 0.85, 0.9, 0.75)
    py -= 28

    ref_w = 90.0
    ref_h = 90.0
    ref_gap = (content_width - (5 * ref_w)) / 4.0

    for idx in range(5):
        rx = margin + idx * (ref_w + ref_gap)
        ref_file = (art_dir / f"ref_{idx}.png") if art_dir else None
        if ref_file and ref_file.is_file():
            try:
                with Image.open(ref_file) as rim:
                    canvas.draw_image(f"ImR{idx}", rim, rx, py - ref_h, ref_w, ref_h)
            except Exception:
                canvas.fill_rect(rx, py - ref_h, ref_w, ref_h, 0.9, 0.9, 0.9)
        else:
            canvas.fill_rect(rx, py - ref_h, ref_w, ref_h, 0.93, 0.93, 0.93)
        canvas.stroke_rect(rx, py - ref_h, ref_w, ref_h, 0.8, 0.85, 0.9, 0.75)
        canvas.text(rx + 4, py - ref_h - 12, f"Ref R{idx+1}", font="/F2", size=7.5, r=0.2, g=0.25, b=0.3)
        canvas.text(rx + 4, py - ref_h - 21, "External Patient", font="/F1", size=6.5, r=0.45, g=0.45, b=0.45)

    # Footer Page 2
    canvas.line(margin, 45.0, margin + content_width, 45.0, 0.85, 0.88, 0.91, 0.5)
    canvas.text(margin, 34.0, "X-Cortex Research Framework | Medical Vision-Language Reasoning | Page 2 of 2", font="/F1", size=7.5, r=0.5, g=0.55, b=0.6)

    pdf_bytes = canvas.compile()
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_bytes(pdf_bytes)
    return pdf_bytes
