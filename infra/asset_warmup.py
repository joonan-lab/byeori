"""Prove the image at build time: convert a PDF, with no network left to fall back on.

Downloading a model on first use would mean the first paper of the day pays for it and a
HuggingFace outage stops ingestion. So the build converts a real document, which pulls exactly
the weights the runtime path uses and leaves them in the image.

It also fails the build for the failure this image exists to avoid. marker's default layout and
recognition models are served through llama.cpp on a machine without an NVIDIA GPU, and a
Fargate task has none; ``disable_ocr`` is what keeps the converter on the small rf-detr detector.
If that ever stops being true, the conversion below raises here instead of at three in the
morning against a paper somebody is waiting for.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from fpdf import FPDF

CAPTION = ("Figure 1. Odds ratio for the association between the variant and the trait, "
           "shown by ancestry group: a European, b East Asian, c African.")


def sample_pdf(path: Path) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=11)
    pdf.multi_cell(0, 6, "Methods\n\nWe genotyped 4,102 participants across three cohorts and "
                         "tested each variant for association with the trait (Figure 1).")
    pdf.ln(60)
    pdf.multi_cell(0, 6, CAPTION)
    pdf.output(str(path))


def main() -> int:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    with tempfile.TemporaryDirectory() as workdir:
        pdf = Path(workdir) / "warmup.pdf"
        sample_pdf(pdf)
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
            renderer="marker.renderers.json.JSONRenderer",
            config={"pdftext_workers": 1, "disable_tqdm": True, "disable_ocr": True},
        )
        rendered = converter(str(pdf))

    pages = rendered.children or []
    if not pages:
        print("warmup: the converter returned no pages", file=sys.stderr)
        return 1
    print(f"warmup: converted {len(pages)} page(s) with no VLM and no OCR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
