"""Fargate asset worker: PDF in S3 -> figure and table crops with their captions and mentions.

The user extracted 11,554 papers this way on their Mac in September 2026, in two stages: marker
turned each PDF into a layout JSON, then a local script cropped the figures and tables and wrote
the caption, the panel letters, the sentences in the paper that mention each one and the assets
cited alongside it. Everything a later answer actually used came from that text, not the images:
measured over the whole set, the PNGs are 108.98 GB and the text 0.26 GB.

This is the same work, beside the GROBID extraction it belongs with, so a paper that arrives from
now on gets its figures without anybody's laptop. Per paper on the user's Mac with eight
processes and no GPU: marker 1.23 s, cropping about 1.6 s. Expect several times that on Fargate's
x86 CPUs, which still leaves a day's intake at minutes.

Input: ``JOB_KEY``, a JSON list of stems, exactly like ``extract_worker.py``.
Output, for each stem with figures::

    papers/{stem}/assets/assets.md      caption, panels, mentions, co-citations, page and bbox
    papers/{stem}/assets/manifest.json  the machine-readable list, with images_stored
    papers/{stem}/assets/{key}.png      one crop per figure or table

``original.pdf``, ``clean.md``, ``grobid.tei.xml`` and ``meta.json`` are never touched, and the
wiki is never written. A paper whose ``assets.md`` is already there is skipped unless ``FORCE=1``,
so a rerun after an interruption costs nothing.

The asset text is uploaded before the images, and ``manifest.json`` records ``images_stored``, so
a reader can tell an absent PNG from a figure that has none.
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import boto3

BUCKET = os.environ["BUCKET_NAME"]
JOB_KEY = os.environ.get("JOB_KEY") or ""
# One paper as it arrives needs no manifest in S3: the trigger passes it here.
STEMS = [part for part in os.environ.get("STEMS", "").split(",") if part.strip()]
TABLE = os.environ.get("TABLE_NAME") or ""
FORCE = os.environ.get("FORCE") == "1"
DPI = int(os.environ.get("ASSET_DPI", "350"))
SCALE = DPI / 72.0
PAD = 6
MAX_PIXELS = 80_000_000          # a very large page would otherwise render into swap
MAX_MENTIONS = 12
MAX_RELATED = 10
MAX_TABLE_CHARS = 4000
EXTRACTOR = os.environ.get("ASSET_EXTRACTOR", "marker-pdf")

FIG_KINDS = ("Figure", "Picture", "Diagram", "FigureGroup", "PictureGroup")
TEXT_KINDS = ("Text", "TextInlineMath", "SectionHeader", "ListItem")
LETTERS = "abcdefghijkl"
PANEL_PUNCT = re.compile(r"(?:^|[\(\s\.,;])([a-lA-L])\s*[\)\,\.]")
# Either case: "a Amyloid burden, b Tau burden" and "A Top: representative traces, B Summary"
# are the same convention, and reading only the first loses the panel list of a sixth of the
# figures that have one (measured over 800 folders of the 2026-09-21 run).
PANEL_SPACED = re.compile(r"(?:^|[\.\;\,\(]\s?|\s\s)([a-lA-L])\s+[A-Z]")
CAPTION_ENDS = re.compile(r"[\.\!\?\"\)\]]\s*$")
CAPTION_NEW_REF = re.compile(r"\s*(Extended\s+Data\s+|Supplementary\s+)?(Fig(?:ure)?\.?|Table)\s*\d", re.I)
CAPTION_HEADING = re.compile(r"\s*(Methods|Results|Discussion|Introduction|References|Acknowledge"
                             r"|Data availability|Code availability|Supplementary)", re.I)
CAPTION_MAX_BLOCKS = 4
CAP_RE = re.compile(
    r"(Extended\s+Data\s+Fig(?:ure)?\.?|Supplementary\s+Fig(?:ure)?\.?|Supplementary\s+Table|"
    r"Extended\s+Data\s+Table|Fig(?:ure)?\.?|Table)\s*([0-9]+)", re.I)
REF_RE = re.compile(
    r"(Extended\s+Data\s+|Supplementary\s+|Suppl\.\s*)?"
    r"(Fig(?:ure)?s?\.?|Tables?)\s*"
    r"(S?\d+[A-Za-z]?(?:\s*(?:[-–,]|and)\s*S?\d+[A-Za-z]?)*)", re.I)

s3 = boto3.client("s3")


def log(message: str) -> None:
    print(f"{datetime.now(UTC).isoformat(timespec='seconds')} {message}", flush=True)


# ---------------------------------------------------------------------------------------------
# Reading one paper's layout: the parts of the local extractor that decide what an asset is
# ---------------------------------------------------------------------------------------------

def strip_html(html: str | None) -> str:
    text = re.sub(r"<[^>]+>", " ", html or "")
    for entity, plain in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")):
        text = text.replace(entity, plain)
    return re.sub(r"\s+", " ", text).strip()


def asset_name(caption: str) -> tuple[str | None, str | None]:
    """``Extended Data Fig. 3`` -> ``("extdatafig3", "Extended Data Fig. 3")``."""
    found = CAP_RE.search(caption)
    if not found:
        return None, None
    kind, number = found.group(1).lower(), found.group(2)
    if "extended" in kind and "table" in kind:
        return f"extdatatable{number}", f"Extended Data Table {number}"
    if "extended" in kind:
        return f"extdatafig{number}", f"Extended Data Fig. {number}"
    if "supplementary" in kind and "table" in kind:
        return f"supptable{number}", f"Supplementary Table {number}"
    if "supplementary" in kind:
        return f"suppfig{number}", f"Supplementary Fig. {number}"
    if "table" in kind:
        return f"table{number}", f"Table {number}"
    return f"figure{number}", f"Figure {number}"


def collect(page: dict) -> list[tuple[str, dict]]:
    items: list[tuple[str, dict]] = []

    def walk(node: dict) -> None:
        kind = node.get("block_type")
        if kind in FIG_KINDS:
            items.append(("fig", node))
        elif kind == "Table":
            items.append(("tab", node))
        elif kind == "Caption":
            items.append(("cap", node))
        elif kind in TEXT_KINDS:
            items.append(("txt", node))
        for child in node.get("children") or ():
            walk(child)

    walk(page)
    return items


def _rect_distance(a, b) -> float:
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return (dx * dx + dy * dy) ** 0.5


def nearest_caption(items: list[tuple[str, dict]], index: int) -> str:
    """The caption closest to this figure on the same page; one above it is penalised.

    A caption block is preferred, but a body block that starts "Figure 3" is accepted at a
    penalty, because some papers do not tag their captions at all.
    """
    figure = items[index][1].get("bbox")
    if not figure:
        return ""
    candidates = []
    for other, (kind, node) in enumerate(items):
        if other == index:
            continue
        box = node.get("bbox")
        if not box:
            continue
        text = strip_html(node.get("html"))
        if kind == "cap":
            penalty = 0
        elif kind == "txt" and re.match(
                r"\s*(Extended\s+Data\s+|Supplementary\s+)?(Fig(?:ure)?\.?|Table)\s*\d", text, re.I):
            penalty = 60
        else:
            continue
        if not CAP_RE.search(text):
            continue
        above = box[3] < figure[1] + 5
        candidates.append((_rect_distance(figure, box) + penalty + (80 if above else 0), text))
    if not candidates:
        return ""
    candidates.sort(key=lambda c: c[0])
    return extend_caption(items, candidates[0][1])


def extend_caption(items: list[tuple[str, dict]], caption: str) -> str:
    """Join the blocks that continue a caption whose sentence has not finished.

    A two-column paper splits a long caption across layout blocks, so the block marker labels as
    the caption often stops mid-word: "... d GFAP+ area fraction (reactive astro-". The local run
    of 2026-09-21 measured this and repaired it afterwards; here it is done in the one pass.

    Only an unfinished caption is touched, and only the blocks that follow it in reading order:
    at most four, nothing shorter than 20 characters, and the run stops at the next figure or
    table reference or at a section heading, because those begin something else.
    """
    if not caption or CAPTION_ENDS.search(caption.strip()):
        return caption
    texts = [(kind, strip_html(node.get("html"))) for kind, node in items]
    start = next((i for i, (kind, text) in enumerate(texts) if kind == "cap" and text == caption), None)
    if start is None:
        return caption
    joined = caption.strip()
    added = 0
    for kind, text in texts[start + 1:]:
        if added >= CAPTION_MAX_BLOCKS:
            break
        if kind not in ("txt", "cap") or not text or len(text) < 20:
            continue
        if CAPTION_NEW_REF.match(text) or CAPTION_HEADING.match(text):
            break
        if text in joined:
            continue
        joined = f"{joined} {text}".strip()
        added += 1
        if CAPTION_ENDS.search(joined):
            break
    return joined


def _numbers(chunk: str) -> set[str]:
    """``2-4`` gives 2 and 4, not 3: expanding a range invents mentions that are not there."""
    out = set()
    for part in re.split(r"\s*(?:[,–-]|and)\s*", chunk):
        found = re.match(r"S?(\d+)", part.strip(), re.I)
        if found:
            out.add(found.group(1))
    return out


def mentions_for(fulltext: str, key: str) -> list[str]:
    """Every sentence in the paper that refers to this asset, and only this one."""
    parsed = re.match(r"([a-z]+)(\d+)$", key)
    if not parsed:
        return []
    kind, number = parsed.group(1), parsed.group(2)
    want_extended = kind.startswith("extdata")
    want_supplementary = kind.startswith("supp")
    want_table = kind.endswith("table")
    out, seen = [], set()
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", fulltext):
        text = sentence.strip()
        if len(text) < 25:
            continue
        for reference in REF_RE.finditer(text):
            prefix = (reference.group(1) or "").lower()
            if ("extended" in prefix) != want_extended:
                continue
            if ("supplement" in prefix or "suppl." in prefix) != want_supplementary:
                continue
            if reference.group(2).lower().startswith("table") != want_table:
                continue
            if number in _numbers(reference.group(3)):
                if text not in seen:
                    seen.add(text)
                    out.append(text)
                break
    return out[:MAX_MENTIONS]


def panels_from_caption(caption: str) -> list[str]:
    """``(a) ... (b) ...`` -> ``["a", "b"]``, stopping at the first letter the caption skips.

    Two shapes, because journals use both and a paper that uses the second one would otherwise
    lose its panel list entirely: ``(a)``/``a,``/``a.`` with punctuation after the letter, and
    ``a Amyloid burden, b Tau burden`` where only a space follows. The label itself is removed
    first, or "Fig. 5 a ..." would read the label's own letter. A single letter is not a panel
    list -- an ordinary sentence produces one by chance -- so two is the minimum.
    """
    body = caption[caption.find("|") + 1:] if 0 <= caption.find("|") < 120 else caption
    label = re.match(r"\s*(Extended\s+Data\s+|Supplementary\s+)?(Fig(?:ure)?\.?|Table)\s*\d+\s*[\.\:\|]?\s*",
                     body, re.I)
    if label:
        body = body[label.end():]
    found = {m.group(1).lower() for m in PANEL_PUNCT.finditer(body)}
    found |= {m.group(1).lower() for m in PANEL_SPACED.finditer(body)}
    panels = []
    for letter in LETTERS:
        if letter not in found:
            break
        panels.append(letter)
    return panels if len(panels) >= 2 else []


def related_keys(sentences: list[str], self_key: str) -> list[str]:
    """The other assets cited in the same sentences: what a reader is sent to alongside this one."""
    keys = set()
    for sentence in sentences:
        for reference in REF_RE.finditer(sentence):
            prefix = (reference.group(1) or "").lower()
            table = reference.group(2).lower().startswith("table")
            if "extended" in prefix:
                base = "extdatatable" if table else "extdatafig"
            elif "supplement" in prefix or "suppl." in prefix:
                base = "supptable" if table else "suppfig"
            else:
                base = "table" if table else "figure"
            for number in _numbers(reference.group(3)):
                keys.add(f"{base}{number}")
    keys.discard(self_key)
    return sorted(keys, key=lambda k: (re.sub(r"\d+", "", k), int(re.sub(r"\D", "", k) or 0)))[:MAX_RELATED]


# ---------------------------------------------------------------------------------------------
# One paper: marker, crops, the Markdown the answers read
# ---------------------------------------------------------------------------------------------

def marker_json(pdf: Path, workdir: Path, converter) -> dict:
    """The layout JSON marker produces for this PDF."""
    rendered = converter(str(pdf))
    # "metadata" is excluded because marker's own writer excludes it (marker/output.py:66) and
    # because it cannot be serialised: a plain model_dump() of a JSONOutput raises
    # "TypeError: unhashable type: 'dict'" on its table statistics. Nothing below reads it.
    if hasattr(rendered, "model_dump_json"):
        return json.loads(rendered.model_dump_json(exclude=["metadata"]))
    payload = rendered.model_dump() if hasattr(rendered, "model_dump") else rendered
    if isinstance(payload, dict) and "children" in payload:
        return payload
    # Older marker builds hand back the document under another name; keep the whole thing.
    return payload if isinstance(payload, dict) else json.loads(json.dumps(payload, default=str))


def assets_markdown(stem: str, entries: list[dict]) -> str:
    """The one file per paper, in the shape the 11,554 already in S3 have."""
    kinds = {}
    for entry in entries:
        kinds[entry["kind"]] = kinds.get(entry["kind"], 0) + 1
    summary = ", ".join(f"{name.title()}s {count}" for name, count in sorted(kinds.items()))
    lines = [f"# {stem}", "", f"{len(entries)} items — {summary}", ""]
    for entry in entries:
        lines += [f"## {entry['label']}", "", f"![{entry['label']}]({entry['image']})", "",
                  "### Caption", "", entry["caption"] or "(no caption found)", ""]
        if entry.get("panels"):
            lines += ["### Panels", "", ", ".join(entry["panels"]), ""]
        if entry.get("mentions"):
            lines += ["### Mentioned in the text", ""] + [f"- {s}" for s in entry["mentions"]] + [""]
        if entry.get("related"):
            lines += ["### Cited alongside", "", ", ".join(entry["related"]), ""]
        if entry.get("table_text"):
            lines += ["### Table (extracted)", "", entry["table_text"], ""]
        lines += ["### Source", "",
                  f"- page {entry['page']}, bbox (pt) {entry['bbox']}, rendered at {entry['dpi']} dpi", ""]
    return "\n".join(lines)


def process(stem: str, converter, workdir: Path) -> dict:
    """Crop one paper's figures and tables and put them beside its stored text."""
    prefix = f"papers/{stem}/assets/"
    if not FORCE:
        try:
            s3.head_object(Bucket=BUCKET, Key=prefix + "assets.md")
            return {"stem": stem, "outcome": "already_there"}
        except s3.exceptions.ClientError:
            pass
    pdf = workdir / f"{stem}.pdf"
    # Two layouts, because the two ingest routes store an original differently: an uploaded PDF
    # goes to papers/{stem}/original.pdf beside its clean.md, and an OpenAlex-hosted one goes to
    # papers/{work_id}.pdf beside sources/{work_id}.md. The crops are written to the folder either
    # way, so a paper's assets are in one place whichever route brought it in.
    for key in (f"papers/{stem}/original.pdf", f"papers/{stem}.pdf"):
        try:
            s3.download_file(BUCKET, key, str(pdf))
            break
        except Exception as exc:  # noqa: BLE001 - one paper never stops the batch
            last = exc
    else:
        return {"stem": stem, "outcome": "no_pdf", "error": type(last).__name__}

    import pypdfium2 as pdfium

    started = time.monotonic()
    try:
        document = marker_json(pdf, workdir, converter)
    except Exception as exc:  # noqa: BLE001
        return {"stem": stem, "outcome": "marker_failed", "error": type(exc).__name__,
                "message": str(exc)[:200]}
    marker_seconds = round(time.monotonic() - started, 1)

    pages = document.get("children") or []
    fulltext = " ".join(strip_html(node.get("html"))
                        for page in pages for kind, node in collect(page) if kind == "txt")

    groups: dict[str, dict] = {}
    for page_index, page in enumerate(pages):
        items = collect(page)
        for index, (kind, node) in enumerate(items):
            if kind not in ("fig", "tab"):
                continue
            caption = nearest_caption(items, index)
            key, label = asset_name(caption)
            box = node.get("bbox")
            if not key or not box:
                continue
            if key in groups and groups[key]["page"] == page_index:
                old = groups[key]["bbox"]
                groups[key]["bbox"] = [min(old[0], box[0]), min(old[1], box[1]),
                                       max(old[2], box[2]), max(old[3], box[3])]
            elif key not in groups:
                groups[key] = {"bbox": list(box), "page": page_index, "caption": caption,
                               "label": label, "kind": "table" if kind == "tab" else "figure",
                               "html": node.get("html") if kind == "tab" else None}
    if not groups:
        return {"stem": stem, "outcome": "no_assets", "marker_seconds": marker_seconds}

    pdf_document = pdfium.PdfDocument(str(pdf))
    rendered_pages: dict[int, tuple] = {}
    entries: list[dict] = []
    images: list[tuple[str, Path]] = []
    for key, group in sorted(groups.items()):
        page_index = group["page"]
        if page_index not in rendered_pages:
            page = pdf_document[page_index]
            width, height = page.get_size()
            scale = SCALE
            if width * height * SCALE * SCALE > MAX_PIXELS:
                scale = (MAX_PIXELS / (width * height)) ** 0.5
            rendered_pages[page_index] = (page.render(scale=scale).to_pil().convert("RGB"), scale)
        image, scale = rendered_pages[page_index]
        x0, y0, x1, y1 = (value * scale for value in group["bbox"])
        box = (max(0, int(x0) - PAD), max(0, int(y0) - PAD),
               min(image.width, int(x1) + PAD), min(image.height, int(y1) + PAD))
        if box[2] - box[0] < 60 or box[3] - box[1] < 40:
            continue
        name = f"{key}.png"
        path = workdir / name
        image.crop(box).save(path)
        images.append((name, path))
        mentions = mentions_for(fulltext, key)
        entries.append({
            "key": key, "label": group["label"], "kind": group["kind"], "page": page_index + 1,
            "image": name, "markdown": "assets.md",
            "width": box[2] - box[0], "height": box[3] - box[1], "dpi": round(scale * 72),
            "caption": group["caption"], "caption_chars": len(group["caption"]),
            "mentions": mentions, "related": related_keys(mentions, key),
            "panels": panels_from_caption(group["caption"]),
            "bbox": [round(v, 1) for v in group["bbox"]],
            "table_text": strip_html(group["html"])[:MAX_TABLE_CHARS] if group["html"] else None,
        })
    if not entries:
        return {"stem": stem, "outcome": "no_assets", "marker_seconds": marker_seconds}

    manifest = {
        "paper": stem, "dpi": DPI, "s3_prefix": prefix, "extractor": EXTRACTOR,
        "extracted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "images_stored": False,
        "images_note": ("The text of these assets is stored; the PNG files named in `image` are "
                        "being uploaded. An absent object means not yet uploaded, not a figure "
                        "without an image."),
        "assets": [{k: v for k, v in entry.items() if k not in ("caption", "mentions", "table_text")}
                   for entry in entries],
    }
    # Text first: it is what an answer reads, and it must never wait on 100 MB of images.
    s3.put_object(Bucket=BUCKET, Key=prefix + "assets.md",
                  Body=assets_markdown(stem, entries).encode("utf-8"),
                  ContentType="text/markdown; charset=utf-8")
    s3.put_object(Bucket=BUCKET, Key=prefix + "manifest.json",
                  Body=(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n").encode("utf-8"),
                  ContentType="application/json")
    written = 0
    for name, path in images:
        s3.upload_file(str(path), BUCKET, prefix + name, ExtraArgs={"ContentType": "image/png"})
        written += path.stat().st_size
        path.unlink(missing_ok=True)
    manifest["images_stored"] = True
    manifest["images_note"] = f"{len(images)} PNG files are stored under {prefix}."
    s3.put_object(Bucket=BUCKET, Key=prefix + "manifest.json",
                  Body=(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n").encode("utf-8"),
                  ContentType="application/json")
    pdf.unlink(missing_ok=True)
    return {"stem": stem, "outcome": "extracted", "assets": len(entries),
            "figures": sum(1 for e in entries if e["kind"] == "figure"),
            "tables": sum(1 for e in entries if e["kind"] == "table"),
            "image_bytes": written, "marker_seconds": marker_seconds}


def build_converter():
    """One marker converter for the whole batch; the weights load once, not once per paper."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    # OCR is off, and that is the whole design rather than a workaround. This worker is the user's
    # stage one: cut the figures and tables out of a born-digital paper and write down what the
    # paper itself already says about each. Every ingested paper is a publisher PDF with an
    # embedded text layer, so there is nothing here for character recognition to add.
    #
    # Leaving it on is not free. marker's line builder flags a text block whose embedded text looks
    # garbled, and in the default "balanced" mode a single flagged block promotes the entire page to
    # full-page recognition (marker/builders/line.py:469-481). That recognition runs through surya,
    # and surya picks its backend from the hardware: on anything without an NVIDIA GPU
    # (_autodetect_backend) it wants to serve the model through llama.cpp and aborts with
    # "llama-server binary not found". A Fargate task has no GPU, so on this path every paper with
    # one odd-looking block would fail. Stage two, which reads these crops again, is a separate
    # decision and does not run here.
    #
    # pdftext forks its own workers by default and one of them died on a 16 MB PDF here on
    # 2026-09-22; a container with a modest process limit is exactly where that happens, so the
    # text extraction stays in this process and the batch keeps its parallelism at the task level.
    # The renderer is named, not configured: "output_format" is a command-line option that
    # marker's ConfigParser turns into a renderer class, and a PdfConverter built directly
    # ignores it and returns Markdown. Everything below reads the layout tree, so it must be JSON.
    return PdfConverter(artifact_dict=create_model_dict(),
                        renderer="marker.renderers.json.JSONRenderer",
                        config={"pdftext_workers": 1, "disable_tqdm": True, "disable_ocr": True})


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    stems = argv or STEMS
    if not stems and JOB_KEY:
        stems = json.loads(s3.get_object(Bucket=BUCKET, Key=JOB_KEY)["Body"].read())
    if not isinstance(stems, list) or not stems:
        log("no papers to do: pass stems as arguments, in STEMS, or in a JOB_KEY manifest")
        return 0
    log(f"{len(stems)} papers; loading marker")
    converter = build_converter()
    log("marker ready")
    counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory() as raw:
        workdir = Path(raw)
        for stem in stems:
            result = process(str(stem), converter, workdir)
            counts[result["outcome"]] = counts.get(result["outcome"], 0) + 1
            log(f"  {str(stem)[:52]:54s} {result['outcome']:14s} "
                f"assets={result.get('assets', 0)} marker={result.get('marker_seconds', 0)}s")
    log(f"done: {json.dumps(counts, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
