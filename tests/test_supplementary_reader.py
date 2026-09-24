"""byeori.supplementary_reader: bounded table reads from stored supplementary files."""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from byeori import supplementary_reader as sr
from lab_fakes import MemoryS3

STEM = "doe-2024-a-paper"
MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
STRICT = "http://purl.oclc.org/ooxml/spreadsheetml/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
STRICT_REL = "http://purl.oclc.org/ooxml/officeDocument/relationships"


class RangeS3(MemoryS3):
    """``MemoryS3`` that also records the range requests it served."""

    def __init__(self, objects=None):
        super().__init__(objects)
        self.ranges: list[tuple[str, str]] = []

    def get_object(self, *, Bucket, Key, Range=None, **kwargs):
        if Range is not None:
            self.ranges.append((Key, Range))
        return super().get_object(Bucket=Bucket, Key=Key, Range=Range, **kwargs)


def cell(ref, value):
    if isinstance(value, int) and not isinstance(value, bool):
        return f'<c r="{ref}"><v>{value}</v></c>'
    if isinstance(value, float):
        return f'<c r="{ref}"><v>{value}</v></c>'
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{int(value)}</v></c>'
    return f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'


def workbook(sheets: dict[str, list[list]], *, shared: bool = True, strict: bool = False) -> bytes:
    """A minimal .xlsx: string cells go to the shared-string table when ``shared``."""
    main, rel = (STRICT, STRICT_REL) if strict else (MAIN, REL)
    strings: list[str] = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        entries, rels = [], []
        for i, (name, rows) in enumerate(sheets.items(), start=1):
            entries.append(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>')
            rels.append(f'<Relationship Id="rId{i}" Type="worksheet" Target="worksheets/sheet{i}.xml"/>')
            xml_rows = []
            for r, values in enumerate(rows, start=1):
                cells = []
                for c, value in enumerate(values, start=1):
                    if value is None:
                        continue
                    ref = f"{sr._column_name(c)}{r}"
                    if isinstance(value, str) and shared:
                        if value not in strings:
                            strings.append(value)
                        cells.append(f'<c r="{ref}" t="s"><v>{strings.index(value)}</v></c>')
                    else:
                        cells.append(cell(ref, value))
                xml_rows.append(f'<row r="{r}">{"".join(cells)}</row>')
            width = sr._column_name(max((len(r) for r in rows), default=1))
            z.writestr(f"xl/worksheets/sheet{i}.xml",
                       f'<worksheet xmlns="{main}"><dimension ref="A1:{width}{len(rows)}"/>'
                       f'<sheetData>{"".join(xml_rows)}</sheetData></worksheet>')
        z.writestr("xl/workbook.xml", f'<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>{"".join(entries)}'
                                      f'</sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",
                   f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   f'{"".join(rels)}</Relationships>')
        if shared:
            items = "".join(f"<si><t>{s}</t></si>" for s in strings)
            z.writestr("xl/sharedStrings.xml", f'<sst xmlns="{main}">{items}</sst>')
    return buffer.getvalue()


def docx(tables: list[list[list[str]]]) -> bytes:
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(
        "<w:tbl>" + "".join("<w:tr>" + "".join(f"<w:tc><w:p><w:r><w:t>{c}</w:t></w:r></w:p></w:tc>" for c in row)
                            + "</w:tr>" for row in table) + "</w:tbl>" for table in tables)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("word/document.xml", f'<w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>')
    return buffer.getvalue()


DEG = [["Supplementary Data 3: DEGs per cell type"], [],
       ["gene", "cluster", "avg_logFC", "p_val_adj"],
       ["SNCA", "DaN", 1.25, 1e-10], ["TH", "DaN", 2.5, 1e-20], ["SNCA", "Astro", -0.4, 0.02],
       ["GFAP", "Astro", 3.1, 1e-30]]


def world(files: dict[str, bytes], *, kept: dict[str, bool] | None = None) -> RangeS3:
    kept = kept or {}
    manifest = {"stem": STEM, "guide": f"papers/{STEM}/supplementary/README.md",
                "files": [{"file": name, "bytes": len(data), "sha256": "x" * 64, "kind": "data_table",
                           "decision": "upload" if kept.get(name, True) else "skip", "label": name,
                           "uses": ["gene_sets"], "summary": "s"} for name, data in files.items()]}
    objects = {f"papers/{STEM}/supplementary/{name}": data for name, data in files.items() if kept.get(name, True)}
    objects[f"papers/{STEM}/supplementary/manifest.json"] = json.dumps(manifest)
    objects[f"papers/{STEM}/supplementary/README.md"] = "# Supplementary files\n\n" + "guide text " * 50
    return RangeS3(objects)


def reader(files, **kw):
    s3 = world(files, **kw)
    return sr.SupplementaryReader(s3, "bucket"), s3


# --- workbooks ---------------------------------------------------------------------------------

@pytest.mark.parametrize("shared, strict", [(True, False), (False, False), (True, True)])
def test_find_returns_the_header_and_every_matching_row_with_row_numbers(shared, strict):
    read, _ = reader({"deg.xlsx": workbook({"DEG": DEG}, shared=shared, strict=strict)})

    result = read.table(STEM, "deg.xlsx", find="snca")

    part = result["parts"][0]
    assert result["matches_total"] == 2 and part["sheet"] == "DEG" and part["dimension"] == "A1:D7"
    assert [row[0] for row in part["rows"]] == [4, 6]
    assert part["rows"][0][1] == ["SNCA", "DaN", "1.25", "1e-10"]
    assert [row[0] for row in part["header"]] == [1, 3, 4]
    assert part["complete"] is True and part["columns"] == ["A", "B", "C", "D"]
    assert part["hit_columns"] == {"4": ["A"], "6": ["A"]}


def test_hit_columns_say_which_side_by_side_table_a_match_belongs_to():
    rows = [["gene", "logFC", None, "gene", "logFC"], ["TH", 1.0, None, "SNCA", 2.0], ["SNCA", 0.5, None, "GFAP", 3.0]]
    read, _ = reader({"blocks.xlsx": workbook({"S": rows})})

    part = read.table(STEM, "blocks.xlsx", find="SNCA")["parts"][0]

    assert part["hit_columns"] == {"2": ["D"], "3": ["A"]}


def test_exact_match_does_not_take_a_longer_symbol_but_contains_does():
    rows = [["gene"], ["APP"], ["APPL1"]]
    read, _ = reader({"t.xlsx": workbook({"S": rows})})

    assert read.table(STEM, "t.xlsx", find="APP")["matches_total"] == 1
    assert read.table(STEM, "t.xlsx", find="APP", match="contains")["matches_total"] == 2


def test_large_shared_string_tables_resolve_only_the_rows_kept(monkeypatch):
    monkeypatch.setattr(sr, "SMALL_STRINGS_BYTES", 0)
    read, _ = reader({"deg.xlsx": workbook({"DEG": DEG})})

    part = read.table(STEM, "deg.xlsx", find="GFAP")["parts"][0]

    assert part["rows"] == [[7, ["GFAP", "Astro", "3.1", "1e-30"]]]
    assert part["header"][1][1] == ["gene", "cluster", "avg_logFC", "p_val_adj"]


def test_without_find_the_first_sheet_is_read_from_start_row_and_every_sheet_is_listed():
    read, _ = reader({"deg.xlsx": workbook({"DEG": DEG, "Other": [["x"]]})})

    result = read.table(STEM, "deg.xlsx", start_row=5, max_rows=2)

    assert [s["name"] for s in result["sheets"]] == ["DEG", "Other"]
    part = result["parts"][0]
    assert [row[0] for row in part["rows"]] == [5, 6] and part["stopped"] == "rows"


def test_find_without_a_sheet_searches_every_sheet_and_names_one_by_name():
    read, _ = reader({"deg.xlsx": workbook({"A": [["x"], ["TH"]], "B": [["y"], ["TH"], ["TH"]]})})

    everywhere = read.table(STEM, "deg.xlsx", find="TH")
    only_b = read.table(STEM, "deg.xlsx", sheet="b", find="TH")

    assert everywhere["matches_total"] == 3 and everywhere["sheets_searched"] == 2
    assert [p["sheet"] for p in only_b["parts"]] == ["B"] and only_b["matches_total"] == 2
    with pytest.raises(sr.ReadError, match="no sheet named"):
        read.table(STEM, "deg.xlsx", sheet="missing")


def test_a_large_workbook_is_read_with_range_requests_not_downloaded_whole(monkeypatch):
    padding = {f"Pad{i}": [[f"p{i}_{j}_{k}" for j in range(40)] for k in range(300)] for i in range(6)}
    data = workbook({"DEG": DEG, **padding}, shared=False)
    s3 = world({"big.xlsx": data})
    original = sr.S3RangeFile.__init__
    monkeypatch.setattr(sr.S3RangeFile, "__init__",
                        lambda self, *a, **k: original(self, *a, **{**k, "block": 16 * 1024}))

    out = sr.SupplementaryReader(s3, "bucket").table(STEM, "big.xlsx", sheet="DEG", find="TH")

    fetched = sum(int(r.split("-")[1]) - int(r.removeprefix("bytes=").split("-")[0]) + 1 for _, r in s3.ranges)
    assert out["matches_total"] == 1
    assert fetched < len(data) / 3


def test_values_are_returned_as_stored_and_wide_rows_are_cut():
    wide = [[f"c{i}" for i in range(80)], [True, 45000, "x" * 500]]
    read, _ = reader({"w.xlsx": workbook({"S": wide}, shared=False)})

    part = read.table(STEM, "w.xlsx")["parts"][0]

    assert part["columns_cut_at"] == sr.MAX_COLUMNS and len(part["header"][0][1]) == sr.MAX_COLUMNS
    values = part["header"][1][1]
    assert values[0] == "TRUE" and values[1] == "45000" and values[2].endswith("…")
    assert len(values[2]) == sr.MAX_CELL_CHARS + 1


# --- other formats -----------------------------------------------------------------------------

def test_csv_tsv_and_docx_tables_are_read():
    csv_data = b"gene,log2FC\nSNCA,1.2\nTH,2.0\n"
    tsv_data = b"gene\tlog2FC\nSNCA\t1.2\n"
    doc = docx([[["Antibody", "Dilution"], ["GFAP", "1:500"]], [["Primer", "Seq"], ["TH-F", "ACGT"]]])
    read, _ = reader({"t.csv": csv_data, "t.tsv": tsv_data, "m.docx": doc})

    assert read.table(STEM, "t.csv", find="TH")["parts"][0]["rows"] == [[3, ["TH", "2.0"]]]
    assert read.table(STEM, "t.tsv", find="SNCA")["matches_total"] == 1
    found = read.table(STEM, "m.docx", find="TH-F")
    assert found["parts"][0]["sheet"] == "table 2" and found["parts"][0]["rows"] == [[2, ["TH-F", "ACGT"]]]
    assert [s["name"] for s in found["sheets"]] == ["table 1", "table 2"]
    assert read.table(STEM, "m.docx", sheet="table 1")["parts"][0]["header"][0][1] == ["Antibody", "Dilution"]


def test_archive_members_are_listed_and_read_by_name():
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("Tables/S1.xlsx", workbook({"DEG": DEG}))
        z.writestr("__MACOSX/._S1.xlsx", b"noise")
        z.writestr("legend.pdf", b"%PDF")
    read, _ = reader({"tables.zip": inner.getvalue()})

    members = read.members(STEM, "tables.zip")
    result = read.table(STEM, "tables.zip::Tables/S1.xlsx", find="TH")

    assert [m["member"] for m in members] == ["tables.zip::Tables/S1.xlsx", "tables.zip::legend.pdf"]
    assert [m["readable"] for m in members] == [True, False]
    assert result["matches_total"] == 1 and result["member"] == "Tables/S1.xlsx"
    with pytest.raises(sr.ReadError, match="is an archive"):
        read.table(STEM, "tables.zip")


def test_a_paper_wide_search_opens_every_readable_table():
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("S2.csv", "gene\nTH\n")
    read, _ = reader({"deg.xlsx": workbook({"DEG": DEG}), "t.zip": inner.getvalue(), "si.pdf": b"%PDF"})

    result = read.search(STEM, "TH")

    assert sorted(r["file"] for r in result["files_with_matches"]) == ["deg.xlsx", "t.zip::S2.csv"]
    assert result["skipped"] == [{"file": "si.pdf", "reason": "not a readable table"}]


# --- boundaries --------------------------------------------------------------------------------

def test_only_kept_files_of_a_valid_stem_are_opened():
    read, _ = reader({"deg.xlsx": workbook({"DEG": DEG}), "rs.pdf": b"%PDF"}, kept={"rs.pdf": False})

    with pytest.raises(sr.ReadError, match="exact paper stem"):
        read.table("../etc", "deg.xlsx")
    with pytest.raises(sr.ReadError, match="not among"):
        read.table(STEM, "../../wiki/index.md")
    with pytest.raises(sr.ReadError, match="was not kept"):
        read.table(STEM, "rs.pdf")
    with pytest.raises(sr.ReadError, match="no stored supplementary"):
        read.table("other-2020-paper", "deg.xlsx")


def test_pdf_and_legacy_formats_say_they_are_not_readable_yet():
    read, _ = reader({"si.pdf": b"%PDF", "old.xls": b"\xd0\xcf"})

    with pytest.raises(sr.ReadError, match="PDF text is not extracted"):
        read.table(STEM, "si.pdf")
    with pytest.raises(sr.ReadError, match="legacy .xls"):
        read.table(STEM, "old.xls")


def test_the_guide_is_read_in_windows_with_the_kept_files():
    read, _ = reader({"deg.xlsx": workbook({"DEG": DEG}), "rs.pdf": b"%PDF"}, kept={"rs.pdf": False})

    first = read.guide(STEM, max_chars=100)
    rest = read.guide(STEM, start=first["next_start"], max_chars=10_000)

    assert first["text"].startswith("# Supplementary files") and len(first["text"]) == 100
    assert rest["next_start"] is None and first["text"] + rest["text"] == first["text"] + rest["text"]
    assert [f["file"] for f in first["files"]] == ["deg.xlsx"] and first["files"][0]["readable"] is True


def test_a_scan_stops_at_its_deadline_and_says_so(monkeypatch):
    rows = [["gene"]] + [[f"G{i}"] for i in range(5000)]
    read, _ = reader({"big.csv": "\n".join(r[0] for r in rows).encode()})
    clock = iter([0.0] + [1000.0] * 10)
    monkeypatch.setattr(sr.time, "monotonic", lambda: next(clock, 1000.0))

    part = read.table(STEM, "big.csv", find="nothing-here")["parts"][0]

    assert part["stopped"] == "time" and part["complete"] is False and part["rows_scanned"] == 2000


def test_bounded_json_drops_rows_then_cuts_cells():
    result = {"parts": [{"header": [[1, ["gene"]]], "rows": [[i, ["x" * 150] * 20] for i in range(40)]}]}

    text = sr.bounded_json(result, 6000)

    assert len(text) <= 6000
    parsed = json.loads(text)
    assert parsed["parts"][0]["rows_cut_to_fit"] is True and len(parsed["parts"][0]["rows"]) >= 1
