"""Bounded reads of a paper's stored supplementary tables, for the agents that run in AWS.

The kept supplementary files of a paper sit at ``papers/{stem}/supplementary/`` with a file guide
(``README.md``) and a ``manifest.json`` (``byeori.supplementary``). A note says what a
table holds; the value a question turns on -- one gene's fold change in one cell type, a cohort's
age range, the antibody a method used -- is only in the table. This module lets an agent read
that value without the table ever leaving AWS.

What it reads, with the standard library only (the Lambda package carries no dependencies):

- ``.xlsx``/``.xlsm`` workbooks, transitional or strict OOXML, streamed sheet by sheet;
- ``.csv``, ``.tsv`` and ``.txt`` tables;
- the tables inside a ``.docx``;
- any of these inside a ``.zip``, named ``archive.zip::inner/member.xlsx``.

PDF text, legacy ``.xls`` and nested archives are reported as not readable yet rather than
guessed at.

Every read is bounded. Objects are read with S3 range requests, so one sheet of a 150 MB workbook
costs the bytes of that sheet, not the workbook. A read returns at most ``max_rows`` rows of at
most ``MAX_COLUMNS`` cells of at most ``MAX_CELL_CHARS`` characters, stops at a deadline, and says
what it did not cover: the rows scanned, whether the scan finished, and how many rows matched in
total. Values are returned as stored. Dates stay Excel serial numbers and gene symbols Excel turned
into dates are not repaired, because the file guide records where that happened.

Only a file the manifest lists as kept may be opened, so a caller cannot name an arbitrary key.
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
import zipfile
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import PurePosixPath
from typing import Any
from xml.etree import ElementTree

from botocore.exceptions import ClientError

__all__ = [
    "MAX_CELL_CHARS", "MAX_COLUMNS", "ReadError", "S3RangeFile", "SupplementaryReader", "bounded_json",
    "readable_format",
]

MAX_CELL_CHARS = 200
MAX_COLUMNS = 60
HEADER_ROWS = 3
MAX_OBJECT_BYTES = 400 * 1024 * 1024       # the largest kept file is 281 MB
MAX_MEMBER_BYTES = 200 * 1024 * 1024       # a table inside an archive is decompressed to read it
SMALL_STRINGS_BYTES = 24 * 1024 * 1024     # shared-string tables below this are loaded whole
BLOCK_BYTES = 4 * 1024 * 1024
CACHE_BLOCKS = 8
STEM = re.compile(r"[a-z0-9][a-z0-9-]{2,200}")
TABLE_FORMATS = {".xlsx": "xlsx", ".xlsm": "xlsx", ".csv": "csv", ".tsv": "tsv", ".txt": "tsv", ".docx": "docx"}
NOT_YET = {".pdf": "PDF text is not extracted in AWS yet; read the file guide for what it holds",
           ".xls": "legacy .xls workbooks are not readable in AWS yet",
           ".doc": "legacy .doc files are not readable in AWS yet"}
MISSING = frozenset({"NoSuchKey", "404", "NotFound"})
NOTE = ("Values are as stored in the file: dates are Excel serial numbers, and gene symbols Excel "
        "turned into dates stay dates (the file guide says where).")


class ReadError(ValueError):
    """A read that cannot be done as asked; the message is safe to show the model."""


def readable_format(name: str) -> str | None:
    """The reader for a file name, or ``None`` when it has none."""
    return TABLE_FORMATS.get(PurePosixPath(name.lower()).suffix)


# ---------------------------------------------------------------------------------------------
# A seekable view of one S3 object
# ---------------------------------------------------------------------------------------------

class S3RangeFile(io.RawIOBase):
    """Read-only, seekable file over one S3 object, fetched in blocks with range requests.

    ``zipfile`` needs to seek to the central directory at the end and then to the one member it
    decompresses; with this view a workbook's sheet costs the bytes of that sheet.
    """

    def __init__(self, s3, bucket: str, key: str, size: int, *, block: int = BLOCK_BYTES):
        super().__init__()
        self.s3, self.bucket, self.key, self.size, self.block = s3, bucket, key, size, block
        self.position = 0
        self.requests = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def _block(self, index: int) -> bytes:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        start = index * self.block
        end = min(start + self.block, self.size) - 1
        response = self.s3.get_object(Bucket=self.bucket, Key=self.key, Range=f"bytes={start}-{end}")
        body = response["Body"]
        try:
            data = body.read()
        finally:
            body.close()
        self.requests += 1
        self._cache[index] = data
        while len(self._cache) > CACHE_BLOCKS:
            self._cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        end = self.size if size is None or size < 0 else min(self.size, self.position + size)
        parts = []
        while self.position < end:
            index, offset = divmod(self.position, self.block)
            chunk = self._block(index)[offset:offset + end - self.position]
            if not chunk:
                break
            parts.append(chunk)
            self.position += len(chunk)
        return b"".join(parts)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)

    def close(self) -> None:
        self._cache.clear()
        super().close()


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _column_index(reference: str) -> int:
    letters = "".join(ch for ch in reference if ch.isalpha()).upper()
    number = 0
    for ch in letters:
        number = number * 26 + (ord(ch) - 64)
    return number


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, rest = divmod(index - 1, 26)
        name = chr(65 + rest) + name
    return name


def _item_text(item) -> str:
    """The text of a shared-string or inline-string item: its runs, never its phonetic guide."""
    parts = []
    for child in item:
        name = _local(child.tag)
        if name == "t":
            parts.append(child.text or "")
        elif name == "r":
            parts.extend(t.text or "" for t in child if _local(t.tag) == "t")
    return "".join(parts)


def _cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "…"


class _Matcher:
    """Case-insensitive match of one cell against the text asked for."""

    def __init__(self, find: str | None, match: str):
        self.needle = (find or "").strip().casefold()
        if match not in {"exact", "contains"}:
            raise ReadError("match must be 'exact' or 'contains'")
        self.contains = match == "contains"

    def __bool__(self) -> bool:
        return bool(self.needle)

    def __call__(self, text: str) -> bool:
        value = text.strip().casefold()
        return self.needle in value if self.contains else value == self.needle


class _Collector:
    """Rows kept from one scan: the first non-empty rows as header, then the rows asked for."""

    def __init__(self, matcher: _Matcher, start_row: int, max_rows: int, deadline: float):
        self.matcher, self.start_row, self.max_rows, self.deadline = matcher, start_row, max_rows, deadline
        self.header: list[tuple[int, dict[int, str]]] = []
        self.rows: list[tuple[int, dict[int, str]]] = []
        self.scanned = 0
        self.matches = 0
        self.stopped: str | None = None

    def full(self) -> bool:
        return not self.matcher and len(self.rows) >= self.max_rows

    def offer(self, number: int, cells: dict[int, str], matched: bool | None = None) -> bool:
        """Consider one row; ``False`` means stop scanning."""
        self.scanned += 1
        if not cells:
            return True
        # The header is the first few non-empty rows (a legend line often sits above the column
        # names); with a later start row, only rows before it can be header.
        if len(self.header) < HEADER_ROWS and not self.rows and (self.start_row == 1 or number < self.start_row):
            self.header.append((number, cells))
        if self.matcher:
            hit = matched if matched is not None else any(self.matcher(v) for v in cells.values())
            if hit:
                self.matches += 1
                if len(self.rows) < self.max_rows:
                    self.rows.append((number, cells))
        elif number >= self.start_row and len(self.rows) < self.max_rows:
            if not self.header or self.header[-1][0] != number:
                self.rows.append((number, cells))
        if self.full():
            self.stopped = "rows"
            return False
        if self.scanned % 2000 == 0 and time.monotonic() > self.deadline:
            self.stopped = "time"
            return False
        return True


def _shape(rows: list[tuple[int, dict[int, str]]]) -> tuple[list[list[Any]], bool]:
    """Rows as ``[row number, [cells from column A]]``, at most ``MAX_COLUMNS`` wide."""
    shaped, cut = [], False
    for number, cells in rows:
        width = max(cells) if cells else 0
        if width > MAX_COLUMNS:
            cut, width = True, MAX_COLUMNS
        shaped.append([number, [_cell(cells.get(i)) for i in range(1, width + 1)]])
    return shaped, cut


# ---------------------------------------------------------------------------------------------
# Workbooks
# ---------------------------------------------------------------------------------------------

class _Workbook:
    """One ``.xlsx`` opened through ``zipfile``; sheets are streamed, never loaded whole."""

    def __init__(self, archive: zipfile.ZipFile):
        self.archive = archive
        names = set(archive.namelist())
        if "xl/workbook.xml" not in names:
            raise ReadError("not an Excel workbook (no xl/workbook.xml)")
        targets = {}
        rels = "xl/_rels/workbook.xml.rels"
        if rels in names:
            for element in ElementTree.fromstring(archive.read(rels)):
                if _local(element.tag) == "Relationship":
                    target = element.get("Target", "")
                    path = target.lstrip("/") if target.startswith("/") else f"xl/{target}"
                    targets[element.get("Id")] = str(PurePosixPath(path))
        self.sheets: list[dict[str, Any]] = []
        root = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        for element in root.iter():
            if _local(element.tag) != "sheet":
                continue
            rid = next((v for k, v in element.attrib.items() if _local(k) == "id"), None)
            self.sheets.append({"name": element.get("name"), "state": element.get("state") or "visible",
                                "part": targets.get(rid)})
        self.strings_part = "xl/sharedStrings.xml" if "xl/sharedStrings.xml" in names else None
        self._strings: list[str] | None = None

    def sheet(self, name: str | None) -> dict[str, Any]:
        if name is None:
            return self.sheets[0]
        for sheet in self.sheets:
            if sheet["name"] == name:
                return sheet
        folded = [s for s in self.sheets if (s["name"] or "").strip().casefold() == name.strip().casefold()]
        if folded:
            return folded[0]
        raise ReadError(f"no sheet named {name!r}; sheets are {[s['name'] for s in self.sheets]}")

    def _iter_strings(self) -> Iterator[str]:
        with self.archive.open(self.strings_part) as handle:
            for event, element in ElementTree.iterparse(handle, events=("end",)):
                if _local(element.tag) == "si":
                    yield _item_text(element)
                    element.clear()

    def strings_small(self) -> bool:
        if self.strings_part is None:
            return True
        return self.archive.getinfo(self.strings_part).file_size <= SMALL_STRINGS_BYTES

    def all_strings(self) -> list[str]:
        if self._strings is None:
            self._strings = list(self._iter_strings()) if self.strings_part else []
        return self._strings

    def matching_strings(self, matcher: _Matcher) -> set[int]:
        return {i for i, text in enumerate(self._iter_strings()) if matcher(text)} if self.strings_part else set()

    def resolve(self, wanted: set[int]) -> dict[int, str]:
        if not wanted or not self.strings_part:
            return {}
        if self._strings is not None:
            return {i: self._strings[i] for i in wanted if i < len(self._strings)}
        found = {}
        for i, text in enumerate(self._iter_strings()):
            if i in wanted:
                found[i] = text
                if len(found) == len(wanted):
                    break
        return found

    def scan(self, sheet: dict[str, Any], collector: _Collector) -> dict[str, Any]:
        """Stream one sheet into ``collector``; shared strings are resolved only for rows kept."""
        part = sheet["part"]
        if not part or part not in self.archive.namelist():
            raise ReadError(f"sheet {sheet['name']!r} has no worksheet part")
        matcher = collector.matcher
        small = self.strings_small()
        strings = self.all_strings() if small else None
        hits = set() if (small or not matcher) else self.matching_strings(matcher)
        dimension = None
        row_number = 0
        with self.archive.open(part) as handle:
            sheet_data = None
            for event, element in ElementTree.iterparse(handle, events=("start", "end")):
                tag = _local(element.tag)
                if event == "start":
                    if tag == "sheetData":
                        sheet_data = element
                    elif tag == "dimension":
                        dimension = element.get("ref")
                    continue
                if tag != "row":
                    continue
                row_number = int(element.get("r") or row_number + 1)
                cells: dict[int, str] = {}
                matched = False
                column = 0
                for c in element:
                    if _local(c.tag) != "c":
                        continue
                    reference = c.get("r")
                    column = _column_index(reference) if reference else column + 1
                    kind = c.get("t")
                    value = None
                    for child in c:
                        name = _local(child.tag)
                        if name == "v":
                            value = child.text
                        elif name == "is":
                            value = _item_text(child)
                    if value is None:
                        continue
                    if kind == "s":
                        index = int(value)
                        if strings is not None:
                            value = strings[index] if index < len(strings) else ""
                        else:
                            matched = matched or index in hits
                            cells[column] = ("\x00s", index)  # resolved after the scan
                            continue
                    elif kind == "b":
                        value = "TRUE" if value == "1" else "FALSE"
                    cells[column] = value
                    if matcher and not matched and isinstance(value, str) and matcher(value):
                        matched = True
                element.clear()
                if sheet_data is not None:
                    del sheet_data[:]
                keep_going = collector.offer(row_number, cells, matched if matcher else None)
                if not keep_going:
                    break
        # Replace shared-string placeholders in the rows that were kept.
        pending = {v[1] for _, row in collector.header + collector.rows for v in row.values()
                   if isinstance(v, tuple)}
        resolved = self.resolve(pending)
        for _, row in collector.header + collector.rows:
            for key, value in list(row.items()):
                if isinstance(value, tuple):
                    row[key] = resolved.get(value[1], "")
        return {"dimension": dimension}


# ---------------------------------------------------------------------------------------------
# Delimited text and Word tables
# ---------------------------------------------------------------------------------------------

def _scan_delimited(handle, delimiter: str, collector: _Collector) -> None:
    if isinstance(handle, io.RawIOBase):
        handle = io.BufferedReader(handle, buffer_size=1024 * 1024)
    text = io.TextIOWrapper(handle, encoding="utf-8", errors="replace", newline="")
    csv.field_size_limit(10 * 1024 * 1024)
    for number, row in enumerate(csv.reader(text, delimiter=delimiter), start=1):
        cells = {i: v for i, v in enumerate(row, start=1) if v != ""}
        if not collector.offer(number, cells):
            break


def _docx_tables(data: bytes) -> list[list[list[str]]]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    tables = []
    for table in root.iter():
        if _local(table.tag) != "tbl":
            continue
        rows = []
        for tr in table:
            if _local(tr.tag) != "tr":
                continue
            row = []
            for tc in tr:
                if _local(tc.tag) != "tc":
                    continue
                paragraphs = []
                for p in tc.iter():
                    if _local(p.tag) == "p":
                        paragraphs.append("".join(t.text or "" for t in p.iter() if _local(t.tag) == "t"))
                row.append(" ".join(x for x in paragraphs if x).strip())
            rows.append(row)
        tables.append(rows)
    return tables


# ---------------------------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------------------------

class SupplementaryReader:
    """Reads one bucket's supplementary files for an agent; every result is JSON-ready."""

    def __init__(self, s3, bucket: str):
        self.s3, self.bucket = s3, bucket
        self._manifests: dict[str, dict[str, Any]] = {}

    # -- manifest and guide ------------------------------------------------------------------

    def manifest(self, stem: str) -> dict[str, Any]:
        if not STEM.fullmatch(stem or ""):
            raise ReadError("use the exact paper stem from a wiki/sources/ note")
        if stem not in self._manifests:
            key = f"papers/{stem}/supplementary/manifest.json"
            try:
                body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"]
            except ClientError as exc:
                if exc.response.get("Error", {}).get("Code") in MISSING:
                    raise ReadError(f"{stem} has no stored supplementary files") from exc
                raise
            try:
                self._manifests[stem] = json.loads(body.read())
            finally:
                body.close()
        return self._manifests[stem]

    def has_supplementary(self, stem: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=f"papers/{stem}/supplementary/manifest.json")
            return True
        except ClientError:
            return False

    def files(self, stem: str) -> list[dict[str, Any]]:
        """The kept files: name, what they are, and whether a table read can open them."""
        listed = []
        for entry in self.manifest(stem).get("files", []):
            if entry.get("decision") != "upload":
                continue
            suffix = PurePosixPath(entry["file"].lower()).suffix
            listed.append({"file": entry["file"], "label": entry.get("label"), "kind": entry.get("kind"),
                           "bytes": entry.get("bytes"), "uses": entry.get("uses") or [],
                           "readable": "archive members" if suffix == ".zip" else bool(readable_format(entry["file"])),
                           "summary": entry.get("summary")})
        return listed

    def guide(self, stem: str, *, start: int = 0, max_chars: int = 8000) -> dict[str, Any]:
        """The file guide, one window of it, and the list of kept files."""
        manifest = self.manifest(stem)
        key = manifest.get("guide") or f"papers/{stem}/supplementary/README.md"
        body = self.s3.get_object(Bucket=self.bucket, Key=key)["Body"]
        try:
            text = body.read().decode("utf-8", "replace")
        finally:
            body.close()
        start = max(0, int(start))
        end = min(len(text), start + max(1, int(max_chars)))
        return {"stem": stem, "key": key, "text": text[start:end], "start": start,
                "next_start": end if end < len(text) else None, "total_chars": len(text),
                "files": self.files(stem)}

    # -- tables ------------------------------------------------------------------------------

    def _entry(self, stem: str, name: str) -> tuple[dict[str, Any], str | None]:
        outer, _, member = name.partition("::")
        for entry in self.manifest(stem).get("files", []):
            if entry.get("file") == outer:
                if entry.get("decision") != "upload":
                    raise ReadError(f"{outer} was not kept, so it is not stored")
                return entry, member or None
        raise ReadError(f"{outer} is not among {stem}'s kept files; list them with the file guide")

    def _object(self, stem: str, entry: dict[str, Any]) -> tuple[str, int]:
        key = f"papers/{stem}/supplementary/{entry['file']}"
        size = int(entry.get("bytes") or self.s3.head_object(Bucket=self.bucket, Key=key)["ContentLength"])
        if size > MAX_OBJECT_BYTES:
            raise ReadError(f"{entry['file']} is {size:,} bytes, above the {MAX_OBJECT_BYTES:,}-byte read limit")
        return key, size

    def members(self, stem: str, name: str) -> list[dict[str, Any]]:
        entry, _ = self._entry(stem, name)
        key, size = self._object(stem, entry)
        with zipfile.ZipFile(S3RangeFile(self.s3, self.bucket, key, size)) as archive:
            return [{"member": f"{entry['file']}::{i.filename}", "bytes": i.file_size,
                     "readable": bool(readable_format(i.filename))}
                    for i in archive.infolist()
                    if not i.is_dir() and not i.filename.startswith("__MACOSX/")
                    and not PurePosixPath(i.filename).name.startswith(("._", "~$"))]

    def table(self, stem: str, file: str, *, sheet: str | None = None, find: str | None = None,
              match: str = "exact", start_row: int = 1, max_rows: int = 30,
              seconds: float = 60.0) -> dict[str, Any]:
        """Rows of one kept table: the header rows plus the rows matching ``find`` or from ``start_row``.

        Without ``sheet``, a workbook read that has ``find`` searches every sheet in order until the
        deadline; one without ``find`` returns the list of sheets and the head of the first.
        """
        deadline = time.monotonic() + max(1.0, float(seconds))
        matcher = _Matcher(find, match)
        max_rows = max(1, min(int(max_rows), 200))
        start_row = max(1, int(start_row))
        entry, member = self._entry(stem, file)
        key, size = self._object(stem, entry)
        outer_suffix = PurePosixPath(entry["file"].lower()).suffix
        inner = member or entry["file"]
        suffix = PurePosixPath(inner.lower()).suffix
        if outer_suffix == ".zip" and member is None:
            raise ReadError(f"{entry['file']} is an archive; name one member as '{entry['file']}::<member>' "
                            f"(members are listed by reading it without a member)")
        kind = readable_format(inner)
        if kind is None:
            raise ReadError(NOT_YET.get(suffix, f"{suffix or 'this'} files are not readable as tables"))
        result: dict[str, Any] = {"stem": stem, "file": file, "key": key, "sha256": entry.get("sha256"),
                                  "format": kind, "find": find or None, "match": match if find else None,
                                  "note": NOTE}
        source = S3RangeFile(self.s3, self.bucket, key, size)
        try:
            if member is not None:
                with zipfile.ZipFile(source) as archive:
                    try:
                        info = archive.getinfo(member)
                    except KeyError as exc:
                        raise ReadError(f"{member} is not in {entry['file']}") from exc
                    if info.file_size > MAX_MEMBER_BYTES:
                        raise ReadError(f"{member} is {info.file_size:,} bytes uncompressed, above the read limit")
                    result["member"] = member
                    return self._read(kind, archive.open(member), result, sheet, matcher, start_row, max_rows,
                                      deadline)
            return self._read(kind, source, result, sheet, matcher, start_row, max_rows, deadline)
        finally:
            result["range_requests"] = source.requests
            source.close()

    def _read(self, kind: str, handle, result: dict[str, Any], sheet: str | None, matcher: _Matcher,
              start_row: int, max_rows: int, deadline: float) -> dict[str, Any]:
        if kind in {"csv", "tsv"}:
            collector = _Collector(matcher, start_row, max_rows, deadline)
            _scan_delimited(handle, "," if kind == "csv" else "\t", collector)
            return self._finish(result, collector, sheet=None)
        if kind == "docx":
            data = handle.read()
            tables = _docx_tables(data)
            result["sheets"] = [{"name": f"table {i}", "rows": len(t)} for i, t in enumerate(tables, start=1)]
            chosen = range(1, len(tables) + 1)
            if sheet:
                number = re.fullmatch(r"(?:table\s*)?(\d+)", sheet.strip().lower())
                if not number or not 1 <= int(number.group(1)) <= len(tables):
                    raise ReadError(f"no {sheet!r}; this document has {len(tables)} tables named 'table N'")
                chosen = [int(number.group(1))]
            elif not matcher:
                chosen = [1] if tables else []
            return self._each(result, [(f"table {i}", tables[i - 1]) for i in chosen],
                              matcher, start_row, max_rows, deadline)
        workbook = _Workbook(zipfile.ZipFile(handle))
        result["sheets"] = [{"name": s["name"], "state": s["state"]} for s in workbook.sheets]
        if sheet is not None:
            targets = [workbook.sheet(sheet)]
        elif matcher:
            targets = workbook.sheets
        else:
            targets = workbook.sheets[:1]
        parts = []
        for target in targets:
            collector = _Collector(matcher, start_row, max_rows, deadline)
            info = workbook.scan(target, collector)
            parts.append(self._part(target["name"], collector, info.get("dimension")))
            if collector.stopped == "time":
                break
        return self._combine(result, parts, len(targets))

    def _each(self, result, named_rows, matcher, start_row, max_rows, deadline):
        parts = []
        for name, rows in named_rows:
            collector = _Collector(matcher, start_row, max_rows, deadline)
            for number, row in enumerate(rows, start=1):
                if not collector.offer(number, {j: v for j, v in enumerate(row, 1) if v}):
                    break
            parts.append(self._part(name, collector, None))
        return self._combine(result, parts, len(named_rows))

    @staticmethod
    def _part(name: str | None, collector: _Collector, dimension: str | None) -> dict[str, Any]:
        header, cut_header = _shape(collector.header)
        rows, cut_rows = _shape(collector.rows)
        part = {"sheet": name, "dimension": dimension, "header": header, "rows": rows,
                "rows_scanned": collector.scanned, "complete": collector.stopped is None,
                "stopped": collector.stopped, "columns_cut_at": MAX_COLUMNS if (cut_header or cut_rows) else None}
        if collector.matcher:
            part["matches"] = collector.matches
            part["returned"] = len(collector.rows)
            # Where in each row the text was found: a sheet that sets several tables side by side
            # puts other genes in the same row, and the column says which table the hit belongs to.
            part["hit_columns"] = {str(number): [_column_name(c) for c, v in sorted(cells.items())
                                                 if isinstance(v, str) and collector.matcher(v)]
                                   for number, cells in collector.rows}
        return part

    def _finish(self, result: dict[str, Any], collector: _Collector, *, sheet: str | None) -> dict[str, Any]:
        return self._combine(result, [self._part(sheet, collector, None)], 1)

    @staticmethod
    def _combine(result: dict[str, Any], parts: list[dict[str, Any]], planned: int) -> dict[str, Any]:
        if result.get("find"):
            with_hits = [p for p in parts if p.get("matches")]
            result["matches_total"] = sum(p.get("matches", 0) for p in parts)
            result["sheets_searched"] = len(parts)
            result["sheets_not_reached"] = planned - len(parts)
            result["parts"] = with_hits or parts[:1]
        else:
            result["parts"] = parts
        for part in result["parts"]:
            part["columns"] = [_column_name(i) for i in range(1, 1 + max(
                [len(r[1]) for r in part["header"] + part["rows"]] or [0]))]
        return result

    def search(self, stem: str, find: str, *, match: str = "exact", max_rows: int = 10,
               seconds: float = 60.0) -> dict[str, Any]:
        """Look for ``find`` in every readable kept table of one paper, smallest files first."""
        deadline = time.monotonic() + max(1.0, float(seconds))
        found, skipped = [], []
        candidates = []
        for entry in self.files(stem):
            if entry["readable"] is True:
                candidates.append((entry.get("bytes") or 0, entry["file"]))
            elif entry["readable"] == "archive members":
                try:
                    for member in self.members(stem, entry["file"]):
                        if member["readable"]:
                            candidates.append((member["bytes"], member["member"]))
                except (ReadError, zipfile.BadZipFile) as exc:
                    skipped.append({"file": entry["file"], "reason": str(exc)[:200]})
            else:
                skipped.append({"file": entry["file"], "reason": "not a readable table"})
        for _, name in sorted(candidates):
            remaining = deadline - time.monotonic()
            if remaining <= 1:
                skipped.append({"file": name, "reason": "not reached before the deadline"})
                continue
            try:
                read = self.table(stem, name, find=find, match=match, max_rows=max_rows, seconds=remaining)
            except (ReadError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
                skipped.append({"file": name, "reason": str(exc)[:200]})
                continue
            if read.get("matches_total"):
                found.append(read)
        return {"stem": stem, "find": find, "match": match, "files_with_matches": found,
                "files_searched": len(candidates) - sum(1 for s in skipped if s["reason"].startswith("not reached")),
                "skipped": skipped, "note": NOTE}


def bounded_json(value: Any, max_chars: int) -> str:
    """``value`` as JSON of at most ``max_chars``: rows go from the end first, then cells are cut."""
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    trimmed = json.loads(text)
    parts = []
    stack = [trimmed]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if isinstance(item.get("rows"), list) and isinstance(item.get("header"), list):
                parts.append(item)
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)

    def size() -> int:
        return len(json.dumps(trimmed, ensure_ascii=False))

    while size() > max_chars and any(len(p["rows"]) > 1 for p in parts):
        longest = max(parts, key=lambda p: len(p["rows"]))
        longest["rows"].pop()
        longest["rows_cut_to_fit"] = True
    for cell_chars, width in ((80, MAX_COLUMNS), (40, 30), (20, 15)):
        if size() <= max_chars:
            break
        for part in parts:
            for row in part["header"] + part["rows"]:
                row[1] = [c if len(c) <= cell_chars else c[:cell_chars] + "…" for c in row[1][:width]]
            part["cells_cut_to_fit"] = {"chars": cell_chars, "columns": width}
    text = json.dumps(trimmed, ensure_ascii=False)
    return text if len(text) <= max_chars else text[:max_chars]
