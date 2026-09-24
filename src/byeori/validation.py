from __future__ import annotations

from .config import Settings


SOURCE_SECTIONS = (
    "## Citation",
    "## Methods",
    "## Results",
    "## Limitations",
    "## Evidence boundary",
)
PAPER_SECTIONS = (
    "## Summary",
    "## Key findings",
    "## Limitations",
    "## Related pages",
)
OVERVIEW_SECTIONS = (
    "## Scope",
    "## Synthesis",
    "## Open questions",
    "## Related papers",
)
LLM_WIKI_SOURCE_SECTIONS = (
    "## One-line Summary",
    "## 1. Document Information",
    "## 2. Key Contributions",
    "## 3. Methodology and Architecture",
    "## 4. Key Results and Benchmarks",
    "## 5. Limitations and Future Work",
    "## 6. Related Work",
    "## 7. Glossary",
)
LLM_WIKI_PAGE_SECTIONS = (
    "## Summary",
    "## Key Contributions",
    "## Methodology and Architecture",
    "## Results",
    "## Related Papers",
)
NON_PAPER_DIRS = {"papers", "overviews", "questions", "drafts", "concepts", "projects", "figures"}
QUESTION_SECTIONS = (
    "## Question",
    "## Sharper follow-up",
    "## What the knowledge base holds",
    "## Tentative answer from the knowledge base",
    "## Related Pages",
)
CONCEPT_SECTIONS = ("## Definition", "## What the notes show", "## Disagreements and limits", "## Related concepts", "## Notes")
SUBTOPIC_SECTIONS = ("## Scope", "## Findings", "## Comparison", "## Open questions", "## Concepts", "## Notes")
CATEGORY_SECTIONS = ("## Landscape", "## Subtopics", "## Key concepts", "## Open questions", "## Coverage")
KIND_SECTIONS = {"concept": CONCEPT_SECTIONS, "subtopic": SUBTOPIC_SECTIONS, "category": CATEGORY_SECTIONS}



def page_errors(key: str, text: str) -> list[str]:
    """Structure checks used by the AWS worker; no client page retrieval."""
    from .synthesis_manifest import parse_frontmatter
    fields, _ = parse_frontmatter(text)
    folder = key.split("/")[1]
    if folder == "sources":
        schemas = (SOURCE_SECTIONS, LLM_WIKI_SOURCE_SECTIONS)
    elif folder == "overviews":
        schemas = (KIND_SECTIONS.get(fields.get("kind"), OVERVIEW_SECTIONS),)
    elif folder == "concepts":
        schemas = (CONCEPT_SECTIONS,)
    elif folder == "questions":
        schemas = (QUESTION_SECTIONS,)
    elif folder == "papers":
        schemas = (PAPER_SECTIONS,)
    else:
        schemas = (LLM_WIKI_PAGE_SECTIONS,)
    missing = min(([heading for heading in schema if heading not in text] for schema in schemas), key=len)
    return [f"missing {heading}" for heading in missing]


def validate(settings: Settings, *, store=None) -> list[str]:
    """Validate the wiki in AWS; receive only errors and pagination metadata."""
    from .aws_store import AwsStore
    store = store or AwsStore(settings)
    errors, count, cursor = [], 0, None
    while True:
        result = store.validate_wiki(cursor=cursor)
        count += result["checked"]
        errors.extend(result["errors"])
        cursor = result.get("next_cursor")
        if not cursor:
            break
    if not count:
        errors.append("No published wiki Markdown found in the configured S3 wiki/ prefix")
    return errors
