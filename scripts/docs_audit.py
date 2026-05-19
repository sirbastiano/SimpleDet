"""Expanded documentation audit for SimpleDet static docs."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


class DocsAuditParser(HTMLParser):
    def __init__(self, path: Path):
        super().__init__(convert_charrefs=True)
        self.path = path
        self.local_targets: list[tuple[int, str, str]] = []
        self.media_targets: list[tuple[int, str, str]] = []
        self.local_media: list[tuple[int, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_dict = {name.lower(): value for name, value in attrs}

        if tag == "a" and "href" in attrs_dict and attrs_dict["href"]:
            self.local_targets.append((self.getpos()[0], "href", attrs_dict["href"]))

        if tag == "link" and "href" in attrs_dict and attrs_dict["href"]:
            self.local_targets.append((self.getpos()[0], "href", attrs_dict["href"]))

        if tag in {"script", "img", "source", "iframe"} and attrs_dict.get("src"):
            self.media_targets.append((self.getpos()[0], tag, attrs_dict["src"]))

        if tag == "img":
            if "alt" not in attrs_dict or not attrs_dict["alt"].strip():
                self.local_media.append((self.getpos()[0], "img"))


def _is_external(target: str) -> bool:
    if not target or target.startswith("#"):
        return True
    if target.startswith(("mailto:", "tel:", "javascript:")):
        return True
    parsed = urlparse(target)
    return bool(parsed.scheme or parsed.netloc)


def _iter_local_target_variants(target: str) -> list[Path]:
    target = target.split("#", 1)[0].split("?", 1)[0]
    if not target or target.startswith("/"):
        return []
    if target.endswith("/"):
        target = f"{target}index.html"
    while target.startswith("./"):
        target = target[2:]
    while target.startswith("../"):
        target = target[3:]
    return [Path("docs") / target]


def _collect_issues(docs_dir: Path) -> list[str]:
    issues: list[str] = []

    required_pages = {
        "api-reference",
        "cli-reference",
        "configuration-guide",
        "core-concepts",
        "datasets",
        "developer-guide",
        "diagrams",
        "evaluation",
        "examples",
        "experiments-reproducibility",
        "faq",
        "getting-started",
        "index",
        "installation",
        "inference",
        "model-coverage",
        "overview",
        "quickstart",
        "roadmap-changelog",
        "screenshots",
        "training",
        "diagrams",
        "troubleshooting",
    }

    missing_pages = [
        name + ".html" for name in sorted(required_pages) if not (docs_dir / f"{name}.html").exists()
    ]
    if missing_pages:
        issues.append(f"missing required page(s): {', '.join(missing_pages)}")

    for md_file in sorted(docs_dir.glob("*.md")):
        issues.append(f"unexpected markdown doc in static docs: {md_file.name}")

    for file in sorted(docs_dir.glob("*.html")):
        text = file.read_text(encoding="utf-8")
        parser = DocsAuditParser(file)
        parser.feed(text)

        for line, attr, target in parser.local_targets:
            if _is_external(target):
                continue
            candidates = _iter_local_target_variants(target)
            if not candidates or not any(candidate.exists() for candidate in candidates):
                issues.append(f"{file.name}:{line}: broken {attr} -> {target}")

        for line, tag, target in parser.media_targets:
            if _is_external(target):
                continue
            candidates = _iter_local_target_variants(target)
            if not candidates or not any(candidate.exists() for candidate in candidates):
                issues.append(f"{file.name}:{line}: missing local {tag} resource -> {target}")

        for line, _ in parser.local_media:
            issues.append(
                f"{file.name}:{line}: <img> missing non-empty alt attribute"
            )

    coverage_file = docs_dir / "model-coverage.html"
    if coverage_file.exists():
        coverage_text = coverage_file.read_text(encoding="utf-8")
        for token in (
            "VFNet",
            "FOVEA",
            "FoveaBox",
            "RepPoints",
            "YOLOF",
            "CenterNet",
            "Grid R-CNN",
            "Cascade R-CNN",
            "compatibility_alias",
            "planned, unsupported",
        ):
            if token not in coverage_text:
                issues.append(f"model-coverage.html: missing required coverage token {token}")

    return issues


def main() -> None:
    docs_dir = Path("docs")
    issues = _collect_issues(docs_dir)
    if issues:
        print("Docs audit failed:")
        for issue in issues:
            print(f"  - {issue}")
        raise SystemExit(1)
    print("Docs audit passed.")


if __name__ == "__main__":
    main()
