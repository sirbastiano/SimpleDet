"""Validation utilities for repository documentation pages."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
import re


def main() -> None:
    docs_dir = Path("docs")
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
        name + ".html" for name in required_pages if not (docs_dir / f"{name}.html").exists()
    ]
    if missing_pages:
        raise SystemExit(
            f"Missing required docs pages: {', '.join(sorted(missing_pages))}"
        )

    markdown_pages = sorted(path.name for path in docs_dir.glob("*.md"))
    if markdown_pages:
        raise SystemExit(
            "Markdown docs should not be present in the static site: "
            + ", ".join(markdown_pages)
        )

    href_re = re.compile(r'''href\s*=\s*["']([^"']+)["']''', re.IGNORECASE)
    for file in docs_dir.glob("*.html"):
        text = file.read_text(encoding="utf-8")
        for target in href_re.findall(text):
            target = target.strip()
            if not target or target.startswith("#"):
                continue

            parsed = urlparse(target)
            if parsed.scheme or parsed.netloc:
                continue
            if target.startswith("/"):
                continue

            target = target.split("#", 1)[0]
            target_path = (docs_dir / target).resolve()
            if target.endswith("/"):
                target_path = (docs_dir / target / "index.html").resolve()
            if not target_path.exists():
                raise SystemExit(f"{file}: broken internal link -> {target}")

    coverage_text = (docs_dir / "model-coverage.html").read_text(encoding="utf-8")
    required_coverage_tokens = {
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
    }
    missing_tokens = sorted(
        token for token in required_coverage_tokens if token not in coverage_text
    )
    if missing_tokens:
        raise SystemExit(
            "Model coverage docs missing required alias/status tokens: "
            + ", ".join(missing_tokens)
        )

    print("Docs verification passed.")


if __name__ == "__main__":
    main()
