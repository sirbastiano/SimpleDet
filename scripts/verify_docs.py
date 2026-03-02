"""Validation utilities for repository documentation pages."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse
import re


def main() -> None:
    docs_dir = Path("docs")
    required_pages = {
        "index",
        "getting-started",
        "core-concepts",
        "installation",
        "screenshots",
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

    print("Docs verification passed.")


if __name__ == "__main__":
    main()
