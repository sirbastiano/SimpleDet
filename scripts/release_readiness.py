"""Final release-readiness audit for the SimpleDet package."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
import sys

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


MIN_CLAIMED_DETECTORS = 31
MIN_CLAIMED_HEADS = 31
DETECTOR_CLAIM_STATUSES = {"runtime_validated", "compatibility_alias"}
HEAD_CLAIM_STATUSES = {"runtime_validated"}
REQUIRED_MAJOR_ALIASES = {
    "VFNet",
    "FOVEA",
    "FoveaBox",
    "RepPoints",
    "YOLOF",
    "CenterNet",
    "Grid R-CNN",
    "Cascade R-CNN",
}
EDITABLE_DEV_COMMAND = "python -m pip install -e '.[cpu,timm,dev]'"


class ReleaseReadinessError(AssertionError):
    """Raised when release-readiness checks fail."""


@dataclass(frozen=True, slots=True)
class CoverageRow:
    name: str
    aliases: tuple[str, ...]
    validation_status: str
    required_extra_text: str


@dataclass(frozen=True, slots=True)
class ReleaseReadinessReport:
    detector_claims: int
    head_claims: int
    public_detector_aliases: int
    public_head_aliases: int
    optional_extras: tuple[str, ...]

    def lines(self) -> tuple[str, ...]:
        extras = ", ".join(self.optional_extras)
        return (
            "Release readiness passed.",
            f"Detector claims: {self.detector_claims}",
            f"Head claims: {self.head_claims}",
            f"Public detector aliases: {self.public_detector_aliases}",
            f"Public head aliases: {self.public_head_aliases}",
            f"Optional extras: {extras}",
        )


class _ModelCoverageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.sections: dict[str, list[CoverageRow]] = {
            "Detector coverage": [],
            "Head coverage": [],
        }
        self._active_section: str | None = None
        self._heading_parts: list[str] | None = None
        self._row_cells: list[str] | None = None
        self._cell_parts: list[str] | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag == "h2":
            self._heading_parts = []
            return
        if tag == "tr" and self._active_section in self.sections:
            self._row_cells = []
            return
        if tag in {"td", "th"} and self._row_cells is not None:
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._heading_parts is not None:
            self._heading_parts.append(data)
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2" and self._heading_parts is not None:
            heading = _normalize_text("".join(self._heading_parts))
            self._active_section = heading if heading in self.sections else None
            self._heading_parts = None
            return

        if tag in {"td", "th"} and self._cell_parts is not None:
            assert self._row_cells is not None
            self._row_cells.append(_normalize_text("".join(self._cell_parts)))
            self._cell_parts = None
            return

        if tag == "tr" and self._row_cells is not None:
            self._record_row(self._row_cells)
            self._row_cells = None

    def _record_row(self, cells: list[str]) -> None:
        if self._active_section not in self.sections or len(cells) < 5:
            return
        if cells[0] in {"Detector", "Head"}:
            return
        self.sections[self._active_section].append(
            CoverageRow(
                name=cells[0],
                aliases=_split_alias_cell(cells[2]),
                validation_status=cells[3],
                required_extra_text=cells[4],
            )
        )


def validate_release_readiness(repo_root: Path | str = ".") -> ReleaseReadinessReport:
    root = Path(repo_root)
    issues: list[str] = []

    project = _read_project_metadata(root / "pyproject.toml")
    optional_extras = tuple(sorted(project["optional-dependencies"]))
    docs = {
        "README.md": (root / "README.md").read_text(encoding="utf-8"),
        "docs/installation.html": (root / "docs" / "installation.html").read_text(
            encoding="utf-8"
        ),
    }
    coverage = read_model_coverage(root / "docs" / "model-coverage.html")
    detector_rows = coverage["Detector coverage"]
    head_rows = coverage["Head coverage"]

    issues.extend(
        _claim_count_issues(
            "detector", detector_rows, DETECTOR_CLAIM_STATUSES, MIN_CLAIMED_DETECTORS
        )
    )
    issues.extend(
        _claim_count_issues("head", head_rows, HEAD_CLAIM_STATUSES, MIN_CLAIMED_HEADS)
    )
    issues.extend(_documented_extra_issues(docs, optional_extras))
    issues.extend(_coverage_extra_issues(detector_rows + head_rows, optional_extras))
    issues.extend(_required_alias_issues(detector_rows + head_rows))

    public_detector_aliases, public_head_aliases = _public_discovery_counts()
    if public_detector_aliases < MIN_CLAIMED_DETECTORS:
        issues.append(
            "public detector aliases below release minimum: "
            f"{public_detector_aliases} < {MIN_CLAIMED_DETECTORS}"
        )
    if public_head_aliases < MIN_CLAIMED_HEADS:
        issues.append(
            "public head aliases below release minimum: "
            f"{public_head_aliases} < {MIN_CLAIMED_HEADS}"
        )

    if issues:
        raise ReleaseReadinessError(
            "Release readiness failed:\n" + "\n".join(f"- {issue}" for issue in issues)
        )

    return ReleaseReadinessReport(
        detector_claims=_claim_count(detector_rows, DETECTOR_CLAIM_STATUSES),
        head_claims=_claim_count(head_rows, HEAD_CLAIM_STATUSES),
        public_detector_aliases=public_detector_aliases,
        public_head_aliases=public_head_aliases,
        optional_extras=optional_extras,
    )


def read_model_coverage(path: Path) -> dict[str, list[CoverageRow]]:
    parser = _ModelCoverageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    return parser.sections


def _read_project_metadata(path: Path) -> dict[str, object]:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def _claim_count(rows: list[CoverageRow], accepted_statuses: set[str]) -> int:
    return sum(1 for row in rows if row.validation_status in accepted_statuses)


def _claim_count_issues(
    label: str,
    rows: list[CoverageRow],
    accepted_statuses: set[str],
    minimum: int,
) -> list[str]:
    count = _claim_count(rows, accepted_statuses)
    if count >= minimum:
        return []
    return [f"claimed {label} count below release minimum: {count} < {minimum}"]


def _documented_extra_issues(
    docs: dict[str, str],
    optional_extras: tuple[str, ...],
) -> list[str]:
    issues: list[str] = []
    expected = set(optional_extras)
    for path, text in docs.items():
        documented = _documented_extras(text)
        missing = sorted(expected - documented)
        unexpected = sorted(documented - expected)
        if missing:
            issues.append(
                f"{path} missing install command(s) for extra(s): {', '.join(missing)}"
            )
        if unexpected:
            issues.append(
                f"{path} documents unknown package extra(s): {', '.join(unexpected)}"
            )
        if "python -m pip install simpledet" not in text:
            issues.append(f"{path} missing base install command")
        if EDITABLE_DEV_COMMAND not in text:
            issues.append(f"{path} missing editable dev install command")
    return issues


def _coverage_extra_issues(
    rows: list[CoverageRow],
    optional_extras: tuple[str, ...],
) -> list[str]:
    known = set(optional_extras)
    unknown: set[str] = set()
    for row in rows:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", row.required_extra_text):
            if token in {"default", "backbone", "uses"}:
                continue
            if token != "-" and token not in known:
                unknown.add(token)
    if unknown:
        return [
            "model coverage documents unknown required extra(s): "
            + ", ".join(sorted(unknown))
        ]
    return []


def _required_alias_issues(rows: list[CoverageRow]) -> list[str]:
    documented = {row.name for row in rows}
    for row in rows:
        documented.update(row.aliases)
    missing = sorted(REQUIRED_MAJOR_ALIASES - documented)
    if missing:
        return [
            "model coverage missing required release alias(es): " + ", ".join(missing)
        ]
    return []


def _public_discovery_counts() -> tuple[int, int]:
    from simpledet.discovery import detector_rows, head_rows

    detectors = detector_rows()
    heads = [
        row
        for row in head_rows()
        if row.validation_status in HEAD_CLAIM_STATUSES
    ]
    return len(detectors), len(heads)


def _documented_extras(text: str) -> set[str]:
    extras: set[str] = set()
    for group in re.findall(r"simpledet\[([^\]]+)\]", text):
        extras.update(part.strip() for part in group.split(",") if part.strip())
    return extras


def _split_alias_cell(text: str) -> tuple[str, ...]:
    if text == "-":
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    try:
        report = validate_release_readiness(Path.cwd())
    except ReleaseReadinessError as exc:
        print(exc, file=sys.stderr)
        return 1
    for line in report.lines():
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
