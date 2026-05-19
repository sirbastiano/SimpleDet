import unittest
from unittest.mock import patch

from scripts.release_readiness import (
    DETECTOR_CLAIM_STATUSES,
    HEAD_CLAIM_STATUSES,
    MIN_CLAIMED_DETECTORS,
    MIN_CLAIMED_HEADS,
    CoverageRow,
    ReleaseReadinessError,
    _claim_count_issues,
    _documented_extra_issues,
    validate_release_readiness,
)


def _rows(count, *, status):
    return [
        CoverageRow(
            name=f"row-{index}",
            aliases=(),
            validation_status=status,
            required_extra_text="cpu",
        )
        for index in range(count)
    ]


class ReleaseReadinessTests(unittest.TestCase):
    def test_release_readiness_gate_passes_current_repo(self):
        report = validate_release_readiness()

        self.assertGreaterEqual(report.detector_claims, MIN_CLAIMED_DETECTORS)
        self.assertGreaterEqual(report.head_claims, MIN_CLAIMED_HEADS)
        self.assertIn("dev", report.optional_extras)

    def test_release_gate_fails_when_claimed_detector_count_is_low(self):
        issues = _claim_count_issues(
            "detector",
            _rows(MIN_CLAIMED_DETECTORS - 1, status="runtime_validated"),
            DETECTOR_CLAIM_STATUSES,
            MIN_CLAIMED_DETECTORS,
        )

        self.assertEqual(len(issues), 1)
        self.assertIn("claimed detector count below release minimum", issues[0])

    def test_release_gate_blocks_handoff_when_claimed_count_is_low(self):
        coverage = {
            "Detector coverage": _rows(
                MIN_CLAIMED_DETECTORS - 1, status="runtime_validated"
            ),
            "Head coverage": _rows(MIN_CLAIMED_HEADS, status="runtime_validated"),
        }

        with patch(
            "scripts.release_readiness.read_model_coverage", return_value=coverage
        ), patch(
            "scripts.release_readiness._public_discovery_counts",
            return_value=(MIN_CLAIMED_DETECTORS, MIN_CLAIMED_HEADS),
        ):
            with self.assertRaisesRegex(
                ReleaseReadinessError, "claimed detector count below release minimum"
            ):
                validate_release_readiness()

    def test_release_gate_fails_when_claimed_head_count_is_low(self):
        issues = _claim_count_issues(
            "head",
            _rows(MIN_CLAIMED_HEADS - 1, status="runtime_validated"),
            HEAD_CLAIM_STATUSES,
            MIN_CLAIMED_HEADS,
        )

        self.assertEqual(len(issues), 1)
        self.assertIn("claimed head count below release minimum", issues[0])

    def test_release_gate_fails_when_docs_omit_package_extra(self):
        docs = {
            "README.md": (
                "python -m pip install simpledet\n"
                "python -m pip install 'simpledet[cpu]'\n"
                "python -m pip install -e '.[cpu,timm,dev]'\n"
            )
        }

        issues = _documented_extra_issues(docs, ("cpu", "dev"))

        self.assertEqual(
            issues, ["README.md missing install command(s) for extra(s): dev"]
        )

    def test_release_error_message_lists_failures(self):
        with self.assertRaisesRegex(ReleaseReadinessError, "example failure"):
            raise ReleaseReadinessError("Release readiness failed:\n- example failure")


if __name__ == "__main__":
    unittest.main()
