from __future__ import annotations

from html import unescape
from pathlib import Path
import re
import unittest


DOC_PATH = Path("docs/model-coverage.html")


def _doc_text() -> str:
    text = DOC_PATH.read_text(encoding="utf-8")
    text = re.sub(r"<[^>]+>", " ", text)
    return unescape(re.sub(r"\s+", " ", text))


class ModelCoverageDocsTests(unittest.TestCase):
    def test_key_detector_aliases_have_expected_status_boundaries(self):
        text = _doc_text()

        expected = {
            "VFNet": "compatibility_alias",
            "FOVEA / FoveaBox": "compatibility_alias",
            "RepPoints": "compatibility_alias",
            "YOLOF": "compatibility_alias",
            "CenterNet": "runtime_validated",
            "Grid R-CNN": "runtime_validated",
            "Cascade R-CNN": "runtime_validated",
        }
        for detector, status in expected.items():
            with self.subTest(detector=detector):
                pattern = rf"{re.escape(detector)}.*?{re.escape(status)}"
                self.assertRegex(text, pattern)

    def test_unvalidated_and_planned_entries_are_not_marked_supported(self):
        text = _doc_text()

        for head in ("RetinaNetHead", "FCOSV2Head", "YOLOHead", "SOLOV2Head"):
            with self.subTest(head=head):
                self.assertRegex(text, rf"{head}.*?unvalidated")

        self.assertIn("planned, unsupported transformer variants", text)
        self.assertIn("Rows marked compatibility_alias or unvalidated must not be presented as fully supported", text)


if __name__ == "__main__":
    unittest.main()
