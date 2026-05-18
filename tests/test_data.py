import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simpledet.detectors.data import create_config, list_formats, load_dataset, _detect_dataset_format


def _write_dummy_png(path: Path, width: int = 10, height: int = 10) -> None:
    png_signature = b"\x89PNG\r\n\x1a\n"
    ihdr = b"IHDR"
    header = (
        png_signature
        + struct.pack(">I", 13)
        + ihdr
        + struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        + struct.pack(">I", 0)
    )
    path.write_bytes(header)


class TestDataAdapters(unittest.TestCase):
    def test_list_formats_contains_expected_adapters(self):
        formats = list_formats()
        self.assertIn("coco", formats)
        self.assertIn("json", formats)
        self.assertIn("csv", formats)
        self.assertIn("yolo", formats)
        self.assertIn("voc", formats)

    def test_detect_dataset_format_with_explicit_csv_works(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "annotations.csv").write_text("path,xmin,ymin,xmax,ymax,label,split\n")
            self.assertEqual(_detect_dataset_format(root, format_override=None), "csv")

    def test_load_dataset_with_csv_adapter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            _write_dummy_png(images / "sample.png")

            csv_payload = (
                "path,xmin,ymin,xmax,ymax,label,split\n"
                "images/sample.png,0,0,10,10,vessel,train\n"
            )
            (root / "annotations.csv").write_text(csv_payload)

            payload = load_dataset(str(root), format="csv")

            self.assertEqual(payload["format"], "csv")
            self.assertEqual(payload["path"], str(root))
            self.assertEqual(payload["images_dir"], str(images))
            self.assertEqual(len(payload["samples"]), 1)
            sample = payload["samples"][0]
            self.assertTrue(sample["annotations"])
            self.assertEqual(sample["annotations"][0]["category_id"], 0)

    def test_load_dataset_invalid_path_raises_file_error(self):
        missing = "/this/path/does/not/exist/really"
        with self.assertRaises(FileNotFoundError) as context:
            load_dataset(missing, format="coco")
        self.assertIn("not found", str(context.exception))

    def test_legacy_python_config_paths_are_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.py"
            path.write_text("model = dict(type='RetinaNet')\n", encoding="utf-8")

            with self.assertRaises(ValueError) as context:
                create_config(str(path))

        message = str(context.exception)
        self.assertIn("Legacy MMDetection .py config import/conversion is unsupported", message)
        self.assertIn("specific converter", message)
