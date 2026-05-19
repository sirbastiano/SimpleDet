import json
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


def _write_voc_xml(
    path: Path,
    *,
    filename: str,
    width: int = 10,
    height: int = 10,
    label: str = "vessel",
    xmin: int = 1,
    ymin: int = 2,
    xmax: int = 6,
    ymax: int = 8,
    with_object: bool = True,
) -> None:
    object_xml = ""
    if with_object:
        object_xml = f"""
  <object>
    <name>{label}</name>
    <pose>Unspecified</pose>
    <difficult>0</difficult>
    <bndbox>
      <xmin>{xmin}</xmin>
      <ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax>
      <ymax>{ymax}</ymax>
    </bndbox>
  </object>"""
    path.write_text(
        f"""<annotation>
  <filename>{filename}</filename>
  <size>
    <width>{width}</width>
    <height>{height}</height>
    <depth>3</depth>
  </size>{object_xml}
</annotation>""",
        encoding="utf-8",
    )


class TestDataAdapters(unittest.TestCase):
    def assertNormalizedPayload(self, payload, *, format_key: str, split: str = "train"):
        self.assertEqual(payload["format"], format_key)
        for key in (
            "images",
            "annotations",
            "samples",
            "categories",
            "category_map",
            "splits",
            "meta",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["meta"]["num_images"], len(payload["images"]))
        self.assertEqual(payload["meta"]["num_samples"], len(payload["samples"]))
        self.assertEqual(payload["meta"]["num_annotations"], len(payload["annotations"]))
        self.assertIn(split, payload["splits"])
        self.assertEqual(payload["images"][0]["split"], split)
        self.assertEqual(payload["samples"][0]["split"], split)
        self.assertEqual(payload["annotations"][0]["split"], split)
        bbox = payload["annotations"][0]["bbox"]
        self.assertEqual(set(bbox), {"x_min", "y_min", "x_max", "y_max"})
        self.assertLess(bbox["x_min"], bbox["x_max"])
        self.assertLess(bbox["y_min"], bbox["y_max"])

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

    def test_load_dataset_with_coco_adapter_normalizes_records(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            annotations = root / "annotations"
            images.mkdir()
            annotations.mkdir()
            _write_dummy_png(images / "sample.png", width=20, height=10)
            annotation_file = annotations / "instances_train.json"
            annotation_file.write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 7, "file_name": "sample.png", "width": 20, "height": 10}
                        ],
                        "annotations": [
                            {
                                "id": 9,
                                "image_id": 7,
                                "category_id": 3,
                                "bbox": [1, 2, 5, 6],
                                "iscrowd": 0,
                            }
                        ],
                        "categories": [{"id": 3, "name": "vessel"}],
                    }
                ),
                encoding="utf-8",
            )

            payload = load_dataset(str(root), format="coco")

            self.assertNormalizedPayload(payload, format_key="coco", split="train")
            self.assertEqual(payload["annotation_file"], str(annotation_file))
            self.assertEqual(payload["category_map"], {3: "vessel"})
            self.assertEqual(
                payload["annotations"][0]["bbox"],
                {"x_min": 1, "y_min": 2, "x_max": 6, "y_max": 8},
            )

    def test_coco_missing_annotations_file_raises_exact_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / "annotations" / "instances_train.json"

            with self.assertRaises(FileNotFoundError) as context:
                load_dataset(str(root), format="coco")

            self.assertEqual(str(context.exception), f"Missing COCO annotation file: {expected}")

    def test_coco_malformed_schema_raises_value_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "annotations"
            annotations.mkdir()
            (annotations / "instances_train.json").write_text(
                json.dumps({"images": [], "annotations": []}),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as context:
                load_dataset(str(root), format="coco")

            self.assertIn("Invalid COCO schema", str(context.exception))

    def test_coco_rejects_image_path_traversal(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            annotations = root / "annotations"
            images.mkdir()
            annotations.mkdir()
            _write_dummy_png(root / "outside.png")
            (annotations / "instances_train.json").write_text(
                json.dumps(
                    {
                        "images": [
                            {"id": 1, "file_name": "../outside.png", "width": 10, "height": 10}
                        ],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 1, 2, 2]}
                        ],
                        "categories": [{"id": 0, "name": "vessel"}],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError) as context:
                load_dataset(str(root), format="coco")

            self.assertIn("Missing COCO image file", str(context.exception))

    def test_load_dataset_with_csv_adapter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            _write_dummy_png(images / "sample.png")

            csv_payload = (
                "path,xmin,ymin,xmax,ymax,label,split\n"
                "sample.png,0,0,10,10,vessel,train\n"
            )
            (root / "annotations.csv").write_text(csv_payload)

            payload = load_dataset(str(root), format="csv")

            self.assertNormalizedPayload(payload, format_key="csv")
            self.assertEqual(payload["path"], str(root))
            self.assertEqual(payload["images_dir"], str(images))
            self.assertEqual(payload["category_map"], {0: "vessel"})

    def test_csv_malformed_bbox_raises_value_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            _write_dummy_png(images / "sample.png")
            (root / "annotations.csv").write_text(
                "path,xmin,ymin,xmax,ymax,label,split\n"
                "sample.png,10,0,0,10,vessel,train\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as context:
                load_dataset(str(root), format="csv")

            self.assertIn("xmin must be less than xmax", str(context.exception))

    def test_load_dataset_with_direct_csv_file_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            _write_dummy_png(images / "sample.png")
            annotation_file = root / "custom.csv"
            annotation_file.write_text(
                "path,xmin,ymin,xmax,ymax,label,split\n"
                "sample.png,0,0,10,10,vessel,test\n",
                encoding="utf-8",
            )

            payload = load_dataset(str(annotation_file), format="csv")

            self.assertNormalizedPayload(payload, format_key="csv", split="test")
            self.assertEqual(payload["annotation_file"], str(annotation_file))

    def test_load_dataset_with_simple_json_adapter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            _write_dummy_png(images / "sample.png")
            (root / "annotations.json").write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "path": "sample.png",
                                "xmin": 1,
                                "ymin": 2,
                                "xmax": 6,
                                "ymax": 7,
                                "label": "vessel",
                                "split": "val",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            payload = load_dataset(str(root), format="json")

            self.assertNormalizedPayload(payload, format_key="json", split="val")
            self.assertEqual(payload["category_map"], {0: "vessel"})

    def test_json_malformed_row_raises_value_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            images.mkdir()
            _write_dummy_png(images / "sample.png")
            (root / "annotations.json").write_text(
                json.dumps([{"path": "sample.png", "xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}]),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError) as context:
                load_dataset(str(root), format="json")

            self.assertIn("missing required columns", str(context.exception))

    def test_load_dataset_with_yolo_adapter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images" / "val"
            labels = root / "labels" / "val"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            _write_dummy_png(images / "sample.png", width=20, height=10)
            (labels / "sample.txt").write_text("1 0.5 0.5 0.4 0.2\n", encoding="utf-8")

            payload = load_dataset(str(root), format="yolo")

            self.assertNormalizedPayload(payload, format_key="yolo", split="val")
            self.assertEqual(payload["category_map"], {1: "1"})
            self.assertEqual(
                payload["annotations"][0]["bbox"],
                {"x_min": 6.0, "y_min": 4.0, "x_max": 14.0, "y_max": 6.0},
            )

    def test_yolo_malformed_label_raises_value_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            _write_dummy_png(images / "sample.png", width=20, height=10)
            (labels / "sample.txt").write_text("0 0.5 0.5 2 0.2\n", encoding="utf-8")

            with self.assertRaises(ValueError) as context:
                load_dataset(str(root), format="yolo")

            self.assertIn("Out-of-range normalized values", str(context.exception))

    def test_load_dataset_with_voc_adapter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "JPEGImages"
            annotations = root / "Annotations"
            splits = root / "ImageSets" / "Main"
            images.mkdir()
            annotations.mkdir()
            splits.mkdir(parents=True)
            _write_dummy_png(images / "sample.png", width=10, height=10)
            _write_voc_xml(annotations / "sample.xml", filename="sample.png")
            (splits / "val.txt").write_text("sample\n", encoding="utf-8")

            payload = load_dataset(str(root), format="voc")

            self.assertNormalizedPayload(payload, format_key="voc", split="val")
            self.assertEqual(payload["category_map"], {0: "vessel"})
            self.assertEqual(
                payload["annotations"][0]["bbox"],
                {"x_min": 1.0, "y_min": 2.0, "x_max": 6.0, "y_max": 8.0},
            )

    def test_voc_malformed_annotation_raises_value_error(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root / "JPEGImages"
            annotations = root / "Annotations"
            images.mkdir()
            annotations.mkdir()
            _write_dummy_png(images / "sample.png", width=10, height=10)
            _write_voc_xml(annotations / "sample.xml", filename="sample.png", with_object=False)

            with self.assertRaises(ValueError) as context:
                load_dataset(str(root), format="voc")

            self.assertIn("must include at least one <object>", str(context.exception))

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
