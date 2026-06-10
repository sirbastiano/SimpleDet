import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_coco_simpledet.py"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_coco_simpledet", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_image(path: Path, content: bytes = b"fake-image") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _coco_payload(*, image_name: str, category_id: int, include_invalid: bool = False) -> dict:
    annotations = [
        {
            "id": 1,
            "image_id": 10,
            "category_id": category_id,
            "bbox": [1, 2, 3, 4],
            "area": 12,
            "iscrowd": 0,
        }
    ]
    if include_invalid:
        annotations.append(
            {
                "id": 2,
                "image_id": 10,
                "category_id": category_id,
                "bbox": [1, 2, 0, 4],
                "area": 0,
                "iscrowd": 0,
            }
        )
    return {
        "images": [{"id": 10, "file_name": image_name, "width": 8, "height": 8}],
        "annotations": annotations,
        "categories": [
            {"id": 5, "name": "alpha"},
            {"id": 9, "name": "beta"},
        ],
    }


class PrepareCocoSimpleDetTests(unittest.TestCase):
    def test_prepare_coco_rewrites_splits_categories_and_copy_reuse(self):
        module = _load_prepare_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "source"
            output = Path(tmpdir) / "prepared"
            _write_image(root / "train2017" / "train.jpg", b"train")
            _write_image(root / "val2017" / "shared.jpg", b"shared")
            _write_json(
                root / "annotations" / "instances_train2017.json",
                _coco_payload(image_name="train.jpg", category_id=9, include_invalid=True),
            )
            _write_json(
                root / "annotations" / "instances_val2017.json",
                _coco_payload(image_name="shared.jpg", category_id=5),
            )

            summary = module.prepare_coco(
                root,
                output,
                train_limit=None,
                val_limit=None,
                test_limit=None,
                image_mode="copy",
                seed=71,
            )

            train = json.loads((output / "annotations" / "train.json").read_text(encoding="utf-8"))
            val = json.loads((output / "annotations" / "val.json").read_text(encoding="utf-8"))
            test = json.loads((output / "annotations" / "test.json").read_text(encoding="utf-8"))
            copied_val_image = (output / "images" / "val2017" / "shared.jpg").read_bytes()

        self.assertEqual(summary["class_count"], 2)
        self.assertEqual(summary["classes"], ["alpha", "beta"])
        self.assertEqual(summary["splits"]["train"]["skipped_annotations"], 1)
        self.assertEqual(train["categories"], [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}])
        self.assertEqual(train["annotations"][0]["category_id"], 2)
        self.assertEqual(train["images"][0]["file_name"], "train2017/train.jpg")
        self.assertEqual(val["images"][0]["file_name"], "val2017/shared.jpg")
        self.assertEqual(test["images"][0]["file_name"], "val2017/shared.jpg")
        self.assertEqual(copied_val_image, b"shared")


if __name__ == "__main__":
    unittest.main()
