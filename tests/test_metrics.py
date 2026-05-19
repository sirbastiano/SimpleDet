import json
import math
import tempfile
import unittest
from pathlib import Path

from simpledet.metrics import build_coco_prediction_export, evaluate_coco_bbox_metrics


class CocoBBoxMetricTests(unittest.TestCase):
    def _write_annotations(
        self,
        root: Path,
        *,
        annotations,
        categories=None,
        images=None,
    ) -> Path:
        path = root / "instances_test.json"
        payload = {
            "images": images
            or [{"id": 1, "file_name": "a.png", "width": 32, "height": 32}],
            "annotations": annotations,
            "categories": categories or [{"id": 1, "name": "wake"}],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_perfect_prediction_scores_above_mismatched_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = self._write_annotations(
                Path(tmpdir),
                annotations=[
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    }
                ],
            )
            perfect = evaluate_coco_bbox_metrics(
                annotation_path,
                [{"image_id": 1, "boxes": [[0, 0, 10, 10]], "scores": [0.99], "labels": [1]}],
            )
            mismatched = evaluate_coco_bbox_metrics(
                annotation_path,
                [
                    {
                        "image_id": 1,
                        "boxes": [[20, 20, 30, 30]],
                        "scores": [0.99],
                        "labels": [1],
                    }
                ],
            )

        self.assertGreater(perfect["summary"]["map"], mismatched["summary"]["map"])
        self.assertGreater(perfect["summary"]["map_50"], mismatched["summary"]["map_50"])
        self.assertAlmostEqual(perfect["summary"]["map"], 1.0)
        self.assertAlmostEqual(perfect["per_class"][0]["ap"], 1.0)
        self.assertAlmostEqual(perfect["recall"]["mean"], 1.0)

    def test_partial_prediction_reports_lower_recall(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = self._write_annotations(
                Path(tmpdir),
                annotations=[
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                    {"id": 2, "image_id": 1, "category_id": 1, "bbox": [20, 20, 5, 5]},
                ],
            )
            metrics = evaluate_coco_bbox_metrics(
                annotation_path,
                [
                    {
                        "image_id": 1,
                        "boxes": [[0, 0, 10, 10], [12, 12, 18, 18]],
                        "scores": [0.99, 0.8],
                        "labels": [1, 1],
                    }
                ],
            )

        self.assertAlmostEqual(metrics["summary"]["recall_50"], 0.5)
        self.assertAlmostEqual(metrics["per_class"][0]["recall_50"], 0.5)
        self.assertGreater(metrics["summary"]["map_50"], 0.0)
        self.assertLess(metrics["summary"]["map_50"], 1.0)

    def test_empty_predictions_return_zero_metrics_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = self._write_annotations(
                Path(tmpdir),
                annotations=[
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}
                ],
            )
            metrics = evaluate_coco_bbox_metrics(
                annotation_path,
                [{"image_id": 1, "boxes": [], "scores": [], "labels": []}],
            )

        self.assertEqual(metrics["prediction_export"], [])
        self.assertEqual(metrics["summary"]["num_predictions"], 0)
        self.assertEqual(metrics["summary"]["map"], 0.0)
        self.assertEqual(metrics["summary"]["mean_recall"], 0.0)
        self.assertTrue(math.isfinite(metrics["summary"]["map"]))

    def test_empty_targets_and_predictions_return_zero_summaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = self._write_annotations(Path(tmpdir), annotations=[])
            metrics = evaluate_coco_bbox_metrics(
                annotation_path,
                [{"image_id": 1, "boxes": [], "scores": [], "labels": []}],
            )

        self.assertEqual(metrics["summary"]["num_ground_truth"], 0)
        self.assertEqual(metrics["summary"]["num_predictions"], 0)
        self.assertEqual(metrics["summary"]["map"], 0.0)
        self.assertEqual(metrics["recall"]["per_class"], [])

    def test_class_mismatched_prediction_does_not_match_perfect_iou(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = self._write_annotations(
                Path(tmpdir),
                annotations=[
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}
                ],
                categories=[{"id": 1, "name": "wake"}, {"id": 2, "name": "ship"}],
            )
            metrics = evaluate_coco_bbox_metrics(
                annotation_path,
                [{"image_id": 1, "boxes": [[0, 0, 10, 10]], "scores": [0.99], "labels": [2]}],
            )

        self.assertEqual(metrics["summary"]["map"], 0.0)
        self.assertEqual(metrics["summary"]["mean_recall"], 0.0)
        self.assertEqual(metrics["per_class"][0]["num_ground_truth"], 1)
        self.assertEqual(metrics["per_class"][0]["num_predictions"], 0)
        self.assertEqual(metrics["per_class"][1]["num_ground_truth"], 0)
        self.assertEqual(metrics["per_class"][1]["num_predictions"], 1)

    def test_prediction_export_converts_xyxy_and_maps_foreground_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            annotation_path = self._write_annotations(
                Path(tmpdir),
                annotations=[],
                categories=[{"id": 3, "name": "wake"}],
            )
            export = build_coco_prediction_export(
                annotation_path,
                [{"image_id": 7, "boxes": [[1, 2, 6, 8]], "scores": [0.75], "labels": [1]}],
            )

        self.assertEqual(
            export,
            [{"image_id": 7, "category_id": 3, "bbox": [1.0, 2.0, 5.0, 6.0], "score": 0.75}],
        )


if __name__ == "__main__":
    unittest.main()
