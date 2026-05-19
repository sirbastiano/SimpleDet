import json
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _write_dummy_png(path: Path, width: int = 10, height: int = 8) -> None:
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


class _FakeNoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class _FakeTensor:
    def __init__(self, data=None, *, shape=None, max_value=1.0):
        self.data = data
        self._shape = tuple(shape) if shape is not None else self._infer_shape(data)
        self._max_value = float(max_value)
        self.dtype = None
        self.device = None

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def to(self, dtype=None, device=None):
        result = _FakeTensor(self.data, shape=self.shape, max_value=self._max_value)
        result.dtype = dtype if dtype is not None else self.dtype
        result.device = device if device is not None else self.device
        return result

    def max(self):
        return self._max_value

    def repeat(self, channels, *_args):
        return _FakeTensor(self.data, shape=(channels, *self.shape[1:]), max_value=self._max_value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.data

    def __truediv__(self, value):
        return _FakeTensor(self.data, shape=self.shape, max_value=self._max_value / float(value))

    def _infer_shape(self, value):
        if isinstance(value, list):
            if value and isinstance(value[0], list):
                if value[0] and isinstance(value[0][0], list):
                    return (len(value), len(value[0]), len(value[0][0]))
                return (len(value), len(value[0]))
            return (len(value),)
        return ()


class _TinyPredictModel:
    class_names = ("wake", "ship")

    def __init__(self):
        self.eval_called = False
        self.seen_shapes = []

    def eval(self):
        self.eval_called = True
        return self

    def predict(self, images):
        self.seen_shapes = [tuple(image.shape) for image in images]
        return [
            {
                "boxes": _FakeTensor([[0, 1, 4, 5], [2, 2, 3, 3]]),
                "scores": _FakeTensor([0.25, 0.95]),
                "labels": _FakeTensor([1, 2]),
            }
            for _image in images
        ]


class _CallableClassIdModel:
    def __init__(self):
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, images):
        return [
            {
                "boxes": [[1, 2, 6, 8]],
                "scores": [0.7],
                "class_ids": [1],
            }
            for _image in images
        ]


def _fake_runtime_modules(*, fail_decode: bool = False):
    fake_torch = types.ModuleType("torch")
    fake_torch.float32 = "float32"
    fake_torch.no_grad = lambda: _FakeNoGrad()

    fake_torchvision = types.ModuleType("torchvision")
    fake_io = types.ModuleType("torchvision.io")

    def _read_image(path):
        if fail_decode:
            raise ValueError(f"decode failed: {path}")
        return _FakeTensor(shape=(3, 8, 10), max_value=255.0)

    fake_io.read_image = _read_image
    fake_torchvision.io = fake_io
    return {
        "torch": fake_torch,
        "torchvision": fake_torchvision,
        "torchvision.io": fake_io,
    }


class InferenceApiTests(unittest.TestCase):
    def test_predict_image_returns_structured_payload_with_metadata(self):
        from simpledet.detectors.infer import predict_image

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "sample.png"
            _write_dummy_png(image_path)
            model = _TinyPredictModel()

            with patch.dict(sys.modules, _fake_runtime_modules()):
                payload = predict_image(model, image_path, score_threshold=0.5)

        self.assertTrue(model.eval_called)
        self.assertEqual(model.seen_shapes, [(3, 8, 10)])
        self.assertEqual(payload["boxes"], [[2.0, 2.0, 3.0, 3.0]])
        self.assertEqual(payload["scores"], [0.95])
        self.assertEqual(payload["labels"], [2])
        self.assertEqual(payload["class_names"], ["ship"])
        self.assertEqual(payload["metadata"]["path"], str(image_path))
        self.assertEqual(payload["metadata"]["file_name"], "sample.png")
        self.assertEqual(payload["metadata"]["height"], 8)
        self.assertEqual(payload["metadata"]["width"], 10)
        self.assertEqual(payload["metadata"]["channels"], 3)

    def test_predict_batch_preserves_order_and_accepts_class_id_outputs(self):
        from simpledet.detectors.infer import predict_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image_paths = [root / "a.png", root / "b.png"]
            for path in image_paths:
                _write_dummy_png(path)
            model = _CallableClassIdModel()

            with patch.dict(sys.modules, _fake_runtime_modules()):
                payloads = predict_batch(
                    model,
                    image_paths,
                    class_names=("wake",),
                    metadata=[{"image_id": 11}, {"image_id": 12}],
                )

        self.assertTrue(model.eval_called)
        self.assertEqual([item["metadata"]["path"] for item in payloads], [str(path) for path in image_paths])
        self.assertEqual([item["metadata"]["image_id"] for item in payloads], [11, 12])
        self.assertEqual(payloads[0]["labels"], [1])
        self.assertEqual(payloads[0]["class_names"], ["wake"])

    def test_missing_image_path_raises_file_not_found_with_path(self):
        from simpledet.detectors.infer import predict_image

        missing = Path("/tmp/simpledet-missing-image.png")
        with self.assertRaises(FileNotFoundError) as context:
            predict_image(_TinyPredictModel(), missing)

        self.assertIn(str(missing), str(context.exception))

    def test_decode_failure_raises_image_loading_error_with_path(self):
        from simpledet.detectors.infer import ImageLoadingError, predict_image

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "corrupt.png"
            image_path.write_text("not an image", encoding="utf-8")
            with patch.dict(sys.modules, _fake_runtime_modules(fail_decode=True)):
                with self.assertRaises(ImageLoadingError) as context:
                    predict_image(_TinyPredictModel(), image_path)

        self.assertIn(str(image_path), str(context.exception))

    def test_predict_batch_is_fail_fast_for_missing_paths(self):
        from simpledet.detectors.infer import predict_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "a.png"
            missing = Path(tmpdir) / "missing.png"
            _write_dummy_png(image_path)
            model = _TinyPredictModel()
            with patch.dict(sys.modules, _fake_runtime_modules()):
                with self.assertRaises(FileNotFoundError) as context:
                    predict_batch(model, [image_path, missing])

        self.assertIn(str(missing), str(context.exception))
        self.assertEqual(model.seen_shapes, [])

    def test_export_predictions_returns_and_writes_json_payload(self):
        from simpledet.detectors.infer import export_predictions

        prediction = {
            "boxes": [[1, 2, 6, 8]],
            "scores": [0.75],
            "labels": [1],
            "class_names": ["wake"],
            "metadata": {"image_id": 7, "path": "sample.png"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "predictions.json"
            payload = export_predictions([prediction], output_path)
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["format"], "simpledet_predictions")
        self.assertEqual(payload["version"], 1)
        self.assertEqual(payload, written)
        self.assertEqual(payload["predictions"][0]["metadata"]["image_id"], 7)

    def test_load_checkpoint_for_inference_delegates_to_lightweight_loader(self):
        from simpledet.detectors import infer

        sentinel = object()
        with patch.object(infer, "load_model", return_value=sentinel) as load_model:
            result = infer.load_checkpoint_for_inference(
                Path("model.pth"),
                device="cpu",
                model_name="retinanet_resnet50_fpn",
                num_classes=3,
                score_threshold=0.2,
                max_detections=5,
                class_names=("wake", "ship"),
            )

        self.assertIs(result, sentinel)
        load_model.assert_called_once_with(
            "model.pth",
            device="cpu",
            model_name="retinanet_resnet50_fpn",
            num_classes=3,
            score_threshold=0.2,
            max_detections=5,
            class_names=("wake", "ship"),
        )

    def test_top_level_exports_are_lazy_and_available(self):
        import simpledet
        from simpledet import api
        from simpledet.detectors.infer import (
            export_predictions,
            load_checkpoint_for_inference,
            predict_batch,
            predict_image,
        )

        self.assertIs(simpledet.predict_image, predict_image)
        self.assertIs(simpledet.predict_batch, predict_batch)
        self.assertIs(simpledet.load_checkpoint_for_inference, load_checkpoint_for_inference)
        self.assertIs(simpledet.export_predictions, export_predictions)
        self.assertEqual(api.predict_image.__name__, "predict_image")
        self.assertEqual(api.predict_batch.__name__, "predict_batch")
        self.assertEqual(api.load_checkpoint_for_inference.__name__, "load_checkpoint_for_inference")
        self.assertEqual(api.export_predictions.__name__, "export_predictions")


if __name__ == "__main__":
    unittest.main()
