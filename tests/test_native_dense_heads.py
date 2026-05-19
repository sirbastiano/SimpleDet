import unittest

from native_tensor_contracts import (
    assert_dense_head_output_contract,
    make_cpu_detector_smoke_batch,
    require_torch,
)


_HEAD_CASES = (
    {
        "name": "RPNHead",
        "alias": "rpn_head",
        "kind": "rpn",
        "loss": "DenseRPNLoss",
        "decoder": "DenseRPNDecoder",
    },
    {
        "name": "RetinaHead",
        "alias": "retina_head",
        "kind": "retina",
        "loss": "DenseRetinaNetLoss",
        "decoder": "DenseRetinaNetDecoder",
    },
    {
        "name": "FCOSHead",
        "alias": "fcos_head",
        "kind": "anchor_free",
        "loss": "DenseFCOSLoss",
        "decoder": "DenseFCOSDecoder",
    },
    {
        "name": "ATSSHead",
        "alias": "atss_head",
        "kind": "anchor",
        "loss": "DenseATSSLoss",
        "decoder": "DenseATSSDecoder",
    },
    {
        "name": "FSAFHead",
        "alias": "fsaf_head",
        "kind": "anchor_free",
        "loss": "DenseFSAFLoss",
        "decoder": "DenseFSAFDecoder",
    },
    {
        "name": "FoveaHead",
        "alias": "fovea_head",
        "kind": "anchor_free",
        "loss": "DenseFoveaLoss",
        "decoder": "DenseFoveaDecoder",
    },
    {
        "name": "FreeAnchorRetinaHead",
        "alias": "free_anchor_head",
        "kind": "retina",
        "loss": "DenseFreeAnchorRetinaNetLoss",
        "decoder": "DenseFreeAnchorRetinaNetDecoder",
    },
)


_ADVANCED_HEAD_CASES = (
    {
        "name": "GFLHead",
        "alias": "gfl_head",
        "kind": "anchor",
        "loss": "DenseGFLLoss",
        "decoder": "DenseGFLDecoder",
        "loss_attrs": ("loss_cls", "loss_dfl"),
    },
    {
        "name": "GFLV2Head",
        "alias": "gflv2_head",
        "kind": "anchor",
        "loss": "DenseGFLV2Loss",
        "decoder": "DenseGFLV2Decoder",
        "loss_attrs": ("loss_cls", "loss_dfl"),
    },
    {
        "name": "VFNetHead",
        "alias": "vfnet_head",
        "kind": "anchor",
        "loss": "DenseVFNetLoss",
        "decoder": "DenseVFNetDecoder",
        "loss_attrs": ("loss_cls", "loss_bbox"),
    },
    {
        "name": "PAAHead",
        "alias": "paa_head",
        "kind": "anchor",
        "loss": "DensePAALoss",
        "decoder": "DensePAADecoder",
    },
    {
        "name": "RepPointsHead",
        "alias": "reppoints_head",
        "kind": "anchor_free",
        "loss": "DenseRepPointsLoss",
        "decoder": "DenseRepPointsDecoder",
        "params": {"point_strides": (16, 32)},
    },
    {
        "name": "YOLOFHead",
        "alias": "yolof_head",
        "kind": "anchor_free",
        "loss": "DenseYOLOFLoss",
        "decoder": "DenseYOLOFDecoder",
    },
    {
        "name": "TOODHead",
        "alias": "tood_head",
        "kind": "anchor_free",
        "loss": "DenseTOODLoss",
        "decoder": "DenseTOODDecoder",
    },
    {
        "name": "DDODHead",
        "alias": "ddod_head",
        "kind": "anchor",
        "loss": "DenseDDODLoss",
        "decoder": "DenseDDODDecoder",
    },
    {
        "name": "AutoAssignHead",
        "alias": "auto_assign_head",
        "kind": "anchor_free",
        "loss": "DenseAutoAssignLoss",
        "decoder": "DenseAutoAssignDecoder",
    },
    {
        "name": "NASFCOSHead",
        "alias": "nas_fcos_head",
        "kind": "anchor_free",
        "loss": "DenseNASFCOSLoss",
        "decoder": "DenseNASFCOSDecoder",
    },
)


class NativeDenseHeadTests(unittest.TestCase):
    def test_list_heads_dense_includes_core_dense_aliases(self):
        require_torch()

        from simpledet.suite import list_heads

        dense_heads = set(list_heads(kind="dense"))

        for case in _HEAD_CASES:
            with self.subTest(alias=case["alias"]):
                self.assertIn(case["alias"], dense_heads)

    def test_list_heads_dense_includes_advanced_dense_aliases(self):
        require_torch()

        from simpledet.suite import list_heads

        dense_heads = set(list_heads(kind="dense"))

        for case in _ADVANCED_HEAD_CASES:
            with self.subTest(alias=case["alias"]):
                self.assertIn(case["alias"], dense_heads)

    def test_advanced_dense_head_registry_metadata_is_validated(self):
        require_torch()

        import simpledet.native.heads  # noqa: F401
        from simpledet.extensions import HEADS

        for case in _ADVANCED_HEAD_CASES:
            with self.subTest(alias=case["alias"]):
                metadata = HEADS.lookup(case["alias"])
                self.assertEqual(metadata.name, case["name"])
                self.assertEqual(metadata.family, "dense")
                self.assertTrue(metadata.required_dependencies)
                self.assertTrue(metadata.tensor_contracts)
                self.assertEqual(metadata.validation_status, "runtime_validated")
                self.assertIn("native", metadata.summary)

    def test_dense_head_num_classes_must_be_positive(self):
        require_torch()

        from simpledet.native.heads import build_native_head
        from simpledet.suite import build_head

        with self.assertRaisesRegex(ValueError, "num_classes.*positive integer"):
            build_head("retina_head", num_classes=0)

        with self.assertRaisesRegex(ValueError, "num_classes.*positive integer"):
            build_native_head(
                type("HeadPlan", (), {"type": "fcos_head", "params": {}})(),
                out_channels=8,
                num_classes=0,
            )

    def test_core_dense_heads_construct_and_match_forward_contracts(self):
        torch = require_torch()
        torch.manual_seed(0)

        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(32, 32),
            feature_channels=8,
            feature_shapes=((2, 2), (1, 1)),
            boxes_per_image=1,
            num_classes=3,
        )

        for case in _HEAD_CASES:
            with self.subTest(head=case["name"]):
                head, head_spec = self._build_head(case)
                head.eval()

                with torch.no_grad():
                    outputs = head(batch.feature_maps)

                self.assertEqual(head_spec.name, case["name"])
                self.assertEqual(head_spec.num_classes, 3)
                self.assertEqual(head_spec.in_channels, 8)
                self._assert_forward_shapes(torch, case, outputs, batch.feature_maps)

    def test_core_dense_heads_compute_finite_loss_smoke(self):
        torch = require_torch()
        torch.manual_seed(0)

        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(32, 32),
            feature_channels=8,
            feature_shapes=((2, 2), (1, 1)),
            boxes_per_image=1,
            num_classes=3,
        )

        for case in _HEAD_CASES:
            with self.subTest(head=case["name"]):
                head, _ = self._build_head(case)
                head.train()
                outputs = head(batch.feature_maps)
                loss_fn = self._dense_op(case["loss"])()

                losses = loss_fn(
                    [batch.images[0]],
                    [batch.targets[0]],
                    [batch.feature_maps],
                    [outputs],
                )

                self.assertIn("loss_total", losses)
                self.assertTrue(bool(torch.isfinite(losses["loss_total"])))

    def test_core_dense_heads_decode_inference_payloads(self):
        torch = require_torch()
        torch.manual_seed(0)

        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(32, 32),
            feature_channels=8,
            feature_shapes=((2, 2), (1, 1)),
            boxes_per_image=1,
            num_classes=3,
        )

        for case in _HEAD_CASES:
            with self.subTest(head=case["name"]):
                head, _ = self._build_head(case)
                head.eval()
                decoder = self._dense_op(case["decoder"])(score_threshold=0.0, detections_per_img=5)

                with torch.no_grad():
                    outputs = self._decode_safe_outputs(torch, case, head(batch.feature_maps))
                    detections = decoder(batch.images[0], batch.feature_maps, outputs)

                self.assertEqual(set(detections), {"boxes", "scores", "labels"})
                self.assertEqual(detections["boxes"].dim(), 2)
                self.assertEqual(detections["boxes"].shape[-1], 4)
                self.assertEqual(detections["scores"].dim(), 1)
                self.assertEqual(detections["labels"].dim(), 1)
                self.assertEqual(detections["boxes"].shape[0], detections["scores"].shape[0])
                self.assertEqual(detections["boxes"].shape[0], detections["labels"].shape[0])

    def test_advanced_dense_heads_construct_and_match_forward_contracts(self):
        torch = require_torch()
        torch.manual_seed(0)

        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(32, 32),
            feature_channels=8,
            feature_shapes=((2, 2), (1, 1)),
            boxes_per_image=1,
            num_classes=3,
        )

        for case in _ADVANCED_HEAD_CASES:
            with self.subTest(head=case["name"]):
                head, head_spec = self._build_head(case)
                head.eval()

                with torch.no_grad():
                    outputs = head(batch.feature_maps)

                self.assertEqual(head_spec.name, case["name"])
                self.assertEqual(head_spec.num_classes, 3)
                self.assertEqual(head_spec.in_channels, 8)
                for attr in case.get("loss_attrs", ()):
                    self.assertTrue(hasattr(head, attr))
                self._assert_forward_shapes(torch, case, outputs, batch.feature_maps)

    def test_advanced_dense_heads_compute_finite_loss_smoke(self):
        torch = require_torch()
        torch.manual_seed(0)

        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(32, 32),
            feature_channels=8,
            feature_shapes=((2, 2), (1, 1)),
            boxes_per_image=1,
            num_classes=3,
        )

        for case in _ADVANCED_HEAD_CASES:
            with self.subTest(head=case["name"]):
                head, _ = self._build_head(case)
                head.train()
                outputs = head(batch.feature_maps)
                loss_fn = self._dense_op(case["loss"])()

                losses = loss_fn(
                    [batch.images[0]],
                    [batch.targets[0]],
                    [batch.feature_maps],
                    [outputs],
                )

                self.assertIn("loss_total", losses)
                self.assertTrue(bool(torch.isfinite(losses["loss_total"])))

    def test_advanced_dense_heads_decode_inference_payloads(self):
        torch = require_torch()
        torch.manual_seed(0)

        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(32, 32),
            feature_channels=8,
            feature_shapes=((2, 2), (1, 1)),
            boxes_per_image=1,
            num_classes=3,
        )

        for case in _ADVANCED_HEAD_CASES:
            with self.subTest(head=case["name"]):
                head, _ = self._build_head(case)
                head.eval()
                decoder = self._dense_op(case["decoder"])(score_threshold=0.0, detections_per_img=5)

                with torch.no_grad():
                    outputs = self._decode_safe_outputs(torch, case, head(batch.feature_maps))
                    detections = decoder(batch.images[0], batch.feature_maps, outputs)

                self.assertEqual(set(detections), {"boxes", "scores", "labels"})
                self.assertEqual(detections["boxes"].dim(), 2)
                self.assertEqual(detections["boxes"].shape[-1], 4)
                self.assertEqual(detections["scores"].dim(), 1)
                self.assertEqual(detections["labels"].dim(), 1)
                self.assertEqual(detections["boxes"].shape[0], detections["scores"].shape[0])
                self.assertEqual(detections["boxes"].shape[0], detections["labels"].shape[0])

    def test_build_head_vfnet_direct_returns_varifocal_native_module(self):
        require_torch()

        from simpledet.native.losses import VarifocalLoss
        from simpledet.suite import HeadSpec, build_head

        spec = build_head(name="vfnet_head", num_classes=3)
        head = build_head(name="vfnet_head", num_classes=3, in_channels=8, num_convs=1)

        self.assertIsInstance(spec, HeadSpec)
        self.assertEqual(spec.name, "vfnet_head")
        self.assertEqual(head.native_head_spec.name, "VFNetHead")
        self.assertEqual(head.native_head_spec.in_channels, 8)
        self.assertIsInstance(head.loss_cls, VarifocalLoss)

    def test_reppoints_requires_point_generation_settings(self):
        require_torch()

        from simpledet.native.heads import build_native_head

        with self.assertRaisesRegex(ValueError, "RepPointsHead requires point-generation settings"):
            build_native_head(
                type("HeadPlan", (), {"type": "reppoints_head", "params": {"num_convs": 1}})(),
                out_channels=8,
                num_classes=3,
            )

    def test_reppoints_rejects_point_generation_level_mismatch(self):
        torch = require_torch()
        torch.manual_seed(0)

        head, _ = self._build_head(
            {
                "name": "RepPointsHead",
                "alias": "reppoints_head",
                "kind": "anchor_free",
                "params": {"point_strides": (16,)},
            }
        )
        batch = make_cpu_detector_smoke_batch(
            batch_size=1,
            image_size=(32, 32),
            feature_channels=8,
            feature_shapes=((2, 2), (1, 1)),
            boxes_per_image=1,
            num_classes=3,
        )

        with self.assertRaisesRegex(ValueError, "point_strides defines 1 levels"):
            head(batch.feature_maps)

    def _build_head(self, case):
        from simpledet.native.heads import build_native_head

        params = dict(case.get("params", {}))
        if case["kind"] != "retina":
            params["num_convs"] = 1
        plan = type("HeadPlan", (), {"type": case["alias"], "params": params})()
        try:
            return build_native_head(plan, out_channels=8, num_classes=3)
        except ImportError as exc:
            self.skipTest(str(exc))

    def _dense_op(self, name):
        from simpledet.native import dense_ops

        return getattr(dense_ops, name)

    def _assert_forward_shapes(self, torch, case, outputs, feature_maps):
        kind = case["kind"]
        if kind == "rpn":
            assert_dense_head_output_contract(
                outputs,
                feature_maps,
                batch_size=1,
                required_keys=("objectness_logits", "bbox_regression"),
            )
            self._assert_per_level_channels(outputs["objectness_logits"], 9)
            self._assert_per_level_channels(outputs["bbox_regression"], 36)
            return

        if kind == "anchor_free":
            assert_dense_head_output_contract(
                outputs,
                feature_maps,
                batch_size=1,
                required_keys=("cls_logits", "bbox_regression", "centerness"),
            )
            self._assert_per_level_channels(outputs["cls_logits"], 3)
            self._assert_per_level_channels(outputs["bbox_regression"], 4)
            self._assert_per_level_channels(outputs["centerness"], 1)
            return

        cls_logits = outputs["cls_logits"]
        bbox_regression = outputs["bbox_regression"]
        if torch.is_tensor(cls_logits):
            expected_priors = sum(int(level.shape[-2]) * int(level.shape[-1]) * 9 for level in feature_maps)
            self.assertEqual(tuple(cls_logits.shape), (1, expected_priors, 3))
            self.assertEqual(tuple(bbox_regression.shape), (1, expected_priors, 4))
            return

        assert_dense_head_output_contract(
            outputs,
            feature_maps,
            batch_size=1,
            required_keys=("cls_logits", "bbox_regression"),
        )
        self._assert_per_level_channels(outputs["cls_logits"], 27)
        self._assert_per_level_channels(outputs["bbox_regression"], 36)

    def _assert_per_level_channels(self, levels, expected_channels):
        for level in levels:
            self.assertEqual(int(level.shape[1]), expected_channels)

    def _decode_safe_outputs(self, torch, case, outputs):
        if case["kind"] == "anchor_free":
            outputs = dict(outputs)
            outputs["bbox_regression"] = [torch.ones_like(level) for level in outputs["bbox_regression"]]
            return outputs
        if case["kind"] == "rpn":
            outputs = dict(outputs)
            outputs["bbox_regression"] = [torch.zeros_like(level) for level in outputs["bbox_regression"]]
            return outputs
        outputs = dict(outputs)
        bbox = outputs["bbox_regression"]
        if torch.is_tensor(bbox):
            outputs["bbox_regression"] = torch.zeros_like(bbox)
        else:
            outputs["bbox_regression"] = [torch.zeros_like(level) for level in bbox]
        return outputs


if __name__ == "__main__":
    unittest.main()
