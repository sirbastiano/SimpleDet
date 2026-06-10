import unittest

from native_tensor_contracts import require_torch


class NativeCNNBlocksTests(unittest.TestCase):
    def test_convnext_block_preserves_nchw_shape(self):
        torch = require_torch()
        from simpledet.native.cnn_blocks import ConvNeXtBlock

        block = ConvNeXtBlock(8)
        block.eval()

        with torch.no_grad():
            output = block(torch.zeros(2, 8, 16, 16))

        self.assertEqual(tuple(output.shape), (2, 8, 16, 16))

    def test_convnext_feature_backbone_returns_selected_feature_maps(self):
        torch = require_torch()
        from simpledet.native.cnn_blocks import ConvNeXtFeatureBackbone

        backbone = ConvNeXtFeatureBackbone(
            model_name="convnext_tiny",
            out_indices=(1, 3, 4),
            depths=(1, 1, 1, 1),
        )
        backbone.eval()

        with torch.no_grad():
            features = backbone(torch.zeros(1, 3, 64, 64))

        self.assertEqual(backbone.feature_channels, (96, 384, 768))
        self.assertEqual(len(features), 3)
        self.assertEqual(tuple(int(feature.shape[1]) for feature in features), (96, 384, 768))
        self.assertEqual(tuple(features[0].shape[-2:]), (16, 16))
        self.assertEqual(tuple(features[-1].shape[-2:]), (2, 2))


if __name__ == "__main__":
    unittest.main()
