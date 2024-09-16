import mmdet 
from mmdet.models.dense_heads import VFNetHead, FCOSHead
import torch 
import warnings
warnings.filterwarnings("ignore")

# testare teste di mmdet: 
INF = 1e8
testa = VFNetHead(11, 7)
testa = FCOSHead(11, 7)

# dummy input:
feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
cls_score, bbox_pred, bbox_pred_refine= testa.forward(feats)

assert len(cls_score) == len(testa.scales)


# export to onnx testa
dummy_input = [torch.randn(1, 7, s, s) for s in [4, 8, 16, 32, 64]]

torch.onnx.export(testa, 
                  dummy_input, 
                  "testa.onnx", 
                  verbose=False, 
                  input_names=["input"], 
                  output_names=["cls_score", "bbox_pred", "centerness"])
