import torch
from model.swin_transformer import SwinTransformer

net = SwinTransformer(
    num_classes=3,
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3
)

dummy_x = torch.randn(1, 3, 224, 224)
logits = net(dummy_x)  # (1,3)
