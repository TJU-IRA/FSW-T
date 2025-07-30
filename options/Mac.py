import torch
from thop import profile
from model import PhysFusionNet  # 替换为你真实的模型定义

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PhysFusionNet(param_dim=3).to(device)
model.eval()

# 构造模型输入（与你训练代码一致）
dummy_roi = torch.randn(1, 6, 64, 64).to(device)
dummy_params = torch.randn(1, 3).to(device)
dummy_seq = torch.zeros(1, 5, 1).to(device)

# 计算 FLOPs 和 参数
macs, params = profile(model, inputs=(dummy_roi, dummy_params, dummy_seq))
print(f"MACs: {macs / 1e6:.2f} M")
print(f"Params: {params / 1e3:.2f} K")
