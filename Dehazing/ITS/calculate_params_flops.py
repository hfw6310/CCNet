import torch
from thop import profile

from models.SANet import build_net

# from basicsr.archs.lkdns_arch import LKDN_S as net

# pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git

model = build_net(40)

net_cls_str = f'{model.__class__.__name__}'

# thop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
inputs = torch.randn(1, 3, 256, 256).to(device)
flops, params = profile(model, (inputs, ))
print(f'Network: {net_cls_str}, with flops(256 x 256): {flops/1e9:.2f} GMac, with active parameters: {params/1e3} K.')
