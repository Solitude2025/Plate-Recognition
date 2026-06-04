import torch

print(torch.__version__)         # 应该输出类似 '2.3.0'
print(torch.version.cuda)        # 应该输出 '12.1'
print(torch.cuda.is_available()) # 应该输出 True
print(torch.cuda.get_device_name(0))  # 应该输出你的 GPU 名称，如 'NVIDIA GeForce RTX 4060'
