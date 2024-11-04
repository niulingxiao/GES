import os
import torch
from torchvision import models
from models.mvss_res2net import get_mvss_res2net, get_mvss
import onnx

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

load_path = '/path.pth'
model = get_mvss(backbone='resnet50',
                         pretrained_base=True,
                         nclass=1,
                         sobel=True,
                         constrain=True,
                         n_input=3,
                         input_image_size=512,
                         ).to(device)
model_checkpoint = torch.load(load_path, map_location='cpu')
model.load_state_dict(model_checkpoint, strict=True)
print("load %s finish" % (os.path.basename(load_path)))
model = model.eval().to(device)

x = torch.randn(1, 3, 512, 512).to(device)
[edge, mask, out_label] = model(x)
print('edge', edge.shape, 'mask', mask.shape, 'out_label', out_label.shape)

# Export the model
with torch.no_grad():
    torch.onnx.export(
        model,                       # 要转换的模型
        x,                           # 模型的任意一组输入
        'mvss_aug.onnx',    # 导出的 ONNX 文件名
        opset_version=11,            # ONNX 算子集版本
        input_names=['input'],       # 输入 Tensor 的名称（自己起名字）
        output_names=['output_edge', 'output_mask', 'output_label']      # 输出 Tensor 的名称（自己起名字）
    )

print('export onnx finish')

# 读取 ONNX 模型
onnx_model = onnx.load('mvss_aug.onnx')
# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)
print('无报错，onnx模型载入成功')
