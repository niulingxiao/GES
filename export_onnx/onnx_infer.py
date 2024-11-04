import onnxruntime
import numpy as np
import torch
import torch.nn.functional as F
import time
import pandas as pd

ort_session = onnxruntime.InferenceSession('mvss_aug_sig.onnx')

# # 随机输入测试
# x = torch.randn(1, 3, 512, 512).numpy()
# # onnx runtime 输入
# ort_inputs = {'input': x}
# # onnx runtime 输出
# ort_output = ort_session.run(['output_mask'], ort_inputs)[0]
# print(ort_output.shape)

# 图像输入测试
img_path = '/home/zzt/pan1/DATASETS/MISD_Dataset/Ground_Truth_Masks/Sp_D_art_00046_cha_00026_art_00008_158/images/Sp_D_art_00046_cha_00026_art_00008_158.png'
# 用 pillow 载入
from PIL import Image
from torchvision import transforms
# 测试集图像预处理：缩放、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize((512, 512)),
                                     # transforms.CenterCrop(256),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.0, 0.0, 0.0],
                                         std=[1.0, 1.0, 1.0])
                                     ])
for i in range(10):
    start = time.time()
    img_pil = Image.open(img_path)
    img_pil = img_pil.convert("RGB")
    input_tensor = test_transform(img_pil).unsqueeze(0).numpy()
    print(input_tensor.shape)

    # ONNX Runtime 输入
    ort_inputs = {'input': input_tensor}
    # ONNX Runtime 输出
    pred_logits, pred_label, pred_edge = ort_session.run(['output_mask', 'output_label', 'output_edge'], ort_inputs)
    end = time.time()
    print('time', end-start)
pred_logits = torch.tensor(pred_logits)
print(pred_logits.shape)
 # score = np.array(torch.sigmoid(out_cls).squeeze().detach().cpu())
# seg = torch.sigmoid(pred_logits).detach().cpu()
seg = pred_logits.detach().cpu()
print(seg.shape)


