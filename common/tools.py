import os
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from common.transforms import direct_val
import torch.nn.functional as F
import pdb
debug = 0


def decode_segmap_train(images, nc=4):  # images为(N,C,H,W)
    images_rgb = []
    # print(images.shape)
    for image in images:
        image = torch.squeeze(image)
        # print(image.shape)
        image_rgb = decode_segmap(image, nc)
        images_rgb.append(image_rgb)
    images_rgb = np.array(images_rgb)
    images_rgb = torch.from_numpy(images_rgb)
    images_rgb = images_rgb.permute(0, 3, 1, 2)
    # print(images_rgb.shape)
    return images_rgb


def decode_segmap(image, nc=4):  # image为(H,W)
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=copymove红, 2=splice绿, 3=inpainting蓝
               (128, 0, 0), (0, 128, 0), (0, 0, 128)])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


transform_pil = transforms.Compose([
    transforms.ToPILImage(),
])


def run_model(model, inputs):
    output = model(inputs)
    return output


def inference_single(img, model, th=0):
    model.eval()
    with torch.no_grad():
        img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
        img = direct_val(img)
        img = img.cuda()
        _, seg = run_model(model, img)
        seg = torch.sigmoid(seg).detach().cpu()
        if torch.isnan(seg).any() or torch.isinf(seg).any():
            max_score = 0.0
        else:
            max_score = torch.max(seg).numpy()
        seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]

        if len(seg) != 1:
            pdb.set_trace()
        else:
            fake_seg = seg[0]
        if th == 0:
            return fake_seg, max_score
        fake_seg = 255.0 * (fake_seg > 255 * th)
        fake_seg = fake_seg.astype(np.uint8)

    return fake_seg, max_score


def inference_single_multiclass(img, model, th=0):
    model.eval()
    # with torch.no_grad():
    img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
    img = direct_val(img)
    img = img.cuda()
    edge, seg, outlabel = run_model(model, img)
    seg = torch.sigmoid(seg).detach().cpu()
    # outlabel = F.softmax(outlabel, dim=0)
    outlabel = np.array(outlabel.squeeze().detach().cpu())

    if torch.isnan(seg).any() or torch.isinf(seg).any():
        max_score = 0.0
    else:
        max_score = torch.max(seg).numpy()
    seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]

    if len(seg) != 1:
        pdb.set_trace()
    else:
        fake_seg = seg[0]
    if th == 0:
        return fake_seg, max_score, outlabel
    fake_seg = 255.0 * (fake_seg > 255 * th)
    fake_seg = fake_seg.astype(np.uint8)

    return fake_seg, max_score, outlabel


def inference_single_multiclass_v2(img, model, th=0):
    model.eval()
    # with torch.no_grad():
    img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
    img = direct_val(img)
    img = img.cuda()
    edge, seg, outlabel = run_model(model, img)
    # seg = torch.sigmoid(seg).detach().cpu()
    # print(seg.dtype)
    # print(seg)

    print(seg.shape)
    # print(seg)
    a = seg.data.cpu().numpy().squeeze()
    print("seg不重复数字：", np.unique(a))
    out_masks = seg.argmax(1)
    print(out_masks.shape)
    a = out_masks.data.cpu().numpy().squeeze()
    print("out_mask不重复数字：", np.unique(a))
    # print(out_masks)
    # print(out_masks.shape)
    seg_numpy = np.array(out_masks.detach().cpu())
    # print(len(seg_numpy[seg_numpy == 1]))
    # print(len(seg_numpy[seg_numpy == 2]))
    # print(len(seg_numpy[seg_numpy == 3]))
    # print(len(seg_numpy[seg_numpy == 0]))

    out_masks = torch.reshape(out_masks, (1, 1, 512, 512))
    # out_masks = img_trans_cat(out_masks)
    out_masks = out_masks.detach().cpu()
    seg = decode_segmap_train(out_masks, nc=4)  # mask2rgb
    print(seg.shape)
    print(seg.dtype)
    # seg_numpy = np.array(seg)
    # print(len(seg_numpy[seg_numpy == 1]))
    # print(len(seg_numpy[seg_numpy == 2]))
    # print(len(seg_numpy[seg_numpy == 3]))
    # print(len(seg_numpy[seg_numpy == 0]))


    outlabel = F.softmax(outlabel, dim=1)
    outlabel = np.array(outlabel.squeeze().detach().cpu())

    if torch.isnan(seg).any() or torch.isinf(seg).any():
        max_score = 0.0
    else:
        max_score = torch.max(seg).numpy()
    seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]

    if len(seg) != 1:
        pdb.set_trace()
    else:
        fake_seg = seg[0]
    if th == 0:
        return fake_seg, max_score, outlabel
    fake_seg = 255.0 * (fake_seg > 255 * th)
    fake_seg = fake_seg.astype(np.uint8)

    return fake_seg, max_score, outlabel


def inference_single_multiclass_v3(img, model, th=0):
    model.eval()
    # with torch.no_grad():
    img = img.reshape((-1, img.shape[-3], img.shape[-2], img.shape[-1]))
    img = direct_val(img)
    img = img.cuda()
    edge, seg, outlabel = run_model(model, img)
    # seg = torch.sigmoid(seg).detach().cpu()
    # print(seg.dtype)
    # print(seg)

    print(seg.shape)
    # print(seg)
    a = seg.data.cpu().numpy().squeeze()
    print("seg不重复数字：", np.unique(a))
    out_masks = seg.argmax(1)
    print(out_masks.shape)
    a = out_masks.data.cpu().numpy().squeeze()
    print("out_mask不重复数字：", np.unique(a))
    # print(out_masks)
    # print(out_masks.shape)
    seg_numpy = np.array(out_masks.detach().cpu())
    # print(len(seg_numpy[seg_numpy == 1]))
    # print(len(seg_numpy[seg_numpy == 2]))
    # print(len(seg_numpy[seg_numpy == 3]))
    # print(len(seg_numpy[seg_numpy == 0]))

    out_masks = torch.reshape(out_masks, (1, 1, 512, 512))
    # out_masks = img_trans_cat(out_masks)
    out_masks = out_masks.detach().cpu()
    seg = decode_segmap_train(out_masks, nc=4)  # mask2rgb
    print("mask_rgb不重复数字：", np.unique(seg))
    print(seg.shape)
    print(seg.dtype)
    # seg_numpy = np.array(seg)
    # print(len(seg_numpy[seg_numpy == 1]))
    # print(len(seg_numpy[seg_numpy == 2]))
    # print(len(seg_numpy[seg_numpy == 3]))
    # print(len(seg_numpy[seg_numpy == 0]))

    print(outlabel.shape)
    outlabel = F.softmax(outlabel.squeeze(), dim=0)
    outlabel = np.array(outlabel.squeeze().detach().cpu())

    if torch.isnan(seg).any() or torch.isinf(seg).any():
        max_score = 0.0
    else:
        max_score = torch.max(seg).numpy()
    seg = [np.array(transform_pil(seg[i])) for i in range(len(seg))]

    if len(seg) != 1:
        pdb.set_trace()
    else:
        fake_seg = seg[0]
    if th == 0:
        return fake_seg, max_score, outlabel
    fake_seg = 255.0 * (fake_seg > 255 * th)
    fake_seg = fake_seg.astype(np.uint8)

    return fake_seg, max_score, outlabel
