import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_boxes(image, boxes, labels=None, color=(0, 255, 0), font_path='simsun.ttc', font_size=20):
    """"

    参数:
    - image: 原图 (np.array)
    - boxes: 边界框列表 [[x1, y1, x2, y2], ...]
    - labels: 标签列表 ['京A12345', ...]
    - color: 框颜色 (BGR)
    - font_path: 中文字体路径
    - font_size: 字体大小
    """
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if labels and i < len(labels):
            draw.text((x1, y1 - font_size), labels[i], font=font, fill=color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)



def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)

    box1_area = max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0)
    box2_area = max(box2[2] - box2[0], 0) * max(box2[3] - box2[1], 0)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def load_model(model, weight_path, device='cpu'):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到模型权重文件：{weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"已加载模型权重：{weight_path}")
    return model


def save_image(img_tensor, save_path):
    img = img_tensor.clone().detach()
    img = img.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    img = img.astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"保存图片到 {save_path}")
