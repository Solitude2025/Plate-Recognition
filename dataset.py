import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=640, transforms=None):
   
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.transforms = transforms
        
        # 收集所有图片路径
        self.image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.image_files.sort()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        
        label_path = os.path.join(self.labels_dir, os.path.splitext(self.image_files[idx])[0] + '.txt')
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                   
                    cls_id, cx, cy, w, h = map(float, line.strip().split())
                    boxes.append([cls_id, cx, cy, w, h])
        boxes = torch.tensor(boxes) if boxes else torch.zeros((0, 5))
        
        # 预处理图片
        if self.transforms:
            img = self.transforms(img)
        else:
            # 默认缩放和归一化
            img = img.resize((self.img_size, self.img_size))
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
        
        return img, boxes, self.image_files[idx]

