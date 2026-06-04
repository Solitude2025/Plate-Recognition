import os
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
transform = transforms.Compose([
    transforms.Resize((32,32)),             # 保证尺寸一致
    transforms.Grayscale(),                 # 单通道
    transforms.ToTensor(),                  # 转 tensor
    transforms.Normalize((0.5,), (0.5,))    # 归一化
])
