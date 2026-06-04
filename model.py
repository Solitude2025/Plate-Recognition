import os
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np
class CharRecNet(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Flatten(), torch.nn.Linear(64*8*8, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.model(x)
