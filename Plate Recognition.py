import os
import cv2
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import numpy as np

PLATE_DETECT_WEIGHTS = 'PlateRec.pt'      # YOLOv8 车牌检测
SEGMENT_WEIGHTS     = 'Best.pt'        # YOLOv8 分割字符
CHAR_REC_WEIGHTS    = 'CharRec.pth'       # 单字符识别


plate_detector = YOLO(PLATE_DETECT_WEIGHTS)
seg_detector   = YOLO(SEGMENT_WEIGHTS)


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


CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
         '云', '京', '冀', '吉', '宁', '川', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']
NUM_CLASSES = len(CHARS)

char_rec_model = CharRecNet(num_classes=NUM_CLASSES)
char_rec_model.load_state_dict(torch.load(CHAR_REC_WEIGHTS, map_location='cpu'))  
char_rec_model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),             # 保证尺寸一致
    transforms.Grayscale(),                 # 单通道
    transforms.ToTensor(),                  # 转 tensor
    transforms.Normalize((0.5,), (0.5,))    # 归一化
])

FONT_PATH   = 'simsun.ttc'   # 系统中文字体文件路径
FONT_SIZE   = 35             # 字号，可自行调整大小
TEXT_COLOR  = (0, 255, 0)    # 字体颜色，RGB格式
TEXT_OFFSET = -30            # 文字垂直偏移量，负值表示往上移像素数
try:
    FONT = ImageFont.truetype(FONT_PATH, FONT_SIZE)
except IOError:
    FONT = ImageFont.load_default()

def process_image(image_path, output_dir):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"无法读取图片: {image_path}")
        return
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

   # dets = plate_detector(image_path)[0] #启用车牌侦测日志
    dets = plate_detector(image_path, verbose=False, show=False)[0] #禁用车牌侦测日志
    plate_number = ''
    for det in dets.boxes.data.tolist():
        x1, y1, x2, y2, score, cls = det
        if score < 0.3:
            continue
        plate_crop = img_bgr[int(y1):int(y2), int(x1):int(x2)]
        #segs = seg_detector(plate_crop)[0] #启用分割日志
        segs = seg_detector(plate_crop, verbose=False, show=False)[0]#禁用分割日志
        chars = []
        for box in segs.boxes.xyxy.tolist():
            sx1, sy1, sx2, sy2 = box
            char_img = plate_crop[int(sy1):int(sy2), int(sx1):int(sx2)]
            pil_char = Image.fromarray(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)).convert('L')
            tensor = transform(pil_char).unsqueeze(0)
            with torch.no_grad():
                idx = char_rec_model(tensor).argmax(dim=1).item()
                chars.append((sx1, CHARS[idx]))
        plate_number = ''.join([c for _, c in sorted(chars, key=lambda x: x[0])])
        # 绘制：边框与文字
        draw.rectangle([x1, y1, x2, y2], outline=TEXT_COLOR, width=2)
        draw.text((x1, y1 + TEXT_OFFSET), plate_number, font=FONT, fill=TEXT_COLOR)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    pil_img.save(out_path)
    print(f"处理完成: {image_path} -> {plate_number}")


if __name__ == '__main__':
    input_folder = 'inputs'
    output_folder = 'results'
    for fname in os.listdir(input_folder):
        if fname.lower().endswith(('.jpg','.png','.jpeg')):
            process_image(os.path.join(input_folder, fname), output_folder)
    print('全部处理完毕')
