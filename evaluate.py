import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

MODEL_PATH = r"D:\VScode\Python project6(Plate Recognition)\PlateRec.pt"
DATA_PATH = r"D:\VScode\Python project3\ultralytics\datasets\mydata.yaml"
IMG_SIZE = 640
BATCH_SIZE = 8
OUT_PATH = 'map_plot.png'

def main():
    model = YOLO(MODEL_PATH)
    metrics = model.val(data=DATA_PATH, imgsz=IMG_SIZE, batch=BATCH_SIZE)

    mp, mr, map50, map95 = metrics.box.mean_results()

    names = ['Precision', 'Recall', 'mAP@0.5', 'mAP@0.5:0.95']
    values = [mp, mr, map50, map95]

    plt.figure(figsize=(8, 6))
    plt.bar(names, values, color='lightcoral')
    plt.ylim(0, 1.3)
    plt.title('YOLOv8 Evaluation Metrics')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=12)
    plt.savefig(OUT_PATH)
    plt.show()

    print(f"验证完成，图像保存为：{OUT_PATH}")

if __name__ == '__main__':
    main()
