from ultralytics import YOLO
import torch

# Kiểm tra thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")
