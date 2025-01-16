import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import cv2
import pytesseract
import torch
from ultralytics import YOLO
import numpy as np

# Kiểm tra và sử dụng GPU (nếu có)
device = "cuda"
model = YOLO("nhandang1/nhandang1_model/weights/best.pt")  # Không cần dùng model.to(device) nữa vì YOLO tự động sử dụng GPU nếu có

# Đặt đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r"E:\Ky2_2024_2025\Thi_giac_may\Tesseract-OCR\tesseract.exe"


def preprocess_plate(plate_image):
    """
    Tiền xử lý ảnh biển số trước khi nhận diện ký tự
    """
    # Chuyển ảnh sang ảnh xám
    gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_RGB2GRAY)

    # Loại bỏ nhiễu bằng Gaussian Blur
    blurred_plate = cv2.GaussianBlur(gray_plate, (5, 5), 0)

    # Tăng cường độ tương phản bằng CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_plate = clahe.apply(blurred_plate)

    # Chuyển ảnh sang nhị phân bằng Otsu thresholding
    _, binary_plate = cv2.threshold(enhanced_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Áp dụng phép toán hình thái học (Dilation)
    kernel = np.ones((3, 3), np.uint8)
    processed_plate = cv2.dilate(binary_plate, kernel, iterations=1)

    return processed_plate


def bienxe():
    file_path = filedialog.askopenfilename(
        title="Chọn một hình ảnh",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not file_path:
        label_result.config(text="Không có hình ảnh nào được chọn.")
        return

    image = cv2.imread(file_path)
    if image is None:
        label_result.config(text="Không thể đọc được ảnh.")
        return

    # Chuyển ảnh sang RGB để sử dụng với mô hình YOLO
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phát hiện đối tượng bằng mô hình YOLO
    kq = model(image_rgb)

    # Lấy các bounding boxes từ kết quả phát hiện
    results = kq[0].boxes.xyxy  # Lấy tọa độ (x1, y1, x2, y2)
    if results is None or len(results) == 0:
        label_result.config(text="Không phát hiện biển số xe nào.")
        return

    detected_texts = []
    # Vẽ kết quả phát hiện lên ảnh
    anh_kq = kq[0].plot()

    # Đọc và in các ký tự từ từng vùng biển số
    for box in results.cpu().numpy():  # Lặp qua từng bounding box
        x1, y1, x2, y2 = map(int, box[:4])  # Lấy tọa độ
        cropped_plate = image_rgb[y1:y2, x1:x2]  # Cắt vùng biển số

        # Tiền xử lý ảnh biển số
        processed_plate = preprocess_plate(cropped_plate)

        # Sử dụng Tesseract OCR để nhận diện ký tự
        text = pytesseract.image_to_string(
            processed_plate, config="--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        detected_texts.append(text.strip())

    # Hiển thị kết quả nhận diện ký tự
    result_text = "\n".join(detected_texts)
    label_result.config(text=f"Biển số đọc được:\n{result_text}")

    # Hiển thị ảnh với vùng biển số được khoanh
    plt.figure(figsize=(10, 10))
    plt.imshow(anh_kq)
    plt.axis("off")
    plt.title("Kết quả")
    plt.show()


# Giao diện tkinter
root = tk.Tk()
root.title("Nhận dạng biển số xe")

root.geometry("600x400")

frame = tk.Frame(root)
frame.pack(expand=True, fill="both")

button_image = tk.Button(frame, text="Chọn ảnh và phát hiện đối tượng", command=bienxe)
button_image.pack(pady=10)

label_result = tk.Label(frame, text="", wraplength=500, justify="left")
label_result.pack(pady=10)

root.mainloop()

