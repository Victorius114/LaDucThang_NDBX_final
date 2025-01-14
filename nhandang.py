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

def bienxe():
    file_path = filedialog.askopenfilename(
        title="Chọn một hình ảnh",
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )
    if not file_path:
        print("Không có hình ảnh nào được chọn.")
        return

    image = cv2.imread(file_path)
    if image is None:
        print("Không thể đọc được ảnh.")
        return

    # Chuyển ảnh sang RGB để sử dụng với mô hình YOLO
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phát hiện đối tượng bằng mô hình YOLO
    kq = model(image_rgb)

    # Lấy các bounding boxes từ kết quả phát hiện
    results = kq[0].boxes.xyxy  # Lấy tọa độ (x1, y1, x2, y2)
    if results is None or len(results) == 0:
        print("Không phát hiện biển số xe nào.")
        return

    # Vẽ kết quả phát hiện lên ảnh
    anh_kq = kq[0].plot()

    # Đọc và in các ký tự từ từng vùng biển số
    for box in results.cpu().numpy():  # Lặp qua từng bounding box
        x1, y1, x2, y2 = map(int, box[:4])  # Lấy tọa độ
        cropped_plate = image_rgb[y1:y2, x1:x2]  # Cắt vùng biển số
        gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_RGB2GRAY)  # Chuyển sang ảnh xám

        # Chuyển ảnh xám thành ảnh nhị phân
        _, binary_plate = cv2.threshold(gray_plate, 5, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Sử dụng Tesseract OCR để nhận diện ký tự
        text = pytesseract.image_to_string(binary_plate, config="--psm 6")
        print(f"Biển số đọc được: {text.strip()}")

    # Hiển thị ảnh với vùng biển số được khoanh
    plt.figure(figsize=(10, 10))
    plt.imshow(anh_kq)
    plt.axis("off")
    plt.title("Kết quả")
    plt.show()

def bienxe_video():
    file_path = filedialog.askopenfilename(
        title="Chọn một video",
        filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")]
    )
    if not file_path:
        print("Không có video nào được chọn.")
        return

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print("Không thể mở video.")
        return

    plt.ion()  # Bật chế độ hiển thị liên tục của Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    img_display = ax.imshow(np.zeros((10, 10, 3), dtype=np.uint8))  # Hình ảnh giả ban đầu
    ax.axis("off")
    plt.title("Phát hiện biển số xe")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển khung hình sang RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Phát hiện đối tượng
        kq = model(frame_rgb)

        # Vẽ kết quả phát hiện
        frame_result = kq[0].plot()

        # Cập nhật hình ảnh hiển thị
        img_display.set_data(frame_result)
        fig.canvas.draw()
        fig.canvas.flush_events()

    cap.release()
    plt.ioff()
    plt.show()

# Giao diện tkinter
root = tk.Tk()
root.title("Nhận dạng biển số xe")

root.geometry("600x400")

frame = tk.Frame(root)
frame.pack(expand=True, fill="both")

button_image = tk.Button(frame, text="Chọn ảnh và phát hiện đối tượng", command=bienxe)
button_image.pack(pady=10)

button_video = tk.Button(frame, text="Chọn video và phát hiện đối tượng", command=bienxe_video)
button_video.pack(pady=10)

root.mainloop()
