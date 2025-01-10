import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

model = YOLO("nhandang1/nhandang1_model/weights/best.pt")

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

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kq = model(image)

    anh_kq = kq[0].plot()

    plt.figure(figsize=(10, 10))
    plt.imshow(anh_kq)
    plt.axis("off")
    plt.title("YOLOv8 Detection kq")
    plt.show()

root = tk.Tk()
root.title("YOLOv8 Image Detection")

root.geometry("600x400")

frame = tk.Frame(root)
frame.pack(expand=True, fill="both")

button = tk.Button(frame, text="Chọn ảnh và phát hiện đối tượng", command=bienxe)
button.pack(expand=True)


root.mainloop()
