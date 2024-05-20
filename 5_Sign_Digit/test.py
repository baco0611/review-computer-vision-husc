import os
from PIL import Image
import pandas as pd

# Đường dẫn tới thư mục chứa các hình ảnh
image_folder = "./dataset/images/0"

# Khởi tạo danh sách lưu trữ độ phân giải
resolutions = []

# Duyệt qua toàn bộ các file trong thư mục
for filename in os.listdir(image_folder):
    if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):  # Chỉ xử lý các file ảnh
        file_path = os.path.join(image_folder, filename)
        with Image.open(file_path) as img:
            width, height = img.size
            resolutions.append((filename, width, height, width * height))

# Chuyển danh sách độ phân giải thành DataFrame
df = pd.DataFrame(resolutions, columns=["filename", "width", "height", "resolution"])

# Sắp xếp theo độ phân giải (tích của width và height)
df = df.sort_values(by="resolution")

# Lưu vào file CSV
output_csv = "resolutions_sorted.csv"
df.to_csv(output_csv, index=False)

print(f"Độ phân giải của các hình ảnh đã được lưu vào {output_csv}")
