# ID_HinhAnh, TenHA, HA_BASE64, ID_NhanVien, Path, Bin_Array
import numpy as np
import pyodbc, os, sys
import base64
from PIL import Image

# Kết nối đến cơ sở dữ liệu
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=112.78.15.3;"
    "Database=ChamCong_Cafe;"
    "uid=ngoi;"
    "pwd=admin123;"
)



cursor = conn.cursor()

# Thực hiện truy vấn SELECT
query = "SELECT ID_NhanVien, TenHA, Path FROM HinhAnh_NhanDien"
cursor.execute(query)

# Lấy tất cả các hàng dữ liệu
rows = cursor.fetchall()

# In ra tất cả các giá trị ID_NhanVien và TenHA
for row in rows:
    id_nhanvien = row.ID_NhanVien
    ten_ha = row.TenHA
    path = row.Path
    print(f"ID_NhanVien: {id_nhanvien}, TenHA: {ten_ha}, Path: {path}")

# Đóng con trỏ và kết nối
cursor.close()
conn.close()
    


def save_array_to_table(array):
    # Tạo con trỏ cursor để thực hiện truy vấn
    cursor = conn.cursor()

    # Chuyển đổi mảng NumPy thành chuỗi byte
    array_bytes = array.tobytes()

    # Thực hiện truy vấn INSERT
    query = "INSERT INTO TableName (ArrayColumn) VALUES (?)"
    cursor.execute(query, (pyodbc.Binary(array_bytes),))

    # Xác nhận việc thêm dữ liệu
    conn.commit()
    print("Đã lưu trữ mảng vào bảng")

    # Đóng con trỏ và kết nối
    cursor.close()
    conn.close()

def retrieve_array_from_table():
    # Tạo con trỏ cursor để thực hiện truy vấn
    cursor = conn.cursor()

    # Thực hiện truy vấn SELECT
    query = "SELECT ArrayColumn FROM TableName"
    cursor.execute(query)

    # Lấy dữ liệu từ cột ArrayColumn
    row = cursor.fetchone()
    array_bytes = row.ArrayColumn

    # Chuyển đổi dữ liệu nhị phân thành mảng NumPy
    array = np.frombuffer(array_bytes, dtype=np.float64).reshape((1, 2048))

    # In ra mảng
    print("Mảng đã lấy từ bảng:")
    print(array)

    # Đóng con trỏ và kết nối
    cursor.close()
    conn.close()


# sys.path.append('D:/Code/FaceID/')    
# from keras.applications.imagenet_utils import preprocess_input
# from tensorflow.keras.utils import load_img, save_img, img_to_array   
# from facereg_model import loadVggFaceModel

# def preprocess_image(image_path):
#     img = load_img(image_path, target_size=(224, 224))
#     img = img_to_array(img)
#     img = np.expand_dims(img, axis=0)
#     img = preprocess_input(img)
#     return img

# model = loadVggFaceModel()
# # array1 = np.random.random((1, 2048))
# array1 = model.predict(preprocess_image("hinhtrain/NV_1022/HS_1022_AFGIVHGSMA_00002.jpg"))
# # Chuyển đổi mảng NumPy thành chuỗi byte
# array_bytes = array1.tobytes()

# # Thực hiện truy vấn INSERT
# bin = pyodbc.Binary(array_bytes)

# array2 = np.frombuffer(bin, dtype=np.float64).reshape((1, 2048))
# print(array1-array2)