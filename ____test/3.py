# ID
# ID_NhanVien
# Ngay
# Gio
# HA_BASE64
import cv2, base64, pyodbc, io
import numpy as np
from PIL import Image
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=112.78.15.3;"
    "Database=ChamCong_Cafe;"
    "uid=ngoi;"
    "pwd=admin123;"
)

def chuyen_base64_sang_anh(anh_base64):
    try:
        anh_base64 = np.frombuffer(base64.b64decode(anh_base64), dtype=np.uint8)
        anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
    except:
        return "chuyen fail"
    return anh_base64

# tenha = 'HS_2027_PKCLLXBGGP_IMG_1201.jpg'
# detected_face1 = cv2.resize(cv2.imread('hinhanh/NV_2027/'+tenha), (224, 224)) #resize to 224x224
# ret, buffer = cv2.imencode('.jpg', cv2.imread('hinhanh/NV_2027/'+tenha))
# image1 = base64.b64encode(buffer).decode("utf-8")
# img = chuyen_base64_sang_anh(image1)
# cv2.imwrite('____test/local_img.jpg', img)

cursor = conn.cursor()
query = "SELECT HA_BASE64, Ngay, ID_NhanVien, Gio FROM ChamCong_Tam"
cursor.execute(query)

# Lấy tất cả các hàng từ kết quả truy vấn
rows = cursor.fetchall()
for i, row in enumerate(rows):
    print("ID_NhanVien:", row.ID_NhanVien, " --- Ngay:", str(row.Ngay).split(' ')[0], " --- Gio:", str(row.Gio).split('.')[0], '#', i)
    # img = chuyen_base64_sang_anh(row.HA_BASE64)
    # cv2.imwrite(f'____test/{i}.jpg', img)