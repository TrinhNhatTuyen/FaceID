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

tenha = 'HS_2027_PKCLLXBGGP_IMG_1201.jpg'
detected_face1 = cv2.resize(cv2.imread('hinhanh/NV_2027/'+tenha), (224, 224)) #resize to 224x224
ret, buffer = cv2.imencode('.jpg', cv2.imread('hinhanh/NV_2027/'+tenha))
image1 = base64.b64encode(buffer).decode("utf-8")
img = chuyen_base64_sang_anh(image1)
cv2.imwrite('____test/local_img.jpg', img)

cursor = conn.cursor()
query = f"SELECT HA_BASE64 FROM HinhAnh_NhanDien WHERE TenHA = 'HS_2027_PKCLLXBGGP_IMG_1201.jpg'"
cursor.execute(query)
row = cursor.fetchone()
if row:
    ha_base64 = row[0]
    img2 = chuyen_base64_sang_anh(ha_base64)
    cv2.imwrite('____test/database_img.jpg', img2)
conn.close()

# ret, buffer = cv2.imencode('.jpg', cv2.imread('hinhanh/NV_2027/'+tenha))
# image2 = base64.b64encode(buffer).decode("utf-8")

with open('hinhanh/NV_2027/'+tenha, "rb") as image_file:
    image2 = base64.b64encode(image_file.read()).decode('utf-8')
# Mã hóa dữ liệu byte thành chuỗi base64



print()