# ID_HinhAnh, TenHA, HA_BASE64, ID_NhanVien, Path, Bin_Array
import numpy as np
import pyodbc, os, sys
import base64, struct
from PIL import Image
from numpy import asarray
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine, euclidean
from tensorflow.keras.utils import load_img, save_img, img_to_array
sys.path.append('D:/Code/FaceID/')
from facereg_model import loadVggFaceModel
model = loadVggFaceModel()
# Kết nối đến cơ sở dữ liệu
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=112.78.15.3;"
    "Database=ChamCong_Cafe;"
    "uid=ngoi;"
    "pwd=admin123;"
)

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

path = 'hinhlaytucamera/NV_2028/2028_1.jpg'
arr1 = model.predict(preprocess_image(path))
binary_data = bytes(arr1.tobytes())


arr2 = np.array(struct.unpack('f' * (len(binary_data) // 4), binary_data), dtype=np.float32)

# Định hình lại mảng thành kích thước mong muốn (1, 2048)
arr2 = arr2.reshape(1, 2048)


similarity = cosine(arr1[0], arr2[0])
minratio=1
if(similarity)<minratio: 
    minratio=similarity
    
eucli = euclidean(arr1[0], arr2[0])

print(f"Similarity: {similarity}  Eucli: {eucli}")