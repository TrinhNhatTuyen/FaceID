import cv2, os, argparse
# import matplotlib.pyplot as plt
# import pandas as pd
import time, base64, pyodbc, shutil
import numpy as np
# import base64
# from math import hypot
# from PIL import Image
# from keras.applications.imagenet_utils import preprocess_input
# import matplotlib.pyplot as plt
# import dlib
# from numpy import asarray
# from scipy.spatial.distance import cosine, euclidean
from keras_vggface.utils import preprocess_input
# from prepare_data import preprocess_image, get_cam, get_lock_id
# from os import link, listdir
from facereg_model import loadVggFaceModel
# from eyeblink import predictor_eye,get_blinking_ratio
# from save_atten import conn,create_GV,create_SV
# from save_atten import conn,create_NV
# from overstepframe import FreshestFrame
from padding_image import padding
from face_detect import detect_face
from tensorflow.keras.utils import load_img, save_img, img_to_array
# from remote_lock import get_accesstoken, unlock

# Folder để lưu dữ liệu lấy từ camera
cam_data_path = "hinhlaytucamera"
if not os.path.exists(cam_data_path):
    os.mkdir(cam_data_path)
    
import pyodbc

# Kết nối đến cơ sở dữ liệu
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=112.78.15.3;"
    "Database=ChamCong_Cafe;"
    "uid=ngoi;"
    "pwd=admin123;"
)

model = loadVggFaceModel()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def img_to_bin(path):
    """ Chuyển embedding array của ảnh sang dạng Binary có thể lưu vào database
        Sau này lấy ra và chuyển lại thành array 

    Args:
        path (_type_): đường dẫn của ảnh
    """
    arr = model.predict(preprocess_image(path))
    binary_data = bytes(arr.tobytes())
    return binary_data
    # array_bytes = arr.tobytes()
    # bin = pyodbc.Binary(array_bytes)
    # return bin

########################################################################################################################
def save_image_to_db(ten_ha, id_nhanvien, img_path, bin_array):
    """ Lưu các thông tin và các trường TenHA, ID_NhanVien, Path, Bin_Array

    Args:
        ten_ha (_type_): _description_
        id_nhanvien (_type_): _description_
        img_path (_type_): _description_
        bin_array (_type_): _description_
    """
    cursor = conn.cursor()
    insert_query = "INSERT INTO HinhAnh_NhanDien (TenHA, ID_NhanVien, Path, Bin_Array) VALUES (?, ?, ?, ?)"
    insert_parameters = (ten_ha, id_nhanvien, img_path, bin_array)
    cursor.execute(insert_query, insert_parameters)   
    conn.commit()            
    print("Image saved to database successfully.")

########################################################################################################################
def check_img_exist(ten_ha, id_nhan_vien, path, bin_array):
    """ Kiểm tra xem ảnh có trong database chưa, chưa thì thêm thông tin ảnh vào database

    Args:
        ten_ha (_type_): tên ảnh
        id_nhan_vien (_type_): id nhân viên
        path (_type_): đường dẫn của ảnh
    """
    # Tạo con trỏ cursor để thực hiện truy vấn
    cursor = conn.cursor()

    # Thực hiện truy vấn kiểm tra
    query = f"SELECT * FROM HinhAnh_NhanDien WHERE TenHA = ? AND ID_NhanVien = ? AND Path = ? AND Bin_Array = ?"
    cursor.execute(query,(ten_ha, id_nhan_vien, path, bin_array))

    # Kiểm tra xem có hàng nào thỏa mãn điều kiện hay không
    row = cursor.fetchone()
    if not row:
        save_image_to_db(ten_ha, id_nhan_vien, path, bin_array)

    # Đóng con trỏ và kết nối
    cursor.close()
    
########################################################################################################################
def loop_hinhlaytucamera(directory):
    """ Lặp qua mọi ảnh trong "hinhlaytucamera", với các ảnh có path đúng định dạng thì kiểm tra có trong database chưa
        Nếu chưa thì add vào database
        

    Args:
        directory (_type_): "hinhlaytucamera"
    """
    for folder_name in os.listdir(directory):
        # Kiểm tra định dạng tên thư mục
        if folder_name.startswith("NV_"):
            # Lấy ID_NhanVien từ tên thư mục
            id_nhan_vien = folder_name.split("_")[1]
            # Lặp qua các ảnh trong thư mục
            for img_name in os.listdir(os.path.join(directory,folder_name)):
                # Kiểm tra định dạng tên ảnh
                if img_name.startswith(str(id_nhan_vien)):
                    img_path = f"{directory}/{folder_name}/{img_name}"
                    bin_array = img_to_bin(img_path)
                    check_img_exist(img_name, id_nhan_vien, img_path, bin_array)
                    
########################################################################################################################
def add_face(cam_id=0, imgs=10):
    """ 

    Args:
        cam_id (int, optional): Index của cam cần dùng trong list prepare_data.get_cam(). Defaults to 0.
        imgs (int, optional): _description_. Defaults to 10.
    """
    id_nhanvien=None
    cursor = conn.cursor()
    
    # Kiểm tra xem name đã có trong database chưa   
    # query = f"SELECT ID_NhanVien FROM HinhAnh_NhanDien WHERE TenHA LIKE 'HS_{name}%'"
    # cursor.execute(query)
    # # Nếu có lấy id_nhanvien có sẵn
    # for row in cursor.fetchall():
    #     id_nhanvien = row[0]
    #     break
    
    # Tạo id_nhanvien mới
    if id_nhanvien==None:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(ID_NhanVien) FROM HinhAnh_NhanDien")
        max_id = cursor.fetchone()[0]
        if max_id is None:
            max_id = 0
        id_nhanvien = max_id + 1
        
    #import ssd caffe model
    detector = cv2.dnn.readNetFromCaffe("pre_model/deploy.prototxt","pre_model/res10_300x300_ssd_iter_140000.caffemodel")

    # urlcam = "rtsp://admin:Admin123@mtkhp2420.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true"
    urlcam = "a.mp4"
    # urlcam = get_cam()[cam_id]
    first_frame = None
    thresh = 127
    img_name = 1
    last_save = None
    cap = cv2.VideoCapture(urlcam)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('------------------------------------------------------------------------------------------------')
            break
        ret, frame = cap.read()
        if not ret:
            break
        if first_frame is None:
            first_frame = frame
            continue
        scale_percent = 100 # percent of original size
        try:
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
        except:
            print("Error NoneType!!!")
            continue
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        timer = time.time()
        # subframe
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        first_frame = cv2.threshold(first_frame, thresh, 255, cv2.THRESH_BINARY)[1]

        second_frame = frame
        second_frame = cv2.cvtColor(second_frame, cv2.COLOR_BGR2GRAY)
        second_frame = cv2.threshold(second_frame, thresh, 255, cv2.THRESH_BINARY)[1]

        res = cv2.absdiff(first_frame, second_frame)
        first_frame = frame
        res = res.astype(np.uint8)
        percentage = (np.count_nonzero(res) * 100)/ res.size
        print('FRAM DIFF: ', percentage)
        
        base_img = frame.copy()

        original_size = base_img.shape
        target_size = (300, 300)
        # img = cv2.resize(frame, target_size)
        aspect_ratio_x = (original_size[1] / target_size[1])
        aspect_ratio_y = (original_size[0] / target_size[0])
        # call face detect
        detections_df = detect_face(frame,detector)
        
        if(percentage>=0.1 and len(detections_df==1)):
            print("DIFF IMAGE")
            for i, instance in detections_df.iterrows():
                # confidence_score = str(round(100*instance["confidence"], 2))+" %"
                left = instance["left"]; right = instance["right"]
                bottom = instance["bottom"]; top = instance["top"]
                crop_img = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), 
                                    int(left*aspect_ratio_x):int(right*aspect_ratio_x)]

                crop_img = padding(crop_img,target_size=(224,224))
                
                try:
                    
                    # print("sau blink")
                    cv2.rectangle(frame, (int(left*aspect_ratio_x)-20, int(top*aspect_ratio_y)-20), 
                                    (int(right*aspect_ratio_x)+20, int(bottom*aspect_ratio_y)+20), 
                                    (0, 255, 0), 2) #draw rectangle to main image

                    ten_ha = 'HS_'+str(id_nhanvien)+'_'+str(img_name)+".jpg"
                    if os.path.exists(cam_data_path + '/NV_' + str(id_nhanvien))==False:
                        os.mkdir(cam_data_path + '/NV_' + str(id_nhanvien)) 
                    if last_save==None or time.time()-last_save>1:
                        cv2.imwrite(os.path.join(cam_data_path + '/NV_' + str(id_nhanvien), ten_ha), crop_img)
                        last_save = time.time()
                        img_name+=1
                    
                except ValueError:
                    print("err2",ValueError())

        endtimer = time.time() + 0.01           
        fps = 2/(endtimer-timer)
        cv2.putText(frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0,0 ), 1)

        frame = cv2.resize(frame, ((int)((frame.shape[1])*0.6),(int)((frame.shape[0])*0.6)))
        cv2.imshow("img", frame)
        if img_name>imgs:
            break
        # time.sleep(3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # frame.release()
    cap.release()
    cv2.destroyAllWindows()
    if img_name>5:
        for i in os.listdir(cam_data_path + '/NV_' + str(id_nhanvien)):
            img_path = cam_data_path + '/NV_' + str(id_nhanvien) + '/' + i
            save_image_to_db(ten_ha = i, 
                            id_nhanvien = id_nhanvien, 
                            img_path = img_path,
                            bin_array = img_to_bin(img_path))
    # Thực hiện commit
    conn.commit()
    cursor.close()
    
    # loop_hinhlaytucamera(cam_data_path)
    
    conn.close()

# if __name__ == "__main__":
#     # Tạo đối tượng ArgumentParser
#     parser = argparse.ArgumentParser()

#     # Thêm các đối số dòng lệnh
#     parser.add_argument("--cam_id", type=int, default=0, help="Chọn cam số mấy trong list urlcam")
#     parser.add_argument("--name", type=str, help="Tên của nhân viên")
#     parser.add_argument("--imgs", type=int, default=10, help="Số lượng ảnh cần lấy")

#     # Phân tích đối số dòng lệnh
#     args = parser.parse_args()

#     # Gọi hàm add_face với các tham số từ đối số dòng lệnh
#     add_face(args.cam_id, args.name, args.imgs)


add_face()

# loop_hinhlaytucamera(cam_data_path)