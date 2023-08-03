from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pyodbc
import numpy as np
import cv2
import base64
import os
# from tqdm import tqdm
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import shutil
import pandas as pd
import dlib
import math
from padding_image import padding
from keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.utils import load_img, save_img, img_to_array
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

   #======== # cua diem danh
# def get_phong_camera(tenphong):
    
#     cursor = conn.cursor()
#     cursor.execute("select RTSP from dbo.CAMERA JOIN dbo.PHONGHOC ON PHONGHOC.MaCamera = CAMERA.MaCamera WHERE TenPhongHoc ="+"'"+tenphong+"'")
#     for row in cursor:
#         link=row[0]
#         # print(link)
#         return link
def get_cam():
    urlcam = ["rtsp://admin:Admin123@mtkhp2408.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true",
              "rtsp://admin:Admin123@mtkhp2420.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true"]
    # urlcam = ["rtsp://admin:Dat1qazxsw2@192.168.6.100:1554/h264_stream"]
    # cursor = conn.cursor()
    # cursor.execute("select RTSP from Camera")
    # for row in cursor:
    #     urlcam.append(row[0]+"/cam/realmonitor?channel=1&subtype=0&unicast=true")
    return urlcam
   #========#

def get_lock_id():
    list_lock_id = ['8624216', 
                    '9006888']
    # list_lock_id = ['9399008']
    return list_lock_id

def clear_thumb():
    for file in os.listdir('hinhanh'):
        path='hinhanh/'+file
        try:
            shutil.rmtree(path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    for file in os.listdir('hinhtrain'):
        path='hinhtrain/'+file
        try:
            shutil.rmtree(path)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
def prepare_data():
    # detectorcnn = MTCNN()
    detectorssd = cv2.dnn.readNetFromCaffe("pre_model/deploy.prototxt","pre_model/res10_300x300_ssd_iter_140000.caffemodel")
    #=-
    # def read_SV(conn):
    #     print("Read_SV")
    #     cursor = conn.cursor()
    #     cursor.execute("select TenHA,MASV,BASE64 from HINHANH_SV")
    #     for row in cursor:
    #         path='./hinhanh/'+'SV_'+f'{row[1]}'
    #         img = chuyen_base64_sang_anh(row[2])
    #         if os.path.exists(path)==False:
    #             os.mkdir(path) 
    #         cv2.imwrite(os.path.join(path , f'{row[0]}'+'.jpg'), img)
    #         cv2.waitKey(0)
    #     print()
    # def read_GV(conn):
    #     print("Read_GV")
    #     cursor = conn.cursor()
    #     cursor.execute("select TenHA,MaGV,BASE64 from HINHANH_GV")
    #     for row in cursor:
    #         path='./hinhanh/'+'GV_'+f'{row[1]}'
    #         img = chuyen_base64_sang_anh(row[2])
    #         if os.path.exists(path)==False:
    #             os.mkdir(path) 
    #         cv2.imwrite(os.path.join(path , f'{row[0]}'+'.jpg'), img)
    #         cv2.waitKey(0)
    #     print()  
    def read_NV(conn):
        print("Tải về dữ liệu nhận diện...")
        cursor = conn.cursor()
        cursor.execute("SELECT TenHA, ID_NhanVien, HA_BASE64 FROM HinhAnh_NhanDien WHERE HA_BASE64 <> 'None'")
        # results = cursor.fetchall()
        # for row in tqdm(results):
        for row in cursor:
            path='./hinhanh/'+'NV_'+f'{row[1]}'
            img = chuyen_base64_sang_anh(row[2])
            if os.path.exists(path)==False:
                os.mkdir(path) 
            cv2.imwrite(os.path.join(path , f'{row[0]}'), img)
            cv2.waitKey(0)
            
        cursor.close()
        #---------------------------------------------------------------------
        # COPY các thư mục trong "hinhlaytucamera" vào "hinhtrain"
        # for folder in os.listdir('hinhlaytucamera'):
        #     duong_dan_thu_muc = os.path.join('hinhlaytucamera', folder)
        #     if os.path.isdir(duong_dan_thu_muc):
        #         duong_dan_moi = os.path.join('hinhtrain', folder)
        #         shutil.copytree(duong_dan_thu_muc, duong_dan_moi)

        # print("Đã copy dữ liệu từ 'hinhlaytucamera' vào 'hinhtrain'")
        
        #---------------------------------------------------------------------
        
        # Kiểm tra các dữ liệu có trong database nhưng k có trong thư mục hinhlaytucamera
        cursor = conn.cursor()
        cursor.execute("SELECT Path FROM HinhAnh_NhanDien")

        # Lấy và in ra các giá trị cột "Path"
        rows = cursor.fetchall()
        for row in rows:
            path = row.Path
            if path!=None:
                if not os.path.exists(path):
                    # Xóa hàng không tồn tại trong cơ sở dữ liệu
                    delete_query = f"DELETE FROM HinhAnh_NhanDien WHERE Path = '{path}'"
                    cursor.execute(delete_query)
                    conn.commit()
                    print(f"{path} sẽ bị xóa trong database do không tồn tại")
        cursor.close()
        conn.close()
            
#####################################################################################################  
    def chuyen_base64_sang_anh(anh_base64):
        try:
            anh_base64 = np.frombuffer(base64.b64decode(anh_base64), dtype=np.uint8)
            anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
        except:
            return "chuyen fail"
        return anh_base64



    # read_SV(conn)
    # read_GV(conn)
    read_NV(conn)
    anh=1
    for file in os.listdir('hinhanh'):
        if os.path.exists('hinhtrain/'+file)==False:
            os.mkdir('hinhtrain/'+file) 
        fullpath='hinhanh/'+file
        full_des_train_path='hinhtrain/'+file
        for img_name in os.listdir(fullpath):
            pixels = pyplot.imread(fullpath+'/'+img_name)
            # faces = detectorcnn.detect_faces(pixels)
            # face=faces[0]
            # x, y, width, height = face['box']
            base_img = pixels.copy()
            h = base_img.shape[0]
            w = base_img.shape[1]
            original_size = base_img.shape
            target_size = (300, 300)
            img = cv2.resize(pixels, target_size)
            aspect_ratio_x = (original_size[1] / target_size[1])
            aspect_ratio_y = (original_size[0] / target_size[0])
            imageBlob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
            detectorssd.setInput(imageBlob)
            detections = detectorssd.forward()
            column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
            detections_df = pd.DataFrame(detections[0][0], columns = column_labels)
            # if detections_df==None:
            #     continue
            #0: background, 1: face
            detections_df = detections_df[detections_df['is_face'] == 1]
            detections_df = detections_df[detections_df['confidence'] >= 0.5]
            if detections_df.empty:
                continue
            detections_df['left'] = (detections_df['left'] * 300).astype(int)
            detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
            detections_df['right'] = (detections_df['right'] * 300).astype(int)
            detections_df['top'] = (detections_df['top'] * 300).astype(int)
            for i, instance in detections_df.iterrows():
                confidence_score = str(round(100*instance["confidence"], 2))+" %"
                left = instance["left"]; right = instance["right"]
                bottom = instance["bottom"]; top = instance["top"]
                if top < 0: top = 0                        
                if left < 0: left = 0                        
                if bottom > h: bottom = h  
                if right > w: right = w
                crop_img = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), 
                                    int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
                # gey = dlib.rectangle(int(left*aspect_ratio_x), int(top*aspect_ratio_y), int(right*aspect_ratio_x), int(bottom*aspect_ratio_y)) 
                # crop_img = pixels[y:y+height,x:x+width]
                
                crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)  

                crop_img = padding(crop_img,target_size=(224,224))

                # crop_img=cv2.resize(crop_img,(224,224))
            # crop_img = pixels[y:y+height,x:x+width]
            # crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)  
            # crop_img=cv2.resize(crop_img,(224,224))

            cv2.imwrite(os.path.join(full_des_train_path , img_name), crop_img)
            # print("so anh ",anh)
            anh+=1
    print(f'Có {anh-1} ảnh trong database.')
    print('Đã chuẩn bị xong lấy dữ liệu nhận diện\n')

clear_thumb()
prepare_data()