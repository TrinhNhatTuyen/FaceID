import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time, struct, pyodbc, threading
import numpy as np
import base64
from math import hypot
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import dlib
import os
import datetime
from numpy import asarray
from scipy.spatial.distance import cosine, euclidean
from keras_vggface.utils import preprocess_input
from prepare_data import preprocess_image, get_cam, get_lock_id
from os import link, listdir
from facereg_model import loadVggFaceModel
from eyeblink import predictor_eye,get_blinking_ratio
# from save_atten import conn,create_GV,create_SV
from save_atten import create_NV
from overstepframe import FreshestFrame
from padding_image import padding
from face_detect import detect_face
from remote_lock import get_accesstoken, unlock
from put_alert_firebase import PutAlertThread
# protoPath = join(dirname(_file_), "deploy.prototxt")
# modelPath = join(dirname(_file_), "res10_300x300_ssd_iter_140000.caffemodel")
# def mainVGGregface():
# real-time frame video!
# modelfacenet = InceptionResNetV1()
# modelfacenet.load_weights('facenet_weights.h5')
model = loadVggFaceModel()
access_token = get_accesstoken()
detector = cv2.dnn.readNetFromCaffe("pre_model/deploy.prototxt","pre_model/res10_300x300_ssd_iter_140000.caffemodel")
# Lưu lại ngày bắt đầu chạy code để biết khi nào sẽ lấy access token mới
if datetime.date.today().day > 27:
    start_date = 1
else:
    start_date = datetime.date.today().day
#------------------------------------------------------------------------------------------------------------------------#
def chuyen_base64_sang_anh(anh_base64):
        try:
            anh_base64 = np.frombuffer(base64.b64decode(anh_base64), dtype=np.uint8)
            anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
        except:
            return "chuyen fail"
        return anh_base64
    
def img_to_bin(path):
    """ Chuyển embedding array của ảnh sang dạng Binary có thể lưu vào database
        Sau này lấy ra và chuyển lại thành array 

    Args:
        path (str): đường dẫn của ảnh
    """
    arr = model.predict(preprocess_image(path))
    binary_data = bytes(arr.tobytes())
    return binary_data

def bin2array(bin):
    """ Chuyển đổi dữ liệu nhị phân thành mảng numpy

    Args:
        bin (_type_): Giá trị Binary của ảnh
    """
    arr = np.array(struct.unpack('f' * (len(bin) // 4), bin), dtype=np.float32)
    # Định hình lại mảng thành kích thước mong muốn (1, 2048)
    arr = arr.reshape(1, 2048)
    return arr

def get_embeddinglist_from_database():
    conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=112.78.15.3;"
    "Database=ChamCong_Cafe;"
    "uid=ngoi;"
    "pwd=admin123;"
    )

    cursor = conn.cursor()
    dict_employees = dict()
    
    cursor.execute("SELECT * FROM HinhAnh_NhanDien")
    results = cursor.fetchall()

    for result in results:
        try:
            if result.TenHA!=None:
                dict_employees[result.TenHA] = bin2array(result.Bin_Array)
        except:
            print('Ảnh mới !!! : ', result.TenHA)
            continue

    cursor.close()
    conn.close()  
     
    return dict_employees

def check_new_img():
    """ 
        Kiểm tra nếu có ảnh mới thêm từ web chấm công:
           #1: Lưu ảnh mới vào "hinhanh" 
           #2: Lưu ảnh mới vào "hinhtrain"
           #3: Thêm thông tin trường Bin_Array
           #4: Cập nhật dict "employees"
    """
    conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=112.78.15.3;"
    "Database=ChamCong_Cafe;"
    "uid=ngoi;"
    "pwd=admin123;"
    )
    
    # TH thêm ảnh qua web chấm công: thêm thông tin cho trường Bin_Array của các ảnh đó nếu chưa có
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM HinhAnh_NhanDien WHERE Bin_Array IS NULL")
    rows = cursor.fetchall()
    # for row in rows:
    #     path='./hinhanh/'+'NV_'+f'{row[1]}'
    #     img = chuyen_base64_sang_anh(row[2])
    #     if os.path.exists(path)==False:
    #         os.mkdir(path) 
    #     cv2.imwrite(os.path.join(path , f'{row[0]}'), img)
    #     cv2.waitKey(0)
    # print()

    if rows:
        print("Đang thêm ảnh mới...")
        for row in rows:
#--------------------------- #1: Lưu ảnh mới vào "hinhanh" ---------------------------------#
            path='./hinhanh/'+'NV_'+f'{row.ID_NhanVien}'
            img = chuyen_base64_sang_anh(row.HA_BASE64)
            img_path = os.path.join(path , f'{row.TenHA}')
            if os.path.exists(path)==False:
                os.mkdir(path) 
            cv2.imwrite(img_path, img)
            cv2.waitKey(0)
            
#--------------------------- #2: Lưu ảnh mới vào "hinhtrain" -------------------------------#
            file = f'NV_{row.ID_NhanVien}'
            if os.path.exists('hinhtrain/'+file)==False:
                    os.mkdir('hinhtrain/'+file) 
            fullpath='hinhanh/'+file
            full_des_train_path='hinhtrain/'+file
            for img_name in os.listdir(fullpath):
                # Nếu là hình cũ thì bỏ qua
                if os.path.exists(os.path.join(full_des_train_path,img_name)):
                    continue
                # Nếu là hình mới chưa cắt mặt, chưa có trong hinhtrain
                pixels = plt.imread(fullpath+'/'+img_name)
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
                imageBlob = cv2.dnn.blobFromImage(image = img)
                detector.setInput(imageBlob)
                detections = detector.forward()
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

                    crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
                    crop_img = padding(crop_img,target_size=(224,224))
                
                cv2.imwrite(os.path.join(full_des_train_path , img_name), crop_img)
            
#--------------------------- #3: Thêm thông tin trường Bin_Array ---------------------------#
            try:
                bin = img_to_bin(f'hinhtrain/NV_{row.ID_NhanVien}/{row.TenHA}')
            except FileNotFoundError:
                print(f'ERROR line 202 "main_facereg.py": Ảnh {row.TenHA} không detect được mặt !!!')
                continue
            
            # bin = img_to_bin(f'hinhtrain/NV_{row.ID_NhanVien}/{row.TenHA}')
            cursor2 = conn.cursor()
            cursor2.execute("UPDATE HinhAnh_NhanDien SET Bin_Array = ? WHERE TenHA = ?",(bin, row.TenHA))
            print('Đã thêm trường Bin_Array cho ảnh ', row.TenHA)
            
#--------------------------- #4: Cập nhật dict "employees" ----------------------      -----#
            employees[row.TenHA] = bin2array(bin)
            
        # Thêm thông tin cho trường Bin_Array xong
        conn.commit()
        cursor2.close()        
    ### TH xóa ảnh qua web chấm công: web chấm công tự xóa thông tin ảnh trong database ###
    cursor.close()
    conn.close()


employees = get_embeddinglist_from_database()
check_new_img()
print("employee representations retrieved successfully")
# check_window_exist = [None,None,None,None,None,None,None,None,None,None,None,None,None]
def mainVGGregface():
    global employees, access_token
    last_unlock = time.time()
    next_unlock = 10
    phong='RoomIT.01'
#------------------------------------------------------------------------------------------------------------------------#
    check_new_img()
    
#------------------------------------------------------------------------------------------------------------------------#

    # define Cosine or use from keras
    # link to ip camera  
    # video_capture = cv2.VideoCapture("rtsp://admin:L284ED54@42.112.115.79:554/cam/realmonitor?channel=1&subtype=0&unicast=true")
    #---------

  #phong hoc=========  
    # maphong = get_ma_phong(phong)
    # linkcamera = get_phong_camera(phong)
    # linkcamera='rtsp://admin:1qazxsw2@42.115.188.149:554/cam/realmonitor?channel=1&subtype=0&unicast=true'
    # if(len(linkcamera) > 1):
    #     linkcamera = str(linkcamera)
    # else:
    #     linkcamera= int(linkcamera)
    #----------
    linkcamera = get_cam()
    list_lock_id = get_lock_id()
    
    fresh=[object(),object(),object(),object(),object(),object(),object(),object(),object(),object()] 
    for AA in range(len(linkcamera)):
        fresh[AA]=FreshestFrame(cv2.VideoCapture(linkcamera[AA]))
        #fresh[AA]=FreshestFrame(cv2.VideoCapture(0))
        # fresh[AA]=FreshestFrame(cv2.VideoCapture("rtsp://admin:Admin123@mtkhp2408.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true"))
 
    frame =[object(),object(),object(),object()]
    second_frame =[None,None,None,None,None,None,None,None,None,None,None,None,None]
    cnt = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    # detect move
    first_frame = [None,None,None,None,None,None,None,None,None,None,None,None,None]
    thresh = 127
    warning=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    alert=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
   # count_frame=[int(0),int(0),int(0),int(0)]
    try:
        while True:
            
            t = datetime.datetime.now()
            # if (t.hour % 2 == 0) and t.minute==0 and t.second<2:
            if (t.hour % 2 == 0) and t.minute==0 and t.second<2:
                if datetime.date.today().day == start_date and t.hour==0:
                    access_token = get_accesstoken()
                raise Exception("Restarting")
            
            f = []
            for CC in range(len(linkcamera)):
                homeid = linkcamera[CC].split('mtkhp')[1].split('.')[0]
                
                # print("INDEX I",CC)
                cnt[CC],frame[CC] = fresh[CC].read(seqnumber=cnt[CC]+1, timeout=5)
                if not cnt[CC]:
                    print(f"Timeout, can't read new frame of cam {CC}!")
                    raise Exception()
                
                # if(count_frame[CC]%3==0):
                if first_frame[CC] is None:
                    first_frame[CC] = frame[CC]
                    print('1')
                    continue
                scale_percent = 100 # percent of original size
                width = int(frame[CC].shape[1] * scale_percent / 100)
                height = int(frame[CC].shape[0] * scale_percent / 100)
                
                if cv2.waitKey(1) & 0xFF == ord('e'):
                    raise Exception("Raise Error")
                
                dim = (width, height)
                frame[CC] = cv2.resize(frame[CC], dim, interpolation = cv2.INTER_AREA)
                
                timer =time.time()
                # subframe
                first_frame[CC] = cv2.cvtColor(first_frame[CC], cv2.COLOR_BGR2GRAY)
                first_frame[CC] = cv2.threshold(first_frame[CC], thresh, 255, cv2.THRESH_BINARY)[1]

                second_frame[CC]=frame[CC]
                second_frame[CC] = cv2.cvtColor(second_frame[CC], cv2.COLOR_BGR2GRAY)
                second_frame[CC] = cv2.threshold(second_frame[CC], thresh, 255, cv2.THRESH_BINARY)[1]

                res = cv2.absdiff(first_frame[CC], second_frame[CC])
                first_frame[CC] = frame[CC]
                res = res.astype(np.uint8)
                percentage = (np.count_nonzero(res) * 100)/ res.size
                # print('FRAM DIFF: ', percentage)
                
                # #
                base_img = frame[CC].copy()

                original_size = base_img.shape
                target_size = (300, 300)
                # img = cv2.resize(frame, target_size)
                aspect_ratio_x = (original_size[1] / target_size[1])
                aspect_ratio_y = (original_size[0] / target_size[0])
                
                # Call Face Detect
                detections_df = detect_face(frame[CC],detector)
                
                if(percentage>=0.1):

                    # Nếu không có ai trong khung hình
                    if len(detections_df)==0:
                        if warning[CC]!=0:
                            put_alert_thread = PutAlertThread(homeid, '0')
                            put_alert_thread.start()
                            warning[CC]=0
                    else:
                        for _, instance in detections_df.iterrows():
                            # confidence_score = str(round(100*instance["confidence"], 2))+" %"
                            left = instance["left"]; right = instance["right"]
                            bottom = instance["bottom"]; top = instance["top"]
                            detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y),
                                                    int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
                            # detected_face1 = base_img[int(top*aspect_ratio_y)-20:int(bottom*aspect_ratio_y)+20,
                            #                         int(left*aspect_ratio_x)-20:int(right*aspect_ratio_x)+20]
                            top1 = int(top*aspect_ratio_y)-20 if (int(top*aspect_ratio_y)-20)>=0 else 0
                            bottom1 = int(bottom*aspect_ratio_y)+20 if (int(top*aspect_ratio_y)+20)<original_size[1] else original_size[1]
                            left1 = int(left*aspect_ratio_x)-20 if (int(left*aspect_ratio_x)-20)>=0 else int(left*aspect_ratio_x)-20
                            right1 = int(right*aspect_ratio_x)+20 if (int(right*aspect_ratio_x)+20)<original_size[0] else original_size[0]
                            detected_face1 = base_img[top1:bottom1, left1:right1]
                        
                            ## lanmark face eyes
                            gey = dlib.rectangle(int(left*aspect_ratio_x), int(top*aspect_ratio_y), 
                                                int(right*aspect_ratio_x), int(bottom*aspect_ratio_y))
                            landmarks = predictor_eye(frame[CC], gey)
                            # Get eyes positions
                            left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks,frame[CC])
                            right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks,frame[CC])
                            blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                            cv2.rectangle(frame[CC], (int(left*aspect_ratio_x)-20, int(top*aspect_ratio_y)-20),
                                        (int(right*aspect_ratio_x)+20, int(bottom*aspect_ratio_y)+20),
                                        (0, 0, 255), 1) #draw rectangle to main image
                            ##
                            print("BLINK RATIO", blinking_ratio)
                            try:
                                
                                print("sau blink")
                                cv2.rectangle(frame[CC], (int(left*aspect_ratio_x)-20, int(top*aspect_ratio_y)-20), (int(right*aspect_ratio_x)+20, int(bottom*aspect_ratio_y)+20), (0, 255, 0), 2) #draw rectangle to main image
                                detected_face = frame[CC][int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)] #crop detected face
                                
                                # #--padding
                                detected_face=padding(detected_face,target_size=(224,224))
                                #--end padding
                            
                                img_pixels = Image.fromarray(detected_face)
                                img_pixels = np.expand_dims(img_pixels, axis = 0)
                                samples = asarray(img_pixels, 'float32')
                                # prepare the face for the model, e.g. center pixels
                                    ####
                                samples = preprocess_input(samples, version=2)
                                samples_fn = asarray(img_pixels, 'float32')
                                samples_fn = preprocess_input(samples_fn, version=2)
                                captured_representation = model.predict(samples_fn)
                                ####
                                minratio=1
                                name='noname'
                                for i in employees:
                                    representation = employees[i]
                                    similarity = cosine(representation[0], captured_representation[0])
                                    # print(euclidean(employees[CC], captured_representation))
                                    if(similarity)<minratio: 
                                        minratio=similarity
                                        name=i
                                eucli = euclidean(employees[name][0], captured_representation[0])
                                if(minratio < 0.38 and eucli <90):
                                    print('>>> Employee Detected ---- Minratio:',minratio, 'Eucli',eucli)
                                    
                                    cv2.putText(frame[CC],'ID: '+str(name).split('_')[1] , 
                                                (int(left*aspect_ratio_x)-20,int(top*aspect_ratio_y)-25),
                                                cv2.FONT_HERSHEY_SIMPLEX,1 , (0, 0,255 ), 2)

                                    # Mở khóa nếu đúng người
                                    # access_token = get_accesstoken()
                                    if (time.time()-last_unlock) > next_unlock:
                                        lock_id = list_lock_id[CC]
                                        last_unlock = time.time()
                                        # unlock(access_token=access_token, lock_id=lock_id)
                                        
                                        # Lưu ảnh người mở khóa
                                        unlock_img = frame[CC].copy()
                                        cv2.putText(unlock_img, "UNLOCK", (960,50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), thickness=3)
                                        cv2.putText(unlock_img, f"Match img: {name}", (20,135), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                        cv2.putText(unlock_img, f"Minratio:  {minratio:.5f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                        cv2.putText(unlock_img, f"Eucli:     {eucli:.5f}", (20,205), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                        
                                        current_time = datetime.datetime.now()
                                        time_string = current_time.strftime("%H%M%S_%d%m%Y")
                                        cv2.imwrite(f"Unlock/{time_string}.jpg", unlock_img)
                                        
                                        # MỞ KHÓA
                                        unlock(access_token=access_token, lock_id=lock_id, client_id='2ce5129232f74cc2ac89e24cdd04ec65')
                                    
                                    # Lưu ảnh người mở khóa vào database
                                    try :
                                        detected_face1 = cv2.resize(detected_face1, (224, 224)) #resize to 224x224
                                        ret, buffer = cv2.imencode('.jpg', detected_face1)
                                        image = base64.b64encode(buffer).decode("utf-8")
                                        conn = pyodbc.connect("Driver={SQL Server};"
                                                            "Server=112.78.15.3;"
                                                            "Database=ChamCong_Cafe;"
                                                            "uid=ngoi;"
                                                            "pwd=admin123;")                      
                                        create_NV(conn,str(name).split('_')[1],image.replace("'", ""))
                                        print("\n SAVE Employee WITH ID = : ", str(name).split('_')[1],"min",minratio, 'eu', eucli)
                                    except Exception as e:
                                        print('ERROR while save unlock_face to db: ',e)
                                                                
                                else:
                                    # print('Minratio:',minratio, 'Eucli',eucli)
                                    warning[CC]+=1
                                    if(warning[CC]>30):
                                        cv2.putText(frame[CC], "WARNING", (200,50),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                                        put_alert_thread = PutAlertThread(homeid, '1')
                                        put_alert_thread.start()
                                    print('Unknown Human',"min:",minratio,"eu",eucli)
                                    cv2.putText(frame[CC],'unknown human', (int(left*aspect_ratio_x)-20,int(top*aspect_ratio_y)-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255 ), 2)
                                    
                                    # Lưu ảnh người lạ
                                    stranger_img = frame[CC].copy()
                                    cv2.putText(stranger_img, f"Minratio:  {minratio:.5f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                    cv2.putText(stranger_img, f"Eucli:     {eucli:.5f}", (20,205), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                    
                                    current_time = datetime.datetime.now()
                                    time_string = current_time.strftime("%H%M%S_%d%m%Y")
                                    cv2.imwrite(f"Unknown_Human/{time_string}.jpg", stranger_img)
                                    cv2.imwrite(f"Unknow_Human_NoBox/{time_string}.jpg", base_img)
                                    
                            except ValueError:
                                print("err1",ValueError())

                endtimer = time.time() + 0.01
                fps = 2/(endtimer-timer)
                f.append(fps)
                if len(f)==2:
                    print("FPS 0: {:.2f}".format(f[0]), "   ---   FPS 1: {:.2f}".format(f[1]))
                    
                cv2.putText(frame[CC], "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)
                frame[CC] = cv2.resize(frame[CC], ((int)((frame[CC].shape[1])*0.6),(int)((frame[CC].shape[0])*0.6)))
                cv2.imshow("img"+str(CC), frame[CC])
                # check_window_exist[CC] = True

            # Kiểm tra xem các camera hiện đủ chưa
            # for CC in range(len(linkcamera)):
            #     if check_window_exist[CC] == True:
            #         if not cv2.getWindowProperty("img"+str(CC), cv2.WND_PROP_VISIBLE) > 0:
            #             raise Exception("Missing camera!")        

            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(e)
        for CC in range(len(linkcamera)):
            fresh[CC].release()
            
    for CC in range(len(linkcamera)):
        fresh[CC].release()
        
    cv2.destroyAllWindows()
    print('Restarting...')
    
#----------------------------------------------------------------------------------------------------#     

while True:
    try:
        mainVGGregface()
    except:
        print("Lỗi nằm ngoài vòng WHILE !!!")
        pass