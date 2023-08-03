import cv2, os, argparse, shutil
import time, base64, pyodbc
import numpy as np
from PIL import Image
from prepare_data import preprocess_image, get_cam, get_lock_id
from facereg_model import loadVggFaceModel
from face_detect import detect_face


# Folder để lưu dữ liệu lấy từ camera
cam_data_path = "hinhlaytucamera/"
if not os.path.exists(cam_data_path):
    os.mkdir(cam_data_path)
    
for file in os.listdir('hinhlaytucamera'):
    path='hinhlaytucamera/'+file
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))
        
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=112.78.15.3;"
    "Database=ChamCong_Cafe;"
    "uid=ngoi;"
    "pwd=admin123;"
)

def save_image_to_db(ten_ha, id_nhanvien, img_base64):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO HinhAnh_NhanDien (TenHA, ID_NhanVien, HA_BASE64) VALUES (?, ?, ?)", (ten_ha, id_nhanvien, img_base64))              
    print("Image saved to database successfully.")
    

def add_face(id_nhanvien = None, cam_id=0, imgs=10):
    cursor = conn.cursor()
    max_img_index = 1
    # Kiểm tra xem id_nhanvien đã có trong database chưa
    # Tìm max_index để đặt tên cho ảnh nếu đã có vài ảnh trước đó 
    sql_query = "SELECT TenHA FROM HinhAnh_NhanDien WHERE ID_NhanVien = ?"
    cursor.execute(sql_query, (id_nhanvien,))
    if cursor.fetchone():
        for row in cursor.fetchone():
            ten_ha = row.TenHA
            try:
                index = ten_ha.split("_")[1].split(".")[0]
                if index > max_img_index:
                    max_img_index = index
            except:
                continue
    # Nếu id_nhanvien chưa có trong database, hoặc id_nhanvien không được truyền tham số    
    else:
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
    img_index = max_img_index
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

                try:
                    
                    # print("sau blink")
                    cv2.rectangle(frame, (int(left*aspect_ratio_x)-20, int(top*aspect_ratio_y)-20), 
                                    (int(right*aspect_ratio_x)+20, int(bottom*aspect_ratio_y)+20), 
                                    (0, 255, 0), 2) #draw rectangle to main image
                    ###############################################################################################
                    ten_ha = str(id_nhanvien)+'_'+str(img_index)+".jpg"
                    if os.path.exists(cam_data_path + 'NV_' + str(id_nhanvien))==False:
                        os.mkdir(cam_data_path + 'NV_' + str(id_nhanvien)) 
                    cv2.imwrite(os.path.join(cam_data_path + 'NV_' + str(id_nhanvien), ten_ha), base_img)
                    img_index+=1
                    ###############################################################################################
                except ValueError:
                    print("err",ValueError())

        endtimer = time.time() + 0.01           
        fps = 2/(endtimer-timer)
        cv2.putText(frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0,0 ), 1)

        
        frame = cv2.resize(frame, ((int)((frame.shape[1])*0.6),(int)((frame.shape[0])*0.6)))
        cv2.imshow("img", frame)
        if (img_index-max_img_index) > imgs:
            break
        # time.sleep(3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # frame.release()
    cap.release()
    cv2.destroyAllWindows()
    ###############################################################################################
    for i in os.listdir(cam_data_path + 'NV_' + str(id_nhanvien)):
        img_path = cam_data_path + 'NV_' + str(id_nhanvien) + '/' + i
        image = Image.open(img_path)
        img_base64 = base64.b64encode(image.tobytes()).decode("utf-8")
        # save_image_to_db(ten_ha = i, 
        #                  id_nhanvien = id_nhanvien, 
        #                  img_base64 = img_base64)
        
        
        # cursor.execute("INSERT INTO HinhAnh_NhanDien (TenHA, ID_NhanVien, HA_BASE64) VALUES (?, ?, ?)", (ten_ha, id_nhanvien, img_base64))
        insert_query = "INSERT INTO HinhAnh_NhanDien (TenHA, ID_NhanVien, HA_BASE64) VALUES (?, ?, ?)"
        insert_parameters = (i, id_nhanvien, img_base64)
        cursor.execute(insert_query, insert_parameters)               
        print("Image saved to database successfully.")
        
    ###############################################################################################
    # Thực hiện commit
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Tạo đối tượng ArgumentParser
    parser = argparse.ArgumentParser()

    # Thêm các đối số dòng lệnh
    parser.add_argument("--id_nhanvien", type=int, help="ID của nhân viên")
    parser.add_argument("--cam_id", type=int, default=0, help="Chọn cam số mấy trong list urlcam")
    parser.add_argument("--imgs", type=int, default=10, help="Số lượng ảnh cần lấy")

    # Phân tích đối số dòng lệnh
    args = parser.parse_args()

    # Gọi hàm add_face với các tham số từ đối số dòng lệnh
    add_face(args.id_nhanvien, args.cam_id, args.imgs)

# add_face(name='VTV')