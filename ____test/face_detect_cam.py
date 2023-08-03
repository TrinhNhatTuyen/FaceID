import cv2
import pandas as pd

def detect_face(ori_img, detector):
    base_img = ori_img.copy()
    original_size = base_img.shape
    target_size = (300, 300)
    img = cv2.resize(ori_img, target_size)
    # imageBlob = cv2.dnn.blobFromImage(image=img)
    imageBlob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(imageBlob)
    detections = detector.forward()
    column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
    detections_df = pd.DataFrame(detections[0][0], columns=column_labels)
    detections_df = detections_df[detections_df['is_face'] == 1]
    detections_df = detections_df[detections_df['confidence'] >= 0.3]
    detections_df['left'] = (detections_df['left'] * original_size[1]).astype(int)
    detections_df['bottom'] = (detections_df['bottom'] * original_size[0]).astype(int)
    detections_df['right'] = (detections_df['right'] * original_size[1]).astype(int)
    detections_df['top'] = (detections_df['top'] * original_size[0]).astype(int)
    
    # Vẽ hình chữ nhật các khuôn mặt
    for _, row in detections_df.iterrows():
        left = row['left']
        top = row['top']
        right = row['right']
        bottom = row['bottom']
        cv2.rectangle(base_img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)

    return base_img

# Tạo đối tượng detector từ mô hình Caffe
prototxt_path = "pre_model/deploy.prototxt"
caffemodel_path = "pre_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

# Mở video stream
video = 'D:/Code/datatest/cam/6.mp4'
video_capture = cv2.VideoCapture(video)
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('____test/'+video.split('/')[-1].split('.')[0]+'_face_detect.mp4', 
                      cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
while True:
    # Đọc từng frame từ video stream
    ret, frame = video_capture.read()
    if ret:
        # Phát hiện khuôn mặt và vẽ hình chữ nhật
        detected_frame = detect_face(frame, detector)

        # Hiển thị kết quả
        resized_frame = cv2.resize(detected_frame, None, fx=0.6, fy=0.6)
        cv2.imshow('Video', resized_frame)
        out.write(detected_frame)
        # Kiểm tra phím nhấn để thoát vòng lặp
    else:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Giải phóng tài nguyên
video_capture.release()
out.release()
cv2.destroyAllWindows()
