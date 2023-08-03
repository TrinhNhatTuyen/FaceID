import cv2, time, os
from overstepframe import FreshestFrame
url = 'rtsp://admin:1qazxsw2@vinaai.ddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true' # Link stream video RTSP
url = 'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:554/cam/realmonitor?channel=1&subtype=0&unicast=true'
fresh = object()
fresh = FreshestFrame(cv2.VideoCapture(url))
frame = object()
cnt = 0

# Tạo đối tượng VideoCapture với URL stream
cap = cv2.VideoCapture(url)
scale_percent = 60
# Kiểm soát việc ghi và lưu video
record = False
out = None
i=1
timer =time.time()
while True:
    try:
        
        cnt, frame = fresh.read(seqnumber=cnt+1)

        base_img = frame.copy()
        
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        endtimer = time.time()
        fps = 2/(endtimer-timer)
        cv2.putText(frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)
        # frame = cv2.flip(frame, 1)
        cv2.imshow('Stream', cv2.resize(frame, dim))
        
        key = cv2.waitKey(1) & 0xFF
        
        # Ấn phím 'r' để bắt đầu ghi và lưu video
        if key == ord('r') and not record:
            record = True
            # Tạo tên của video mới
            list_name = []
            for i in os.listdir('D:/Code/datatest/cam'):
                try:
                    if i.split('.')[0].isdigit():
                        list_name.append(int(i.split('.')[0]))
                except:
                    continue
            new_video_name = f'smthgelse/{max(list_name)+1}.mp4'
            
            # Tạo đối tượng VideoWriter để ghi video
            out = cv2.VideoWriter(new_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame.shape[1], frame.shape[0]))
            print('Started recording...')
        
        # Ấn phím 'q' để dừng và lưu video
        if key == ord('t') and record:
            record = False
            out.release()
            i+=1
            print('Stopped recording. Video saved as recorded_video.mp4')
        
        # Ghi và lưu video nếu đang trong quá trình ghi
        if record:
            out.write(base_img)

        timer =time.time()
    except:
        continue
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
fresh.release()
cv2.destroyAllWindows()

