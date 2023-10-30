import os, cv2, time
import numpy as np

def load_images_in_folders(root_folder):
    image_data = {}  # Dictionary để lưu trữ dữ liệu ảnh

    # Ghi lại thời gian bắt đầu
    start_time = time.time()
    i = 1
    # Duyệt qua các thư mục trong thư mục gốc
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)

        # Kiểm tra xem folder_path có phải là một thư mục và tồn tại không
        if os.path.isdir(folder_path):
            folder_images = {}  # Dictionary để lưu trữ dữ liệu ảnh trong mỗi thư mục

            # Duyệt qua các tệp hình ảnh trong thư mục
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Kiểm tra xem image_path có phải là tệp hình ảnh và tồn tại không
                if os.path.isfile(image_path) and image_name.endswith(('.jpg', '.jpeg', '.png')):
                    # Sử dụng OpenCV để đọc hình ảnh và chuyển đổi thành mảng NumPy
                    image = cv2.imread(image_path)
                    if image is not None:
                        folder_images[image_name] = image
                        print(i)
                        i+=1

            # Thêm dictionary của thư mục vào dictionary tổng
            image_data[folder_name] = folder_images

    # Ghi lại thời gian kết thúc và tính thời gian chạy của hàm
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Thời gian chạy hàm: {:.2f} giây".format(elapsed_time))

    return image_data

# Sử dụng hàm để đọc ảnh từ thư mục "hinhtrain"
image_data = load_images_in_folders("D:/Code/FaceID/hinhtrain")
print()
# Bây giờ, image_data chứa dữ liệu ảnh với cấu trúc: {tên thư mục: {tên ảnh: mảng NumPy}}
