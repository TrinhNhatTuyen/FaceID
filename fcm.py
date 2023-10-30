import requests

def send_fcm_notification(fcm_token, title, body, data=None):
    # Đường dẫn API FCM
    url = 'https://fcm.googleapis.com/fcm/send'
    
    # Đặt thông báo đẩy
    payload = {
        'to': fcm_token,
        'notification': {
            'title': title,
            'body': body
        }
    }
    
    # Thêm dữ liệu tùy chỉnh (nếu có)
    if data:
        payload['data'] = data
    
    # Đặt tiêu đề của thông báo gửi tới FCM
    headers = {
        'Authorization': 'Key=AAAAUM0_kA0:APA91bFq6fvEmRIHZrF4VYTpTcsZHDo_bXvfm1jearG3A8BuNh_pEHtQtYhfGkbDkzsPm_lEwSh-t1LKB50c89wTaEs6N_RAqw7-JhNoUgmA_S5XyNA63E9MICw19QGwCSshw_o_sefG',
        'Content-Type': 'application/json'
    }
    
    # Gửi yêu cầu POST tới API FCM
    response = requests.post(url, json=payload, headers=headers)
    
    # Kiểm tra kết quả gửi
    if response.status_code == 200:
        print('Thông báo đẩy đã được gửi thành công.')
    else:
        print('Gửi thông báo đẩy không thành công. Mã lỗi:', response.status_code)

# Sử dụng hàm send_fcm_notification để gửi thông báo đẩy
if __name__ == "__main__":
    fcm_token = 'fSbWrJ7aTO234suaLfNnSg:APA91bHJG4H3i7h7VTFMUM_IGVOwKX2mpTJrzLmG_fX4LIepvQ2Y2vhdOxqhUusj95DNTia4GyQ6EVpbmxCXS5aERdSwQlh4U3h-YUD0wqotJrp1rrb6KYYo2bYdIBWG_DoIxC1mClVK'  # Thay thế bằng FCM token của thiết bị bạn muốn gửi thông báo đến
    title = 'Tiêu đề thông báo'
    body = 'Nhà đã bị trộm'

    # Thêm dữ liệu tùy chỉnh (nếu có)
    data = {
        'key1': 'value1',
        'key2': 'value2'
    }
    
    send_fcm_notification(fcm_token, title, body, data)
