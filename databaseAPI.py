from flask import Flask, render_template, Response, request, send_file
from flask_cors import CORS, cross_origin
from flask import request
import cv2
import base64
import time
from prepare_data import clear_thumb, prepare_data
from main_facereg import mainVGGregface
import win32gui
import win32ui
import win32con
import win32api
from pynput.keyboard import Key, Controller
from PIL import Image
import pytesseract

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def stop1():
    # init window handle
    for i in range(4):
        window_name = "img"+str(i)
        #hwnd = win32gui.FindWindow(None, window_name)
        hwnds = find_all_windows(window_name)
        # bring each window to the front
        import win32gui
        import win32com.client
        shell = win32com.client.Dispatch("WScript.Shell")
        shell.SendKeys('%')
        for hwnd in hwnds:
            win32gui.SetForegroundWindow(hwnd)
        keyboard = Controller()
        key = "q"

        keyboard.press(key)
        keyboard.release(key)
        keyboard.press(key)
        keyboard.release(key)

def find_all_windows(name):
    result = []

    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd) == name:
            result.append(hwnd)
    win32gui.EnumWindows(winEnumHandler, None)
    return result

@app.route('/train', methods=['POST', 'GET'])
@cross_origin(origin='*')
def train_process():
    clear_thumb()
    prepare_data()
    mainVGGregface()
    return ('Quá trình train hoàn tất')

@app.route('/stop', methods=['POST', 'GET'])
@cross_origin(origin='*')
def stop():
    stop1()
    return ('Tắt server thành công')

    rtsp = request.args.get('rtsp', default = '*', type = str)
    return Response(generate_frames(rtsp),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_data', methods=['POST'])
@cross_origin(origin='*')
def add_data():
    
    return ('Dữ liệu mới')

@app.route('/get_data', methods=['GET'])
@cross_origin(origin='*')
def get_data():
    
    return ('Lấy dữ liệu')

# Start Backend
if __name__ == '__main__':
   app.run()
