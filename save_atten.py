import pyodbc
from datetime import datetime
import pytz

# def create_SV(conn, id,phong):
#     cursor = conn.cursor()
#     try:
#         cursor.execute('insert into DIEMDANH_T_SV(MaSV,MaPhongHoc) values('+ id +','+phong+');')
#     except Exception as e:
#         print(e)
#     conn.commit()
# def create_GV(conn, id,phong):
#     cursor = conn.cursor()
#     try:
#         cursor.execute('insert into DIEMDANH_T_GV(MaGV,MaPhongHoc) values('+ id +','+phong+');')
#     except Exception as e:
#         print(e)
#     conn.commit()
def create_NV(conn, id, hinhanh):

    cursor = conn.cursor()
   
    try:
        date=str(datetime.today().strftime('%Y-%m-%d'))
        data = "'"+date+"'"
        tz = pytz.timezone('Asia/Ho_Chi_Minh')
        time = str(datetime.now(tz).strftime('%H:%M:%S'))
        time = "'"+time+"'"
        sql = 'insert into ChamCong_Tam( ID_NhanVien, Ngay, Gio, HA_BASE64 ) values('+str(id)+','"'"+date+"'"','+time+','+"'"+hinhanh+"'"+')'
        cursor.execute(sql)
    except Exception as e:
        print(e)
    conn.commit()
    conn.close()
# conn = pyodbc.connect(
#     "Driver={SQL Server};"
#     "Server=112.78.15.3;"
#     "Database=ChamCong_Cafe;"
#     "uid=ngoi;"
#     "pwd=admin123;"
# )
