# ID_HinhAnh, TenHA, HA_BASE64, ID_NhanVien, Path, Bin_Array

import pyodbc
conn = pyodbc.connect("Driver={SQL Server};"
                      "Server=112.78.15.3;"
                      "Database=ChamCong_Cafe;"
                      "uid=ngoi;"
                      "pwd=admin123;")

# Tạo đối tượng cursor để thực hiện truy vấn
cursor = conn.cursor()

# Truy vấn dữ liệu
# cursor.execute("SELECT TenHA, ID_NhanVien FROM HinhAnh_NhanDien")
cursor.execute("SELECT TenHA, ID_NhanVien FROM HinhAnh_NhanDien WHERE Bin_Array IS NOT NULL")
# cursor.execute("SELECT TenHA, ID_NhanVien FROM HinhAnh_NhanDien WHERE Bin_Array IS NULL")
# Lấy kết quả truy vấn
results = cursor.fetchall()

# In kết quả
for row in results:
    print("TenHA:", row.TenHA, "ID_NhanVien:", row.ID_NhanVien)#, "ID_HinhAnh:", row[0])

# Đóng kết nối
conn.close()
