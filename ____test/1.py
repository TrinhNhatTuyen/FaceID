import pyodbc

# Kết nối tới cơ sở dữ liệu SQL Server
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=112.78.15.3;"
    "Database=ChamCong_Cafe;"
    "uid=ngoi;"
    "pwd=admin123;"
)

# Tạo con trỏ cursor để thực hiện câu truy vấn
cursor = conn.cursor()

# Thực hiện câu truy vấn SELECT để lấy các hàng có cột "Bin_Array" là None
# query = "SELECT TenHA, ID_NhanVien FROM HinhAnh_NhanDien WHERE Bin_Array IS NULL"
query = "SELECT Ngay, ID_NhanVien, Gio FROM ChamCong_Tam"
cursor.execute(query)

# Lấy tất cả các hàng từ kết quả truy vấn
rows = cursor.fetchall()

# In ra TenHA và ID_NhanVien của các hàng có Bin_Array là None
for row in rows:
    print("Ngay:", row.Ngay)
    print("Gio:", row.Gio)
    print("ID_NhanVien:", row.ID_NhanVien)
    
    print()

# Đóng kết nối và con trỏ cursor
cursor.close()
conn.close()