import pyodbc

# Kết nối đến cơ sở dữ liệu
conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=112.78.15.3;"
    "Database=ChamCong_Cafe;"
    "uid=ngoi;"
    "pwd=admin123;"
)

# Tạo con trỏ cursor để thực hiện truy vấn
cursor = conn.cursor()
id = 2028

query = f"SELECT * FROM HinhAnh_NhanDien WHERE ID_NhanVien = {id}"
cursor.execute(query)

# Lấy kết quả của truy vấn
rows = cursor.fetchall()

# Kiểm tra xem có hàng nào hay không
if len(rows) > 0:
    print(f"Có hàng trong bảng HinhAnh_NhanDien với ID_NhanVien là {id}.")
    # Thực hiện truy vấn DELETE
    query = f"DELETE FROM HinhAnh_NhanDien WHERE ID_NhanVien = {id}"
    cursor.execute(query)

    # Xác nhận việc xóa hàng
    conn.commit()
    print("Đã xóa các hàng có ID là", id)
else:
    print(f"Không có hàng trong bảng HinhAnh_NhanDien với ID_NhanVien là {id}.")




# Đóng con trỏ và kết nối
cursor.close()
conn.close()




# import pyodbc

# # Kết nối tới cơ sở dữ liệu
# conn = pyodbc.connect(
#     "Driver={SQL Server};"
#     "Server=112.78.15.3;"
#     "Database=ChamCong_Cafe;"
#     "uid=ngoi;"
#     "pwd=admin123;"
# )

# # Tạo con trỏ cursor để thực hiện câu truy vấn
# cursor = conn.cursor()

# # Thực hiện truy vấn xóa
# query = "DELETE FROM HinhAnh_NhanDien"
# cursor.execute(query)

# # Xác nhận thực hiện xóa
# conn.commit()

# # Đóng kết nối và con trỏ cursor
# cursor.close()
# conn.close()

# print("Đã xóa tất cả các hàng trong bảng HinhAnh_NhanDien.")