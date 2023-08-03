import pyodbc
from datetime import datetime
import pytz

conn = pyodbc.connect(
    "Driver={SQL Server};"
    "Server=112.78.15.3;"
    "Database=ChamCong_Cafe;"
    "uid=ngoi;"
    "pwd=admin123;"
)

# Lấy tên các bảng trong cơ sở dữ liệu
cursor = conn.cursor()
tables = cursor.tables(tableType='TABLE')
table_names = [table.table_name for table in tables]
# print(table_names)
for table in table_names:
    # In tên của bảng
    print(f"Table: {table}")
    cursor.execute(f"SELECT * FROM {table}")
    columns = [column[0] for column in cursor.description]
    
    # # # Lấy tên các cột trong bảng
    # columns = cursor.columns(table=table.table_name)
    # column_names = [column.column_name for column in columns]
    
    # In tên các cột
    print(f"Columns: {columns}\n")

# Đóng kết nối
conn.close()