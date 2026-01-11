import pandas as pd
import os

# 定义CSV文件所在的目录（如果CSV文件和你的Python脚本在同一个目录下，就留空或'.'）
# 如果不在同一个目录，请修改为实际的目录路径
input_directory = '.' # 示例：'C:/Your/Folder/Containing/CSV_Files'

# 定义要转换的CSV文件列表
csv_files = [
    'off_DBS_test.csv',
    'off_Medication_test.csv',
    'on_DBS_test.csv',
    'on_Medication_test.csv'
]

# 循环处理每个CSV文件
for csv_file_name in csv_files:
    # 构造完整的CSV文件路径
    csv_full_path = os.path.join(input_directory, csv_file_name)

    # 构造对应的Excel文件路径
    # 例如：'off_DBS_test.csv' 会变成 'off_DBS_test.xlsx'
    xlsx_file_name = csv_file_name.replace('.csv', '.xlsx')
    xlsx_full_path = os.path.join(input_directory, xlsx_file_name)

    try:
        # 1. 读取CSV文件
        df = pd.read_csv(csv_full_path)
        
        # 2. 将数据写入Excel文件
        # index=False 意味着不将DataFrame的索引写入Excel的第一列
        df.to_excel(xlsx_full_path, index=False)
        
        print(f"成功将 '{csv_file_name}' 转换为 '{xlsx_file_name}'")
    
    except FileNotFoundError:
        print(f"错误：文件 '{csv_full_path}' 未找到。请检查路径和文件名是否正确。")
    except pd.errors.EmptyDataError:
        print(f"警告：文件 '{csv_full_path}' 是空的，无法转换为Excel。")
    except Exception as e:
        print(f"处理文件 '{csv_file_name}' 时发生未知错误：{e}")

print("\n所有文件转换尝试完成。")