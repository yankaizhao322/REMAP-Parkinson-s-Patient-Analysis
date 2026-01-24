# Data source: https://data.bris.ac.uk/data/dataset/9e748876b7bf30218ef7e4ec4d7f026a
import pandas as pd
import os


csv_files = [
    'off_DBS_test.csv',
    'off_Medication_test.csv',
    'on_DBS_test.csv',
    'on_Medication_test.csv'
]

for csv_file_name in csv_files:
    csv_full_path = os.path.join(input_directory, csv_file_name)

    xlsx_file_name = csv_file_name.replace('.csv', '.xlsx')
    xlsx_full_path = os.path.join(input_directory, xlsx_file_name)

    try:
        df = pd.read_csv(csv_full_path)
        
        df.to_excel(xlsx_full_path, index=False)
        
        print(f"成功将 '{csv_file_name}' 转换为 '{xlsx_file_name}'")
    
    except FileNotFoundError:
        print(f"错误：文件 '{csv_full_path}' 未找到。请检查路径和文件名是否正确。")
    except pd.errors.EmptyDataError:
        print(f"警告：文件 '{csv_full_path}' 是空的，无法转换为Excel。")
    except Exception as e:
        print(f"处理文件 '{csv_file_name}' 时发生未知错误：{e}")

print("\n所有文件转换尝试完成。")