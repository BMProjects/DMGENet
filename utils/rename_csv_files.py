import os


def rename_csv_files(directory):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 检查文件是否为 CSV 文件
        if filename.endswith('.csv'):
            # 分割文件名
            parts = filename.split('_')

            # 检查是否有足够的部分
            if len(parts) >= 9:  # 至少要有9个部分才能删除前六个和最后三个
                # 删除前六个部分
                parts = parts[6:]

                # 删除最后三个部分
                if len(parts) >= 3:
                    parts = parts[:-3]

                # 重新组合成新的文件名
                new_filename = '_'.join(parts) + '.csv'

                # 构建完整的文件路径
                old_filepath = os.path.join(directory, filename)
                new_filepath = os.path.join(directory, new_filename)

                try:
                    # 重命名文件
                    os.rename(old_filepath, new_filepath)
                    print(f"文件 {filename} 已成功重命名为 {new_filename}")
                except FileNotFoundError:
                    print(f"文件 {filename} 未找到")
                except PermissionError:
                    print(f"没有权限重命名文件 {filename}")
                except Exception as e:
                    print(f"发生错误: {e}")
            else:
                print(f"文件 {filename} 的部分不足九个，无需重命名。")

# 指定目录
directory = '../dataset/Delhi/2023/'
# 调用函数
rename_csv_files(directory)