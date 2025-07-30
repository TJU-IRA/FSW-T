import pandas as pd

# 读取CSV文件
file_path = r'F:\OneDrive\WorkSpace\FSW_Thermal_Correction\src\AdaGrad.csv'  # 替换为你的CSV文件路径
data = pd.read_csv(file_path)

# 对Loss数据进行归一化处理
# 归一化公式：(x - min) / (max - min)
data['Loss'] = (data['Loss'] - data['Loss'].min()) / (data['Loss'].max() - data['Loss'].min())

# 保存为新的CSV文件
new_file_path = r'F:\OneDrive\WorkSpace\FSW_Thermal_Correction\src\normalized_data_AdaGrad.csv'  # 新文件的保存路径
data.to_csv(new_file_path, index=False)

print(f"归一化后的数据已保存到 {new_file_path}")