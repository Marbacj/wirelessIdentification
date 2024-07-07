# # -*- coding: utf-8 -*-
#
# import os
# import subprocess
#
# # 定义输入文件夹路径和输出文件夹路径
# input_folder_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/scoliosis'
# output_folder_path = 'C:/Users/Uncle/PycharmProjects/wirelessIdentification/data/csv'
# # 确保输出文件夹存在
# os.makedirs(output_folder_path, exist_ok=True)
#
# # 遍历输入文件夹中的所有文件
# for filename in os.listdir(input_folder_path):
#     if filename.endswith('.dat'):
#         dat_file_path = os.path.join(input_folder_path, filename)
#         csv_file_path = os.path.join(output_folder_path, filename.replace('.dat', '.csv'))
#         # 构建csikit命令
#         command = f'csikit --csv {dat_file_path} --csv_dest {csv_file_path}'
#         # 执行命令
#         subprocess.run(command, shell=True)
#
# print("所有文件已成功转换并保存到指定目录！")
