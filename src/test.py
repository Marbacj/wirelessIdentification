# -*- coding: utf-8 -*-

import pandas as pd

# 加载数据
df = pd.read_csv('/Users/bachmar/wiridt/wirelessIdentification/output.csv')

# 查看数据的基本信息
print(df.head())
print(df.info())