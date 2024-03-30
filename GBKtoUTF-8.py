import pandas as pd

# 读取CSV文件
df = pd.read_csv('jiudian_senti_100k .csv', encoding='GBK')

# 将数据保存为UTF-8编码的CSV文件
df.to_csv('jiudian_senti_100kUTF8.csv', index=False, encoding='utf-8')
