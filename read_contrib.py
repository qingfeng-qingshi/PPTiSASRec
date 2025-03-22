import pandas as pd

# 读取pkl文件中的字典
popularity_dict = pd.read_pickle('商品流行度.pkl')

# 将字典转换为DataFrame
df_popularity = pd.DataFrame(list(popularity_dict.items()), columns=['关键词ID', '流行度'])

# 保存为csv文件，分隔符为制表符
df_popularity.to_csv('关键词流行度.csv', sep='\t', index=False)
