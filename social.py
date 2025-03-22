import pickle

import pandas as pd
import numpy as np
# 读取数据
df = pd.read_csv('./data/social_relationships.csv', sep='\t')

# 计算每个用户的 Followee 数量
followee_counts = df.groupby('Follower')['Followee'].count()
# 计算平均值
average_followee_count = followee_counts.mean()
# 找到最大的 Followee 数量
max_followee_count = followee_counts.max()
# 计算每个用户的影响力
influence_scores = {user_id: np.log(count + 1) / np.log(max_followee_count + 1) for user_id, count in followee_counts.items()}
#将influence_scores作为二进制文件保存到磁盘
# 将 influence_scores 作为 pkl 文件保存到磁盘
with open('./每个用户的影响力.pkl', 'wb') as file:
    pickle.dump(influence_scores, file)
# 将影响力值乘以1000并取整数部分
influence_scores = {user_id: int(influence * 1000) for user_id, influence in influence_scores.items()}
# 更新 df 中的 Weight 字段
df['Weight'] = df['Follower'].map(influence_scores)
df.to_csv('./data/social_relationships_influence2.csv', sep='\t', index=False)
# 输出结果
print(influence_scores)
print(f"每个用户的平均 Followee 数量: {average_followee_count}")
