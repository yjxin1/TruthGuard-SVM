import pandas as pd
from sklearn.model_selection import train_test_split

# 读取CSV文件
data = pd.read_csv('./data/news.csv')
print(data.head())

# 提取title和label列
data['title'] = data['title'].fillna(data['content'])  # 如果title为空，使用content列的数据替代空的title列
data = data[data['platform'] != '微博']  # 剔除platform为微博的数据
data = data[[ 'title', 'label']]
data = data.rename(columns={'title': 'text'})  # 将列名title改为text

print(data.head())
# 划分数据集为训练集、验证集和测试集（8:1:1）
train_data, temp_data = train_test_split(data, test_size=0.2,random_state=42)#
validation_data, test_data = train_test_split(temp_data, test_size=0.5,random_state=42)

# 保存数据集到CSV文件
train_data.to_csv('./data/ChnNews/train.csv',index=False)#
validation_data.to_csv('./data/ChnNews/validation.csv',index=False)
test_data.to_csv('./data/ChnNews/test.csv',index=False)
