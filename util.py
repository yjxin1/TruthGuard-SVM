import pandas as pd

# 加载数据集
news_data = pd.read_csv('./data/news.csv')

# 计算每种标签的数量
label_counts = news_data['label'].value_counts()

# 计算各个标签的占比
total_samples = len(news_data)
rumor_ratio = label_counts['谣言'] / total_samples
fact_ratio = label_counts['事实'] / total_samples
undetermined_ratio = label_counts['尚无定论'] / total_samples

# 打印占比
print("谣言的占比:", rumor_ratio)
print("事实的占比:", fact_ratio)
print("尚无定论的占比:", undetermined_ratio)
