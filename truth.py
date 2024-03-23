import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics

#导入训练集和验证集
datasetTrain = pd.read_csv('data/ChnNews/train.csv', usecols=[0,1])
datasetValidation = pd.read_csv('data/ChnNews/validation.csv', usecols=[0,1])
train_x = datasetTrain['text']
train_y = datasetTrain['label']
vali_x = datasetValidation['text']
vali_y = datasetValidation['label']


#对标签进行编码
encoder = LabelEncoder()
train_y= encoder.fit_transform(train_y)
vali_y = encoder.fit_transform(vali_y)
# 获取类别标签的编码
class_labels = encoder.classes_
print("Class labels:", class_labels)
# 保存类别标签到索引的映射关系
label_to_index = {label: index for index, label in enumerate(encoder.classes_)}
joblib.dump(label_to_index, 'label_to_index.pkl')

# 使用TF-IDF进行特征提取
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # 设置最大特征数
train_features = tfidf_vectorizer.fit_transform(train_x)
vali_features = tfidf_vectorizer.transform(vali_x)

#使用svm
model = svm.SVC(probability=True)
model.fit(train_features, train_y)
joblib.dump(model, './svm_model.pkl')  # 保存模型
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl') #保存TF-IDF 转换器

# 获取每个类别的预测概率
prediction = model.predict(vali_features)
prediction_proba = model.predict_proba(vali_features)

# 打印每个样本的预测概率和实际类别
for i, prob in enumerate(prediction_proba):
    predicted_class = class_labels[np.argmax(prob)]  # 获取概率最高的类别的标签
    actual_class = class_labels[vali_y[i]]  # 获取实际类别的标签
    print("Sample {}: Predicted probabilities: {}, Predicted class: {}, Actual class: {}".format(i+1, prob, predicted_class, actual_class))


