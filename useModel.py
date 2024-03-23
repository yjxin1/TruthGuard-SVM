import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
# 加载模型和 TF-IDF 转换器，加载类别标签到索引的映射关系
loaded_model = joblib.load('svm_model.pkl')
loaded_tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
label_to_index = joblib.load('label_to_index.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # 获取 POST 请求中的文本数据
    custom_text = request.json.get('text')

    # 使用 TF-IDF 转换器将自定义文本转换成特征向量
    # 使用加载的模型进行预测
    text_features = loaded_tfidf_vectorizer.transform([custom_text])
    prediction_result = loaded_model.predict_proba(text_features)

    # 获取预测标签和对应的预测概率
    prediction_label = loaded_model.classes_
    prediction_proba = prediction_result[0]
    prediction_result = {label: proba for label, proba in zip(prediction_label, prediction_proba)}

    # 打印预测结果
    # 根据索引查找类别标签
    for label_index, proba in enumerate(prediction_proba):
        label = prediction_label[label_index]
        label_name = list(label_to_index.keys())[list(label_to_index.values()).index(label)]
        print("标签:", label_name, "概率:", proba)
    return jsonify(prediction_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)