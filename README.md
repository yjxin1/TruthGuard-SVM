data文件夹里的news.csv是整个数据集，dataProcess.py将news.csv里的数据集将8：1：1的比例分为了test.csv，train.csv, validation.csv，保存在ChnNews文件夹中。

truth.py训练模型并保存一些相关的转换器，useModel使用flask框架可以通过访问网络地址调用模型并返回预测结果。代码里面都包含注释。