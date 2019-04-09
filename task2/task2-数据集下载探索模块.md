# 1.IMDB数据集

## 1.1 数据下载
1.下载地址：http://ai.stanford.edu/~amaas/data/sentiment/

2.通过TensorFlow的keras模块下载

	from imdb = keras.datasets.imdb
	# 参数 num_words=10000 会保留训练数据中出现频次在前 10000 位的字词
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

## 1.2 数据说明
IMDB 数据集，其中包含来自互联网电影数据库的 50000 条影评文本。我们将这些影评拆分为训练集（25000 条影评）和测试集（25000 条影评）。训练集和测试集之间达成了平衡，意味着它们包含相同数量的正面和负面影评。

影评分为“正面”或“负面”影评。这是一个二元分类（又称为两类分类）的示例，也是一种重要且广泛适用的机器学习问题。


# 2.THUCnews数据集

## 2.1 数据下载
THUCNews中文数据集：https://pan.baidu.com/s/1hugrfRu 密码：qfud

## 2.2 数据说明
THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。我们在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。

这是一个多分类问题，可以用CNN进行处理。


# 3.评价指标
## 3.1准确率 vs. 召回率

**混淆矩阵**







## Ref：
1.[TensorFlow官方教程：影评文本分类](https://tensorflow.google.cn/tutorials/keras/basic_text_classification)

2.[CSDN博客：电影评论分类-二分类问题（IMDB数据集）](https://blog.csdn.net/Einstellung/article/details/82683652)

3.[CSDN博客：THUCNews学习（CNN模型）](https://blog.csdn.net/qq_42418416/article/details/87973149)

4.[CSDN博客：CNN字符级中文文本分类-基于TensorFlow实现](https://blog.csdn.net/u011439796/article/details/77692621)

5.[GitHub: text-classification-cnn-rnn/cnews_loader.py](https://github.com/gaussic/text-classification-cnn-rnn/blob/master/data/cnews_loader.py)

6.[慕课笔记：机器学习之类别不平衡问题 (2) —— ROC和PR曲线](https://www.imooc.com/article/48072)

7.[博客：机器学习算法中的准确率(Precision)、召回率(Recall)、F值(F-Measure)](https://www.cnblogs.com/Zhi-Z/p/8728168.html)


