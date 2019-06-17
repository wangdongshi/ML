#!/usr/bin/python
# -*- coding: UTF-8 -*-
###########################################################################
# 该作业涉及以下几方面的机器学习问题
# 1. 异常值处理
# 2. 样本不均衡的处理方式：欠采样、过采样
# 3. 样本的交叉验证（scikit-learn.cross_validation）
# 4. sklearn中主要模型的比较
# 5. 模型泛化能力的评估方法
###########################################################################

# 导入必要的python包
import pandas as pd
import numpy as np
import calendar
import types

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier

###########################################################################
# 下面的处理在程序外进行（直接修改csv文件）
#
# 首先在助教给定数据的基础上，进行数据的预处理：
# 1. 删除duration这一列（关联性太强，给定条件错误）
# 2. 后面都是处理unknown项，存在unknown的属性共有6项，分别是：job、marital、education、default、housing、loan。缺失数据（unknown）的处理方式主要有：删除元组、数据补齐、不处理，以下主要采用删除元组的处理方法，理由如下：
# 3. 首先看default这一列，unknown项非常多，yes只有一项，考虑完全删除此列（对分类来说几乎没有意义）
# 4. 还剩job、marital、education、housing、loan这5列，统计这5项分别为unknown的数据的个数：job(33)、marital(8)、education(144)、housing(100)、loan(100，但与housing的unknown项完全重合)，因此，这些数据总共约有285项，仅占全部3260组数据的8.7%，因此简便起见，考虑全部删除这些数据。
# 5. 整理完毕的数据有3345组，其中标记为yes的数据有354组，可以看出，这是一个样本不均衡的学习问题，yes:no = 1:9
#
###########################################################################

# 数据加载与数值化
# 读入csv数据，第一行为列名
data = pd.read_csv("bankTraining_1.csv", header=0)
# 第一列是label，所以取第2列到最后一列为x
x = data.iloc[:, 1:]
# 取列名为"y"的label作为y
y = data.y
# 将label "yes"、"no"分别转化成1、0
le = LabelEncoder()
le1 = LabelEncoder()
le2 = LabelEncoder()
x.job = le.fit_transform(x.job) # 将job列从名称字符串转化为数值（每个值表示一种Job）
x.marital = le.fit_transform(x.marital) # 将marital列从名称字符串转化为数值（同上）
x.education = le.fit_transform(x.education) # 将education列从名称字符串转化为数值（同上）
x.housing = le.fit_transform(x.housing) # 将housing列从名称字符串转化为数值（同上）
x.loan = le.fit_transform(x.loan) # 将loan列从名称字符串转化为数值（同上）
x.contact = le.fit_transform(x.contact) # 将contact列从名称字符串转化为数值（同上）
x.poutcome = le.fit_transform(x.poutcome) # 将poutcome列从名称字符串转化为数值（同上）
# 建立月份缩写字典，并将month列的月份缩写转化为相应数值
monthDict = {}
month = x["month"].tolist()
dic = dict((v,k) for k,v in enumerate(calendar.month_abbr))
for i, j in dic.items():
    monthDict[i.lower()] = j
for i in range(0, len(month)) :
	month[i] = monthDict[month[i]]
x.month = month
#print(x.month)
# 建立月份缩写字典，并将day_of_week列的星期缩写转化为相应数值
weekDict = {}
week = x["day_of_week"].tolist()
dic = dict((v,k) for k,v in enumerate(calendar.day_abbr))
for i, j in dic.items():
    weekDict[i.lower()] = j+1
for i in range(0, len(week)) :
	week[i] = weekDict[week[i]]
x.day_of_week = week
#print(x.day_of_week)
#print(x.iloc[0:5, 0:9])
#print(x)
y = le.fit_transform(y) # 将y列（Label）从名称字符串转化为数值（同上）
#print(y)

###########################################################################
# 样本分割
# 划分训练测试数据集，test_size=0.3表示测试数据：训练数据=3:7
# 一开始采用固定分割（shuffle=False），待选定模型后采用随机抽取方式对模型进行多次验证
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=False)
# 对训练数据进行归一化
x_train = preprocessing.scale(x_train)

###########################################################################
# 训练及测试
#clf = tree.DecisionTreeClassifier() # 构建tree模型
#clf = neighbors.KNeighborsClassifier(5, weights="uniform") # 构建knn模型
clf = svm.SVC(kernel="rbf", class_weight="balanced", C=0.2) # 构建svm模型(采用高斯核函数)
#clf = svm.SVC(kernel="linear", class_weight="balanced", C=1.0) # 构建svm模型(采用线性核函数)
#clf = MLPClassifier(hidden_layer_sizes=[16,16], activation='relu', solver ='lbfgs', random_state=3) # 构建ANN模型
#clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5), algorithm="SAMME", n_estimators=200, learning_rate=0.8) # 构建AdaBoost模型
# 模型训练
clf.fit(x_train, y_train)
# 模型保存
joblib.dump(clf, "knn.m")
# 模型导入
clf = joblib.load("knn.m")
# 对测试数据进行归一化
x_test = preprocessing.scale(x_test)
# 模型预测
y_pred = clf.predict(x_test)
# 将预测结果及原来的样本标注输出成文件
result = pd.DataFrame([y_test,y_pred], index=["test", "pred"])
result.T.to_csv("result.csv")
#print(type(y_pred))
#print(type(y_test))
#print(result.T)

###########################################################################
# 模型效果评估
# 精度(precision) = yes预测为yes的个数(TP)/所有被预测为yes的个数(TP+FP)
# 召回率(recall) = yes预测为yes的个数(TP)/样本中实际的yes个数(TP+FN)
# F1 = 2*精度*召回率/(精度+召回率)
# precision又称查准率，是衡量检索系统和检索者拒绝非相关信息的能力
# recall又称查全率，是衡量检索系统和检索者检出相关信息的能力
# 因为查准与查全是矛盾的，因此又有F1作为综合的判定指标
# F1值是Precision和Recall的加权调和平均值，当F1较高时则比较说明模型比较理想
# AUC是ROC曲线下面积，正常情况AUC应大于0.5
# AUC大于0.8：分类非常准确；AUC在0.6～0.8间：有优化空间；AUC小于0.6：效果较差。
target_names = ["no", "yes"]
print (metrics.classification_report(y_test, y_pred, target_names=target_names))
print ("accuracy")
print (metrics.accuracy_score(y_test, y_pred))
print ("precision_score")
print (metrics.precision_score(y_test, y_pred))
print ("recall_score")
print (metrics.recall_score(y_test, y_pred))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
print ("auc")
print (metrics.auc(fpr, tpr))
print ("f1_score")
print (metrics.f1_score(y_test, y_pred))

