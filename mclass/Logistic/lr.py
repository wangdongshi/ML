#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 导入必要的python包
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

# 读入csv数据，第一行为列名
data = pd.read_csv("data.csv", header=0)
# 第一列是id,第二列是label，所以取第3列到最后一列为x
x = data.iloc[:, 2:]
# 取列名为"diagnosis"的label作为y
y = data.diagnosis
# 将label "M"、"B"分别转化成1、0
le = LabelEncoder()
y = le.fit_transform(y)

# 划分训练测试数据集，test_size=0.3表示测试数据：训练数据=3:7
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
# 对训练数据进行归一化
x_train = preprocessing.scale(x_train)

# 构建logistic regression模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# 模型训练
clf.fit(x_train, y_train)
# 模型保存
joblib.dump(clf, "lr.m")
# 模型导入
clf = joblib.load("lr.m")
# 对测试数据进行归一化
x_test = preprocessing.scale(x_test)
# 模型预测
y_pred = clf.predict(x_test)

# 模型效果评估
target_names = ['Benign', 'Malignant']
print( metrics.classification_report(y_test, y_pred, target_names=target_names) )
print( "accuracy" )
print( metrics.accuracy_score(y_test, y_pred) )
print( "precision_score" )
print( metrics.precision_score(y_test, y_pred) )
print( "recall_score" )
print( metrics.recall_score(y_test, y_pred) )
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
print( "auc" )
print( metrics.auc(fpr, tpr) )
print( "f1_score" )
print( metrics.f1_score(y_test, y_pred) )
