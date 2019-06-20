#!/usr/bin/python
# -*- coding: UTF-8 -*-

###########################################################################
# 该作业涉及以下几方面的机器学习问题
# 1. 异常值处理
# 2. 样本不均衡的处理方式：欠采样、过采样
# 3. 样本的交叉验证（scikit-learn.cross_validation）
# 4. sklearn中主要模型的构建及调优
# 5. 模型泛化能力的评估方法
###########################################################################

# 导入必要的python包
import pandas as pd
import numpy as np
import calendar
import types
import matplotlib.pyplot as plt

from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics

from sklearn.pipeline import Pipeline
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance

from numpy.core.umath_tests import inner1d
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

class AdaCostClassifier(AdaBoostClassifier):
    #  新定义的代价调整函数
    def _beta(self, y, y_hat):
        res = []
        for i in zip(y, y_hat):
            if i[0] == i[1]:
                res.append(1)   # 正确分类，系数保持不变，按原来的比例减少
            elif i[0] == 1 and i[1] == -1:
                res.append(1.25)  # 在信用卡的情景下，将好人误杀代价应该更大一些，比原来的增加比例要高
            elif i[0] == -1 and i[1] == 1:
                res.append(1)  # 将负例误判为正例，代价不变，按原来的比例增加
            else:
                print(i[0], i[1])
        return np.array(res)
    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1),
                                       axis=0)

        incorrect = y_predict != y

        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        proba = y_predict_proba  # alias for readability
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps

        estimator_weight = (-1. * self.learning_rate
                                * (((n_classes - 1.) / n_classes) *
                                   inner1d(y_coding, np.log(y_predict_proba))))

        # 样本更新的公式，只需要改写这里
        if not iboost == self.n_estimators - 1:
            sample_weight *= np.exp(estimator_weight *
                                    ((sample_weight > 0) |
                                     (estimator_weight < 0)) *
                                    self._beta(y, y_predict))  # 在原来的基础上乘以self._beta(y, y_predict)，即代价调整函数
        return sample_weight, 1., estimator_error

if __name__ == "__main__":

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
	# 对job、marital、education、poutcome四个特征（1，2，3，12）进行OneHot编码
	oe = OneHotEncoder(categorical_features = [1,2,3,7,8,12])
	x = oe.fit_transform(x)
	#print(x.day_of_week)
	#print(x.iloc[0:5, 0:9])
	#print(x)
	y = le.fit_transform(y) # 将y列（Label）从名称字符串转化为数值（同上）
	#print(y)

	###########################################################################
	# 样本分割
	# 划分训练测试数据集，test_size=0.3表示测试数据：训练数据=3:7
	# 一开始采用固定分割（shuffle=False），待选定模型后采用随机抽取方式对模型进行多次验证
	#x_ready, x_test, y_ready, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
	x_ready, x_test, y_ready, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

	# 欠采样、过采样
	rus = RandomUnderSampler(ratio=0.25, random_state=31, replacement=True) # 采用随机欠采样（under sampling）
	x_under, y_under = rus.fit_sample(x_ready, y_ready)
	sos = SMOTE(ratio="auto", kind="svm")
	x_smote, y_smote = sos.fit_sample(x_under, y_under)

	x_train = x_ready
	y_train = y_ready

	pData = pd.DataFrame([y_train], index=["y_train"])
	pData.T.to_csv("y_sampling.csv")

	# 对训练数据进行归一化
	pData = pd.DataFrame(x_train.toarray())
	pData.to_csv("x_sampling1.csv")
	x_train = preprocessing.scale(x_train, with_mean=False)
	pData = pd.DataFrame(x_train.toarray())
	pData.to_csv("x_sampling.csv")

	###########################################################################
	# 训练及测试
	#clf = tree.DecisionTreeClassifier() # 构建DecisionTree模型
	#clf = neighbors.KNeighborsClassifier(5, weights="uniform") # 构建KNN模型
	# 构建SVM模型
	clf1 = svm.SVC(
			kernel="rbf",				# 采用高斯核函数，另外有"linear"
			class_weight={0:1, 1:4},	# "balanced"
			C=0.22
			)
	# 构建ANN模型
	# clf = MLPClassifier(
			# hidden_layer_sizes=[16,16,16],
			# activation="relu",
			# solver ="lbfgs",
			# random_state=3
			# )
	# 构建AdaBoost模型
	# clf = AdaBoostClassifier(
			# tree.DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
			# algorithm="SAMME",
			# n_estimators=200,
			# learning_rate=0.8
			# )
	# 构建xgBoost模型
	clf2 = XGBClassifier(
			learning_rate=0.01,        # 学习率
			n_estimators=10,           # 树的个数-10棵树建立xgboost
			max_depth=4,               # 树的深度
			min_child_weight=1,        # 叶子节点最小权重
			#gamma=0.01,               # 惩罚项中叶子结点个数前的参数
			subsample=1,               # 所有样本建立决策树
			colsample_btree=1,         # 所有特征建立决策树
			scale_pos_weight=1.8,      # 解决样本个数不平衡的问题
			random_state=27,           # 随机数
			#scoring="roc_auc",
			cv=5,
			slient=0
			)
	# 构建RandomForest模型
	clf3 = RandomForestClassifier(
			bootstrap=True,
            class_weight={0:1, 1:4.5},  #"balanced"
            criterion="gini",
            max_depth=8,
			max_features="auto",
			max_leaf_nodes=None,
            min_impurity_decrease=0.0,
			min_impurity_split=None,
            min_samples_leaf=4,
			min_samples_split=10,
            min_weight_fraction_leaf=0.0,
			n_estimators=300,
            oob_score=True,
            random_state=23,
            verbose=0,
			warm_start=False
			)
	# 构建AdaCost模型
	#clf = AdaCostClassifier(n_estimators=100)
	
	# 构建投票器
	#clf = VotingClassifier(estimators=[("svm",clf1),("xgboost",clf2),("rf",clf3)], voting="hard", weights=[1,1,1])
	clf = clf1

	# 模型训练
	#clf1.fit(x_train, y_train)
	#clf2.fit(x_train, y_train)
	#clf3.fit(x_train, y_train)
	clf.fit(x_train, y_train)
	
	# 模型保存
	#joblib.dump(clf, "knn.m")
	# 模型导入
	#clf = joblib.load("knn.m")
	# 对测试数据进行归一化
	x_test = preprocessing.scale(x_test, with_mean=False)
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
	# AUC大于0.8：分类非常准确；AUC在0.6～0.8间：有优化空间；AUC小于0.6：效果较差

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
	
	if "clf2" in vars() and clf2 == clf:
		fig, ax = plt.subplots(figsize=(10,15))
		plot_importance(clf, height=0.5, max_num_features=64, ax=ax)
		plt.show()
