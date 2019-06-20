#!/usr/bin/python
# -*- coding: UTF-8 -*-

###########################################################################
# 作业数据说明：
# bank client data:
#  1 - age (numeric)   年龄
#  2 - job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")   职业
#  3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)   婚姻状况
#  4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")     受教育程度
#  5 - default: has credit in default? (categorical: "no","yes","unknown")   是否有信用卡（几乎无效）
#  6 - housing: has housing loan? (categorical: "no","yes","unknown")   是否有房产
#  7 - loan: has personal loan? (categorical: "no","yes","unknown")   是否有个人贷款
#
# related with the last contact of the current campaign:
#  8 - contact: contact communication type (categorical: "cellular","telephone")   联系沟通类型（用手机还是座机）
#  9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")  最后一次联系的月份
# 10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")  最后一次联系日
# 11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.  最后联系持续时间，以秒为单位（高度相关，应抛弃）
#
# other attributes:
# 12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)   广告期间和此客户端执行的联系人数量
# 13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)  从上一个广告系列上次联系客户端之后经过的天数
# 14 - previous: number of contacts performed before this campaign and for this client (numeric)  此广告系列之前和此客户端之间执行的联系人数量
# 15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")  上一次营销活动的结果（强相关！！！！）
#
# social and economic context attributes
# 16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)  就业变化率
# 17 - cons.price.idx: consumer price index - monthly indicator (numeric)  消费者价格指数   
# 18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)   消费者信心指数     
# 19 - euribor3m: euribor 3 month rate - daily indicator (numeric)   欧元区3月拆借利率  
# 20 - nr.employed: number of employees - quarterly indicator (numeric)   雇员数量  
#
# Output variable (desired target):
# 21 - y - has the client subscribed a term deposit? (binary: "yes","no")
# 判断某人是否认购了定期存款
###########################################################################  

###########################################################################
# 该作业涉及以下几方面的机器学习问题
# 1. 异常值处理
# 2. 样本不均衡的处理方式：欠采样、过采样
# 3. 样本的交叉验证（scikit-learn.cross_validation）
# 4. sklearn中主要模型的构建及调优
# 5. 模型泛化能力的评估方法
# 该程序的运行格式如下：
# >> python banktraining.py XXX YYY ZZZ
# XXX : input csv file name
# YYY : train/test
# ZZZ : classifier type when XXX = train (span:1-9)
###########################################################################

# 导入必要的python包
import sys
import types
import calendar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from numpy.core.umath_tests import inner1d
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE

from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
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

# 网上的一个AdaCost算法实现，python3.5环境运行有误！
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

# 数据加载及预处理
def pretreatment(file="./bankTraining.csv", method="train", sampling=False):
	###########################################################################
	# 首先在助教给定数据的基础上，进行数据的预处理：
	# 1. 删除duration这一列（看数据描述duration与结果的关联性太强）
	# 2. 后面都是处理unknown项，存在unknown的属性共有6项，分别是：job、marital、education、default、housing、loan。缺失数据（unknown）的处理方式主要有：删除元组、数据补齐、不处理，以下主要采用删除元组的处理方法，理由如下：
	# 3. 首先看default这一列，unknown项非常多，yes只有一项，考虑完全删除此列（对分类来说几乎没有意义）
	# 4. 还剩job、marital、education、housing、loan这5列，统计这5项分别为unknown的数据的个数：job(33)、marital(8)、education(144)、housing(100)、loan(100，但与housing的unknown项完全重合)，因此，这些数据总共约有285项，仅占全部3260组数据的8.7%，因此简便起见，考虑全部删除这些数据。
	# 5. month和day_of_week这两个特征看不出与分类的相关性，这两个特征进行OneHot编码容易淹没其它特征，因此也考虑删除。
	# 6. 整理完毕的数据有3345组，其中标记为yes的数据有354组，可以看出，这是一个样本不均衡的学习问题，yes:no = 1:9

	# 数据加载
	# 读入csv数据，第一行为列名
	data = pd.read_csv(file, header=0)
	
	# 数据清理
	# 删除duration/default特征
	data = data.drop(["duration", "default"], axis=1)
	# 删除job、marital、education、housing、loan这5个特征取值为"unknown"的样本
	data = data[~data["job"].isin(["unknown"])]
	data = data[~data["marital"].isin(["unknown"])]
	data = data[~data["education"].isin(["unknown"])]
	data = data[~data["housing"].isin(["unknown"])]
	data = data[~data["loan"].isin(["unknown"])]
	# 删除month/day_of_week特征？（想不出相关性）
	data = data.drop(["month", "day_of_week"], axis=1)
	# 生成预处理后的数据
	data.to_csv("pretreatment.csv") 
	
	# 最后一列是label，所以取第2列到最后一列为x
	x = data.drop(["y"], axis=1)
	# 取列名为"y"的label作为y
	y = data.y
	
	# 特征及标注取值数值化
	# 有序编码
	le = LabelEncoder()
	x.job = le.fit_transform(x.job) # 将job列从名称字符串转化为数值（每个值表示一种Job）
	x.marital = le.fit_transform(x.marital) # 将marital列从名称字符串转化为数值（同上）
	x.education = le.fit_transform(x.education) # 将education列从名称字符串转化为数值（同上）
	x.housing = le.fit_transform(x.housing) # 将housing列从名称字符串转化为数值（同上）
	x.loan = le.fit_transform(x.loan) # 将loan列从名称字符串转化为数值（同上）
	x.contact = le.fit_transform(x.contact) # 将contact列从名称字符串转化为数值（同上）
	x.poutcome = le.fit_transform(x.poutcome) # 将poutcome列从名称字符串转化为数值（同上）
	
	# 建立月份缩写字典，并将month列的月份缩写转化为相应数值
	# monthDict = {}
	# month = x["month"].tolist()
	# dic = dict((v,k) for k,v in enumerate(calendar.month_abbr))
	# for i, j in dic.items():
		# monthDict[i.lower()] = j
	# for i in range(0, len(month)) :
		# month[i] = monthDict[month[i]]
	# x.month = month
	# 建立月份缩写字典，并将day_of_week列的星期缩写转化为相应数值
	# weekDict = {}
	# week = x["day_of_week"].tolist()
	# dic = dict((v,k) for k,v in enumerate(calendar.day_abbr))
	# for i, j in dic.items():
		# weekDict[i.lower()] = j+1
	# for i in range(0, len(week)) :
		# week[i] = weekDict[week[i]]
	# x.day_of_week = week
	
	# 对job、marital、education、poutcome四个特征（1，2，3，10）进行OneHot编码
	#oe = OneHotEncoder(categorical_features = [1,2,3,7,8,12])
	oe = OneHotEncoder(categorical_features = [1,2,3,10])
	x = oe.fit_transform(x)
	#print(x.day_of_week)
	#print(x.iloc[0:5, 0:9])
	#print(x)
	y = le.fit_transform(y) # 将y列（Label）从名称字符串转化为数值（同上）
	#print(y)

	###########################################################################
	# 只用于测试过程的预处理（不需要样本分割）
	if method == "test" :
		x_test = preprocessing.scale(x, with_mean=False)
		y_test = y
		return x_test, x_test, y_test, y_test
	elif method == "train" :
		###########################################################################
		# 样本分割
		# 划分训练数据集和测试数据集，test_size=0.3表示测试数据：训练数据=3:7
		# 一开始采用固定分割（shuffle=False），待选定模型后采用随机抽取方式对模型进行多次验证
		x_ready, x_test, y_ready, y_test = train_test_split(x, y, test_size=0.3, shuffle=True)
		#x_ready, x_test, y_ready, y_test = train_test_split(x, y, test_size=0.3, shuffle=False)

		###########################################################################
		# 样本采样
		if sampling : # 欠采样、过采样
			rus = RandomUnderSampler(ratio=0.2, random_state=31, replacement=True) # 采用随机欠采样（under sampling）
			x_under, y_under = rus.fit_sample(x_ready, y_ready)
			sos = SMOTE(ratio="auto", kind="svm")
			x_smote, y_smote = sos.fit_sample(x_ready, y_ready)
		else : # 不采样
			x_smote, y_smote = x_ready, y_ready

		# 采样后数据
		x_train = x_smote
		y_train = y_smote

		###########################################################################
		# 对数据进行标准化（OneHot编码时不能去均值）
		#pData = pd.DataFrame(x_train.toarray())
		#pData.to_csv("before_sampling1.csv")
		x_train = preprocessing.scale(x_train, with_mean=False) # 训练数据正规化
		x_test  = preprocessing.scale(x_test,  with_mean=False) # 测试数据正规化
		#pData = pd.DataFrame(x_train.toarray()) # 从稀疏矩阵到DataFrame的变换（因为采用了OneHot码）
		#pData.to_csv("after_sampling.csv")
		#pData = pd.DataFrame([y_train], index=["y_train"])
		#pData.T.to_csv("y_sampling.csv")
		
		return x_train, x_test, y_train, y_test
	else :
		return x, x, y, y

# 多种分类模型的构建与训练
def train(x, y, type=7):
	clf1 = tree.DecisionTreeClassifier() # 构建DecisionTree模型
	clf2 = neighbors.KNeighborsClassifier(5, weights="uniform") # 构建KNN模型
	# 构建SVM模型
	# model = svm.SVC(
			# kernel="rbf",				# 采用高斯核函数，另外有"linear"
			# class_weight="balanced"
			# )
	# param = {"C": [1e-2, 4e-1, 3e-1, 2e-1, 1e-1, 1.0], "gamma": [0.1, 0.01, 0.001, 0.0001]}   
	# clf3 = GridSearchCV(model, param, n_jobs = 8, verbose=1, cv=4)
	clf3 = svm.SVC(
			kernel="rbf",				# 采用高斯核函数，另外有"linear"
			class_weight="balanced",	# 手工测试过{0:1, 1:4}
			C=0.2,
			gamma=0.0001
			)
	# 构建ANN模型
	clf4 = MLPClassifier(
			hidden_layer_sizes=[16,16,16],
			activation="relu",
			solver ="lbfgs",
			random_state=3
			)
	# 构建AdaBoost模型
	clf5 = AdaBoostClassifier(
			tree.DecisionTreeClassifier(max_depth=3, min_samples_split=10, min_samples_leaf=5),
			algorithm="SAMME",
			n_estimators=800,
			learning_rate=0.8
			)
	# 构建xgBoost模型
	clf6 = XGBClassifier(
			booster="gbtree",			# gbtree or gbliner
			slient=0,					# 不输出中间过程
			scale_pos_weight=10,		# 当正负样本比例
			n_estimators=100,			# 迭代次数（决策树个数）
			max_depth=6,				# 树的深度
			min_child_weight=0.01,		# 叶子节点最小权重
			subsample=1,				# 每棵树使用的样本比例，1：据所有样本建立决策树，取值范围0.5-1
			colsample_btree=1,			# 每棵树是用的特征比例，1：据所有特征建立决策树，取值范围0.5-1
			learning_rate=0.3,			# 学习率
			gamma=0.01,					# 惩罚项系数，指定节点分裂所需的最小损失函数下降值
			random_state=2,				# 随机数
			scoring="roc_auc",			# 对于分类器可取：accuracy、f1、neg_log_loss、precison、recall、roc_auc
			cv=5
			)
	# 构建RandomForest模型
	# model = RandomForestClassifier(
			# bootstrap=True,				# 是否采用有放回样本的方式
            # class_weight="balanced",	# {0:1, 1:4.5}
            # criterion="gini",			# 度量分裂的标准，可取值范围：mae、mse、gini
			# max_features="auto",		# 寻找最佳分裂点时考虑的属性数目，可取值范围：auto/none、sqrt、log2、表百分数的数值
			# max_leaf_nodes=None,		# 最优优先方式
            # min_impurity_decrease=0.0,	# 节点分裂的阈值
			# min_impurity_split=None,	# 树增长提前结束的阈值
            # min_weight_fraction_leaf=0.0,
            # oob_score=True,				# 交叉验证
            # verbose=0,					# 构建树过程的冗长度
			# warm_start=False,			# True：重新使用之前的结构去拟合样例并且加入更多的估计器
            # random_state=23				# 根据y标签值自动调整权值与输入数据的类频率成反比
			# )
	# param = {"max_depth":[4,5,6,7,8], "min_samples_leaf":[3,5,10,100], "min_samples_split":[10,50,100,200,1000], "n_estimators":[30,100,1000]}   
	# clf7 = GridSearchCV(model, param, n_jobs = 8, verbose=1, cv=4)
	clf7 = RandomForestClassifier(
			bootstrap=True,				# 是否采用有放回样本的方式
            class_weight="balanced",	# "balanced"、"balanced_subsample"、{0:1, 1:4}
            criterion="gini",			# 度量分裂的标准，可取值范围：mae、mse、gini
            max_depth=4,				# 树的最大深度
			max_features="auto",		# 寻找最佳分裂点时考虑的属性数目，可取值范围：auto/none、sqrt、log2、表百分数的数值
			max_leaf_nodes=None,		# 最优优先方式
            min_impurity_decrease=0.0,	# 节点分裂的阈值
			min_impurity_split=None,	# 树增长提前结束的阈值
            min_samples_leaf=3,			# 叶子节点上应有的最少样例数
			min_samples_split=1000,		# 分裂内部节点需要的最少样例数
            min_weight_fraction_leaf=0.0,
			n_estimators=1000,			# 迭代次数
            oob_score=True,				# 交叉验证
            verbose=0,					# 构建树过程的冗长度
			warm_start=False,			# True：重新使用之前的结构去拟合样例并且加入更多的估计器
            random_state=23				# 根据y标签值自动调整权值与输入数据的类频率成反比
			)
	# 构建AdaCost模型（类代码有问题）
	clf8 = AdaCostClassifier(n_estimators=100)
	
	# 构建投票器
	clf9 = VotingClassifier(estimators=[("svm",clf3),("xgboost",clf6),("rf",clf7)], voting="hard", weights=[3,4,3])
	
	# 最终决定采用何种分类器
	clf_array = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9]
	clf = clf_array[type-1]
	
	# 打印所采用分类器名称
	print("----------------------------------------------------------")
	if clf == clf1 : 	print("Use DecisionTree Classifier.")
	elif clf == clf2 :	print("Use K Neighbors Classifier.")
	elif clf == clf3 :	print("Use SVM Classifier.")
	elif clf == clf4 :	print("Use Multi-Layer NN Classifier.")
	elif clf == clf5 :	print("Use AdaBoost Classifier.")
	elif clf == clf6 :	print("Use XGBoost Classifier.")
	elif clf == clf7 :	print("Use Random Forest Classifier.")
	elif clf == clf8 :	print("Use AdaCost Classifier.")
	elif clf == clf9 :	print("Use Voting Classifier.")
	print("----------------------------------------------------------")
		
	# 模型训练
	clf.fit(x, y)
	# 如果分类器是Grid Search，则打印检索出的最优参数
	if (isinstance(clf, GridSearchCV)) :
		print("The best parameters are %s ." % clf.best_params_)
	
	# 采用xgBoost时打印特征重要性
	if clf6 == clf:
		fig, ax = plt.subplots(figsize=(10,15))
		plot_importance(clf, height=0.5, max_num_features=64, ax=ax)
		plt.show()
		
	# 模型保存
	joblib.dump(clf, "model.m")
	
	# 返回模型
	return clf
	
if __name__ == "__main__":

	###########################################################################
	# 数据加载及预处理
	print(len(sys.argv))
	for i in range(len(sys.argv)) :
		print(sys.argv[i])
	if len(sys.argv) == 1 :
		x_train, x_test, y_train, y_test = pretreatment()
	elif len(sys.argv) == 2 :
		x_train, x_test, y_train, y_test = pretreatment(sys.argv[1])
	elif len(sys.argv) == 3 or len(sys.argv) == 4 :
		x_train, x_test, y_train, y_test = pretreatment(sys.argv[1], sys.argv[2])
	else :
		print("Parameter size is error!")
	
	###########################################################################
	# 模型构建、训练
	# （分类器类型）
	# Type1	： DecisionTree
	# Type2	： K Neighbors
	# Type3	： SVM
	# Type4	： Multi-Layer NN
	# Type5	： AdaBoost
	# Type6	： XGBoost
	# Type7	： Random Forest
	# Type8	： AdaCost
	# Type9	： Voting
	if (len(sys.argv) >= 3) and (sys.argv[2] == "test") :
		clf = joblib.load("model.m") # 模型加载
	else :
		if len(sys.argv) == 4 :
			clf = train(x_train, y_train, int(sys.argv[3])) # 构建模型
		else :
			clf = train(x_train, y_train) # 构建模型
		
	###########################################################################
	# 模型预测
	#x_test, y_test = x_train, y_train
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
	# precision又称查准率（精度），是衡量检索系统和检索者拒绝非相关信息的能力
	# recall又称查全率（召回率），是衡量检索系统和检索者检出相关信息的能力
	# 因为查准与查全是矛盾的，因此又有F1作为综合的判定指标
	# F1值是Precision和Recall的加权调和平均值，当F1较高时则比较说明模型比较理想
	# AUC是ROC曲线下面积，正常情况AUC应大于0.5
	# AUC大于0.8：分类非常准确；AUC在0.6～0.8间：有优化空间；AUC小于0.6：效果较差
	target_names = ["no", "yes"]
	print (metrics.classification_report(y_test, y_pred, target_names=target_names))
	print("----------------------------------------------------------")
	print ("Positive case discriminant evaluation result :")
	print ("accuracy")
	print (metrics.accuracy_score(y_test, y_pred))
	print ("precision_score")
	print (metrics.precision_score(y_test, y_pred))
	print ("recall_score")
	print (metrics.recall_score(y_test, y_pred))
	fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
	print ("f1_score")
	print (metrics.f1_score(y_test, y_pred))	
	print ("roc_auc")
	print (metrics.auc(fpr, tpr))
	print("----------------------------------------------------------")
	
