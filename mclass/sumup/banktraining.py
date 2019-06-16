#!/usr/bin/python
# -*- coding: UTF-8 -*-

# 导入必要的python包
import pandas as pd
import calendar
import types
from sklearn.preprocessing import LabelEncoder

# 首先在助教给定数据的基础上，进行数据的预处理：
# 1. 删除duration这一列（关联性太强，给定条件错误）
# 2. 后面都是处理unknown项，存在unknown的属性共有6项，分别是：job、marital、education、default、housing、loan。缺失数据（unknown）的处理方式主要有：删除元组、数据补齐、不处理，以下主要采用删除元组的处理方法，理由如下：
# 3. 首先看default这一列，unknown项非常多，yes只有一项，考虑完全删除此列（对分类来说几乎没有意义）
# 4. 还剩job、marital、education、housing、loan这5列，统计这5项分别为unknown的数据的个数：job(33)、marital(8)、education(144)、housing(100)、loan(100，但与housing的unknown项完全重合)，因此，这些数据总共约有285项，仅占全部3260组数据的8.7%，因此简便起见，考虑全部删除这些数据。
# 5. 整理完毕的数据有3345组，其中标记为yes的数据有354组，可以看出，这是一个样本不均衡的学习问题，yes:no = 1:9


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
dict((v,k) for k,v in enumerate(calendar.month_abbr))
#le1.fit(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
#print(le1.classes_)
x.month = le1.transform(x.month) # 将month列从名称字符串转化为数值（同上）
le2.fit(["mon", "tue", "wed", "thu", "fri"])
print(le2.classes_)
x.day_of_week = le2.transform(x.day_of_week) # 将day_of_week列从名称字符串转化为数值（同上）
y = le.fit_transform(y)

#print(x.month)
#print(type(y))