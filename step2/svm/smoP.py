#coding:utf-8
import csv
import numpy  as np
import random as rnd

#读取数据集
def readData(filename):
    data = []
    with open(filename,'r') as csvfile:
        spamreader = csv.reader(csvfile,delimiter=',')
        for row in spamreader:
            data.append(row)
    return np.array(data)

#产生一个随机数i，a<=i<=b 且 i不等于z
def rand_int(a,b,z):
    i = z
    while i == z:
        i = rnd.randint(a,b)
    return i

#线性核函数
def kernel(x1,x2):
    return np.dot(x1,x2.T)

#对输入xi的预测值与真实输出yi之差
def E(x_k, y_k, w, b):
    return np.sign(np.dot(w.T, x_k.T) + b).astype(int) - y_k

def SVM(x,y):
    N = x.shape[0]
    iters = 0 #迭代次数
    w = []
    b = 0
    while True:
        alpha_prev = np.copy(alpha)
        iters += 1
        for j in range(0,N):
            i = rand_int(0,N-1,j)
            #对应于李老师书中的xi,xj
            x_i, x_j, y_i, y_j = x[i,:], x[j,:], y[i], y[j]
            eta = kernel(x_i, x_i) + kernel(x_j, x_j) - 2 * kernel(x_i, x_j)
            if eta == 0:
                continue
            w = np.dot(alpha * y,x)
            #b求出来实际上是一个区间的值，所以这里取平均
            b = np.mean(y - np.dot(w.T,x.T))
            #计算Ei和Ej
            E_i = E(x_i, y_i, w, b)
            E_j = E(x_j, y_j, w, b)
            #计算a2newunc
            a2old, a1old = alpha[j], alpha[i]
            a2newunc = a2old + float(y_j * (E_i - E_j))/eta
            #计算H和L的值
            L, H = 0, 0
            if y_i != y_j:
                L = max(0,a2old - a1old)
                H = min(C,C + a2old - a1old)
            else:
                L = max(0,a2old + a1old - C)
                H = min(C,a2old + a1old)
            alpha[j] = max(a2newunc, L)
            alpha[j] = min(a2newunc, H)
            alpha[i] = a1old + y_i*y_j * (a2old - alpha[j])
            
        diff = np.linalg.norm(alpha - alpha_prev)#计算误差，实际上是向量的二范数
        if diff < epsilon:
            break
        if iters >= max_iter:
            return
    w = np.dot(alpha * y,x)
    #b = np.mean(y - np.dot(w.T,x.T))
    b = 0
    return w, b

#绘制散点图
def plotFeature(dataMat, labelMat,w ,b):
    plt.figure(figsize=(8, 6), dpi=80)
    x = []
    y = []
    l = []
    for data in dataMat:
        x.append(data[0])
        y.append(data[1])
    for label in labelMat:
        if label > 0:
            l.append('r')
        else:
            l.append('b')
    plt.scatter(x, y, marker = 'o', color = l, s = 15)
    #分割超平面
    x1 = 0
    x2 = 10
    y1 = -b / w[1]
    y2 = (-b - w[0] * x2 ) / w[1]
    lines = plt.plot([x1, x2], [y1, y2])
    lines[0].set_color('green')
    lines[0].set_linewidth(2.0)
    plt.show()

if __name__ == '__main__':
	max_iter = 1000 #最大迭代次数
	C = 1.0  #常量
	epsilon = 0.001  #精确度

	filename = 'Dim2.csv'
	data = readData(filename)
	data = data.astype(float)
	x, y = data[:,1:], data[:,0:1].astype(int) #点及其对应的label
	n = x.shape[0]
	#print(x)
	alpha = np.zeros((n))
	w, b = SVM(x,y)
	print (w, b)
