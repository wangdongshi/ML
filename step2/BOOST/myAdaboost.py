#!/usr/bin/python
#-*-coding:utf-8
import sys
import math

import csv
import pdb

def load_csv(file):
	outFile = open("sample_skeleton_train_plus.csv",'w')
	csvFile = open(file, "r")
	reader = csv.reader(csvFile)
	result = {}
	for i,rows in enumerate(reader):
		#if i < 5000:
		#	row = rows
		#	print(row)
		#else :
		#	break
		#pdb.set_trace()
		if (rows[1] == '1' or rows[2] == '1'):
			print(rows, file=outFile)
	csvFile.close()
	outFile.close()




state_M = 4
word_N = 0

A_dic = {}		#转移矩阵用字典初始化
B_dic = {}		#发射矩阵用字典初始化
Pi_dic = {}		#初始矩阵用字典初始化
Count_dic = {}	#各状态总频数计数用字典初始化
word_set = set()
state_list = ['B','M','E','S']
line_num = -1

PROB_START = "prob_start.py"
PROB_EMIT = "prob_emit.py"
PROB_TRANS = "prob_trans.py"

def removeBom(file):
	BOM = b'\xef\xbb\xbf'
	existBom = lambda s: True if s == BOM else False
	f = open(file, 'rb')
	if existBom(f.read(3)):
		fbody = f.read()
		with open(file, 'wb') as f:
			f.write(fbody)

def init():
	global state_M
	global word_N
	for state in state_list:
		A_dic[state] = {}
		for state1 in state_list:
			A_dic[state][state1] = 0.0 #转移矩阵是一个4x4的矩阵
	for state in state_list:
		Pi_dic[state] = 0.0		#初始矩阵是一个4维的向量
		B_dic[state] = {}
		Count_dic[state] = 0	#计数矩阵是一个4维的向量

def getList(input_str):
	outpout_str = []
	if len(input_str) == 1:
		outpout_str.append('S')
	elif len(input_str) == 2:
		outpout_str = ['B','E']
	else:
		M_num = len(input_str) -2
		M_list = ['M'] * M_num
		outpout_str.append('B')
		outpout_str.extend(M_list)
		outpout_str.append('E')
	return outpout_str

def Output():
	start_fp = open(PROB_START,'w')
	emit_fp = open(PROB_EMIT,'w')
	trans_fp = open(PROB_TRANS,'w')

	#打印语料库中的总分词数
	print("len(word_set) = %s " % (len(word_set)))
	for key in Pi_dic:
		#初始矩阵归一化
		Pi_dic[key] = Pi_dic[key] / line_num
		'''
		if Pi_dic[key] != 0:
			#pdb.set_trace()
			Pi_dic[key] = -1*math.log(Pi_dic[key] * 1.0 / line_num)
		else:
			Pi_dic[key] = -3.14e+100
		'''
	print(Pi_dic, file=start_fp)

	for key in A_dic:
		for key1 in A_dic[key]:
			#转移矩阵归一化（除以相应状态总频数）
			A_dic[key][key1] = A_dic[key][key1] / Count_dic[key]
			'''
			if A_dic[key][key1] != 0:
				#pdb.set_trace()
				A_dic[key][key1] = -1*math.log(A_dic[key][key1] / Count_dic[key])
			else:
				A_dic[key][key1] = -3.14e+100
			'''
	print(A_dic, file=trans_fp)

	for key in B_dic:
		for word in B_dic[key]:
			#转移矩阵归一化（除以相应状态总频数）
			B_dic[key][word] = B_dic[key][word] / Count_dic[key]
			'''
			if B_dic[key][word] != 0:
				#pdb.set_trace()
				B_dic[key][word] = -1*math.log(B_dic[key][word] / Count_dic[key])
			else:
				B_dic[key][word] = -3.14e+100
			'''
	print(B_dic, file=emit_fp)
	
	start_fp.close()
	emit_fp.close()
	trans_fp.close()

def main():
	if len(sys.argv) != 2:
		print("Usage [%s] [input_data] " % (sys.argv[0]))
		sys.exit(0)
	removeBom(sys.argv[1])
	ifp = open(sys.argv[1], 'r')
	init()
	global word_set
	global line_num
	for line in ifp:
		line_num += 1
		if line_num % 10000 == 0:
			print(line_num)

		line = line.strip() #移除头尾空格
		if not line:continue

		word_list = []
		for i in range(len(line)):
			if line[i] == " ":continue
			word_list.append(line[i]) #将一行中的字符串拆成字符装进word_list
		word_set = word_set | set(word_list) #没有使用，set是集合

		lineArr = line.split(" ") #按空格切分一行中的词汇（分词语料库中每行是按空格切分词汇的），split的返回值是列表
		line_state = []
		for item in lineArr:
			line_state.extend(getList(item)) #extend在列表末尾一次性追加另一个序列中的多个值
		if len(word_list) != len(line_state):
			print("[line_num = %d][line = %s]" % (line_num, line))
		else:
			for i in range(len(line_state)): #按照一行中的字符数循环
				if i == 0:
					Pi_dic[line_state[0]] += 1 #在一行的开始处把状态置为'B'或'S'
					Count_dic[line_state[0]] += 1 #B状态的计数加1
				else:
					A_dic[line_state[i-1]][line_state[i]] += 1 #发生一次转移
					Count_dic[line_state[i]] += 1 #相应状态的计数加1
					if word_list[i] in B_dic[line_state[i]] :
						#发射矩阵的相应状态计数加1
						B_dic[line_state[i]][word_list[i]] += 1
					else:
						#如发射矩阵某状态中不存在该字符，将该字符对应的发射矩阵项置为0
						B_dic[line_state[i]][word_list[i]] = 0.0 
	Output()
	ifp.close()

if __name__ == "__main__":
#	main()
	load_csv("/mnt/share/sample_train/sample_skeleton_train.csv")
