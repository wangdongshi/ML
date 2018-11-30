# -*- coding:UTF-8 -*-
import sys
import codecs
import re
import pdb

state = ['B','M','E','S'] # 所有分词状态
words = [] # 所有词语
parts = [] # 所有词性
freqs = {} # 词性计数
count = {} # 分词计数
pPi = {} # 先验矩阵(词性)
pA  = {} # 转移矩阵(词性)
pB  = {} # 发射矩阵(词性)
cPi = {} # 先验矩阵(分词)
cA  = {} # 转移矩阵(分词)
cB  = {} # 发射矩阵(分词)

def preprocess(in_name, out_name):
	#打开语料文件
	fin  = codecs.open(in_name, "r")
	#去除不正确表述的元素
	strl = fin.read()
	strl = re.sub("\[","",strl)
	strl = re.sub("]nt","",strl)
	strl = re.sub("]ns","",strl)
	strl = re.sub("]nz","",strl)
	strl = re.sub("]l","",strl)
	strl = re.sub("]i","",strl)
	strl = re.sub("\n", "@", strl)
	strl = re.sub("\s+"," ", strl)
	strl = re.sub("@","\n",strl)
	strl = re.sub(" \n","\n",strl)
	strl = re.sub(" ","@",strl)
	strl = re.sub("\s+","\n",strl)
	strl = re.sub("@"," ",strl)
	#减少词性分类
	strl = re.sub("y","u",strl)
	strl = re.sub("vd","v",strl)
	strl = re.sub("ad","d",strl)
	strl = re.sub("k","n",strl)
	strl = re.sub("j","n",strl)
	strl = re.sub("an","n",strl)
	strl = re.sub("vvn","n",strl)
	strl = re.sub("vn","n",strl)
	strl = re.sub("nr","n",strl)
	strl = re.sub("nz","n",strl)
	strl = re.sub("nx","n",strl)
	strl = re.sub("na","n",strl)
	strl = re.sub("nt","n",strl)
	strl = re.sub("ns","n",strl)
	strl = re.sub("ny","n",strl)
	strl = re.sub("Ag","g",strl)
	strl = re.sub("Bg","g",strl)
	strl = re.sub("Tg","g",strl)
	strl = re.sub("Rg","g",strl)
	strl = re.sub("Vg","g",strl)
	strl = re.sub("Dg","g",strl)
	strl = re.sub("Mg","g",strl)
	strl = re.sub("Ng","g",strl)
	strl = re.sub("Yg","g",strl)
	#写入新的语料文件
	fout = open(out_name, "w")
	fout.write(strl)
	#关闭语料文件
	fout.close()
	fin.close()

def setWordsParts(in_name):
	# set words' set and parts' set.
	with open(in_name, 'r') as f:
		line_num = 0
		for line in f.readlines():
			line_num += 1
			if line_num % 10000 == 0: print('Reading words line: %d' %line_num)
			if line.strip() == '': continue
			else:
				words_in_line = line.strip().split(' ')
				words_num = len(words_in_line)
				if words_num == 0: continue
				for i in range(1, words_num):
					#if i == 0: continue # 根据199801语料库的特点，忽略第一个词
					word = words_in_line[i].split('/')[0]
					if not words_in_line[i].split('/')[1]: print(words_in_line[i])
					part = words_in_line[i].split('/')[1]
					if part not in parts:
						parts.append(part)
					if word not in words:
						words.append(word)

	#打印语料库中的总分词数
	print("len(words) = %s " % (len(words)))

def getList(str):
	out_str = []
	if len(str) == 1:
		out_str.append('S')
	elif len(str) == 2:
		out_str = ['B', 'E']
	else:
		M_num = len(str) - 2
		M_list = ['M'] * M_num
		out_str.append('B')
		out_str.extend(M_list)
		out_str.append('E')
	return out_str
	
def trainCorpusParam(in_name, pi_name, a_name, b_name):
	# initialize all matrix
	for i in state:
		count[i] = 0
		cPi[i] = 0
		cA[i] = {}
		cB[i] = {}
		for j in state:
			cA[i][j] = 0
		for j in words:
			cB[i][j] = 0

	# set other matrix
	with open(in_name, 'r') as f:
		line_num = 0
		for line in f.readlines():
			line = line.strip() #移除头尾空格
			if not line: continue
			else:
				line_num += 1
				if line_num % 10000 == 0: print('Corpus matrix line: %d' %line_num)
				words_in_line = line.strip().split(' ')
				words_num = len(words_in_line)
				word_list = []
				stat_list = []
				for i in range(1, words_num):
					#if i == 0: continue # 根据199801语料库的特点，忽略第一个词
					word = words_in_line[i].split('/')[0]
					word_list.extend(word)
					stat_list.extend(getList(word))
				#pdb.set_trace()
				if len(word_list) != len(stat_list):
					print('[line_num = %d][line = %s]' % (line_num, line))
				else:
					for i in range(len(stat_list)): #按照一行中的字符数循环
						if i == 0:
							cPi[stat_list[0]] += 1 #在一行的开始处把状态置为'B'或'S'
							count[stat_list[0]] += 1 #B或S状态的计数加1
						else:
							cA[stat_list[i-1]][stat_list[i]] += 1 #发生一次转移
							count[stat_list[i]] += 1 #相应状态的计数加1
							if word_list[i] in cB[stat_list[i]] :
								#发射矩阵的相应状态计数加1
								cB[stat_list[i]][word_list[i]] += 1
							else:
								#如发射矩阵某状态中不存在该字符，将该字符对应的发射矩阵项置为0
								cB[stat_list[i]][word_list[i]] = 0.0 
				
		#将分词矩阵信息输出到文件
		start_fp = open(pi_name,'w')
		trans_fp = open(a_name,'w')
		emit_fp = open(b_name,'w')
		#输出先验矩阵(分词)
		for key in cPi:
			#初始矩阵归一化
			cPi[key] = cPi[key] / line_num
		print(cPi, file=start_fp)
		#输出转移矩阵(分词)
		for key in cA:
			for key1 in cA[key]:
				#转移矩阵归一化（除以相应状态总频数）
				cA[key][key1] = cA[key][key1] / count[key]
		print(cA, file=trans_fp)
		#输出发射矩阵(分词)
		for key in cB:
			for word in cB[key]:
				#转移矩阵归一化（除以相应状态总频数）
				cB[key][word] = cB[key][word] / count[key]
		print(cB, file=emit_fp)
		start_fp.close()
		emit_fp.close()
		trans_fp.close()
	
def trainPartsParam(in_name, pi_name, a_name, b_name):
	# calculate words and parts number
	num_parts = len(parts)
	num_words = len(words)

	# initialize all matrix
	for i in parts:
		freqs[i] = 0
		pPi[i] = 0
		pA[i] = {}
		pB[i] = {}
		for j in parts:
			pA[i][j] = 0
		for j in words:
			pB[i][j] = 0

	# set other matrix
	with open(in_name, 'r') as f:
		line_num = 0
		for line in f.readlines():
			if line.strip() == '': continue
			else:
				line_num += 1
				if line_num % 10000 == 0: print('Parts  matrix line: %d' %line_num)
				words_in_line = line.strip().split(' ')
				words_num = len(words_in_line)
				for i in range(1, words_num):
					word = words_in_line[i].split('/')[0]
					part = words_in_line[i].split('/')[1]
					pre_word = words_in_line[i-1].split('/')[0]
					pre_part = words_in_line[i-1].split('/')[1]
					freqs[part] += 1
					if i == 1:
						pPi[part] += 1
					else:
						pA[pre_part][part] += 1
					pB[part][word] += 1

	for i in parts:
		pPi[i] = pPi[i] * 1.0 / line_num
		for j in words:
			pB[i][j] = (pB[i][j] + 1) * 1.0 / (freqs[i] + num_words)
		for j in parts:
			pA[i][j] = (pA[i][j] + 1) * 1.0 / (freqs[i] + num_parts)
	
	# write parts' matrix parameter into file
	f = open(pi_name,'w')
	print(pPi, file = f)
	f.close()
	f = open(a_name,'w')
	print(pA, file = f)
	f.close()
	f = open(b_name,'w')
	print(pB, file = f)
	f.close()

def trainHMM(file_name):
	setWordsParts(file_name)
	trainCorpusParam(file_name, 'cPi.txt', 'cA.txt', 'cB.txt')
	trainPartsParam(file_name, 'pPi.txt', 'pA.txt', 'pB.txt')

if __name__ == "__main__":
	preprocess('199801.utf8', 'rmrb1998.utf8')
	trainHMM('rmrb1998.utf8')
	
	