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
	fin  = codecs.open(in_name, "r")
	strl = fin.read()
	'''
	strl = re.sub("\[","[/mq ",strl)
	strl = re.sub("\]/"," ]/",strl)
	strl = re.sub("\n", "@", strl)
	strl = re.sub("\s+"," ", strl)
	strl = re.sub("@","\n",strl)
	strl = re.sub(" \n","\n",strl)
	strl = re.sub(" ","@",strl)
	strl = re.sub("\s+","\n",strl)
	strl = re.sub("@"," ",strl)
	'''
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
	
	fout = open(out_name, "w")
	fout.write(strl)
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
				for i in range(words_num):
					word = words_in_line[i].split('/')[0]
					if not words_in_line[i].split('/')[1]: print(words_in_line[i])
					part = words_in_line[i].split('/')[1]
					if part not in parts:
						parts.append(part)
					if word not in words:
						words.append(word)

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
	cnt = 0
	with open(in_name, 'r') as f:
		line_num = 0
		for line in f.readlines():
			line_num += 1
			if line_num % 10000 == 0: print('Corpus matrix line: %d' %line_num)
			line = line.strip() #移除头尾空格
			if not line: continue
			else:
				cnt += 1
				words_in_line = line.strip().split(' ')
				words_num = len(words_in_line)
				word_list = []
				stat_list = []
				for i in range(0, words_num):
					word = words_in_line[i].split('/')[0]
					word_list.extend(word)
					stat_list.extend(getList(word))
				pdb.set_trace()
				if len(word_list) != len(line_state):
					print('[line_num = %d][line = %s]' % (line_num, line))
				
				'''
				
				for i in range(len(line)):
					if line[i] == " ": continue
					word = line[i].split('/')[0]
					word_list.append(word)
					pre_word = words_in_line[i-1].split('/')[0]
					
				words_in_line = line.strip().split(' ')
				words_num = len(words_in_line)
				
					
				
				word_list = []
				for i in range(len(line)):
					if line[i] == " ":continue
					word_list.append(line[i]) #将一行中的字符串拆成字符装进word_list
				word_set = word_set | set(word_list) #没有使用，set是集合
			
			
			
			
			
				cnt += 1
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
		pPi[i] = pi[i] * 1.0 / cnt
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
				'''	
	
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
	cnt = 0
	with open(in_name, 'r') as f:
		line_num = 0
		for line in f.readlines():
			line_num += 1
			if line_num % 10000 == 0: print('Parts  matrix line: %d' %line_num)
			if line.strip() == '': continue
			else:
				cnt += 1
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
		pPi[i] = pi[i] * 1.0 / cnt
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
	#trainPartsParam(file_name, 'pPi.txt', 'pA.txt', 'pB.txt')
	trainCorpusParam(file_name, 'cPi.txt', 'cA.txt', 'cB.txt')

if __name__ == "__main__":
	#preprocess('1998.txt', 'rmrb1998.txt')
	trainHMM('rmrb1998.txt')
	
	