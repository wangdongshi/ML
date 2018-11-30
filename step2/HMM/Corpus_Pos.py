#!/usr/bin/python
#-*-coding:utf-8
import os
import sys
import pdb
import jieba

def load_model(f_name):
	ifp = open(f_name, 'rb')
	return eval(ifp.read())

def viterbi(states, obs, start_p, trans_p, emit_p):
	#pdb.set_trace()
	V = [{}] #tabular
	path = {}
	for y in states: #init
		V[0][y] = start_p[y] * emit_p[y].get(obs[0],1e-20)
		path[y] = [y]
	for t in range(1,len(obs)):
		V.append({})
		newpath = {}
		for y in states:
			#pdb.set_trace()
			l = [(V[t-1][y0] * trans_p[y0].get(y,1e-20) * emit_p[y].get(obs[t],1e-20), y0) for y0 in states if V[t-1][y0]>0]
			if len(l) != 0 :
				(prob, state) = max(l)
				V[t][y] = prob
				newpath[y] = path[state] + [y]
			# 这里之前的代码感觉有问题，概率不能从路径中间截断为0
			#else :
			#	prob = 0.0
			#	state = y
			#V[t][y] = prob
			#newpath[y] = path[state] + [y]
		path = newpath
	(prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
	return (prob, path[state])

def cut(sentence):
	prob, pos_list =  viterbi(('B','M','E','S'), sentence, corpus_start, corpus_trans, corpus_emit)
	return (prob, pos_list)

def part(sentence):
	prob, pos_list =  viterbi(tuple(part_start.keys()), sentence, part_start, part_trans, part_emit)
	return (prob, pos_list)

if __name__ == "__main__":
	# 加载矩阵
	corpus_start = load_model("cPi.txt")
	corpus_trans = load_model("cA.txt")
	corpus_emit = load_model("cB.txt")
	part_start = load_model("pPi.txt")
	part_trans = load_model("pA.txt")
	part_emit = load_model("pB.txt")
	
	#filename = r'geci.txt'
	#print('《水流众生》歌词分词:')
	filename = r'article.txt'
	print('《人民日报》文章分词')
	sentence_set = []
	with open(filename, 'r') as f:
		jieba_result = ''
		hmm_result = ''
		for line in f.readlines():
			# cut by jieba
			word = jieba.cut(line)
			jieba_result += "/".join(word)
			# cut by HMM model
			line = line.strip()
			prob, pos_list = cut(line)
			cut_word_set = []
			cut_word = ''
			for i in range(0, len(line)):
				#pdb.set_trace()
				hmm_result += line[i]
				cut_word += line[i]
				if pos_list[i] == 'E' or pos_list[i] == 'S':
					hmm_result += '/'
					cut_word_set.append(cut_word)
					cut_word = ''
			hmm_result += '\n'
			sentence_set.append(cut_word_set)
		print("--------------------------")
		print("<结巴分词器>：")
		print(jieba_result)
		print("--------------------------")
		print("<自制分词器>：")
		print(hmm_result)
		print("--------------------------")
	
	print()
	print("<词性标注>")
	print("--------------------------")
	for line in sentence_set:
		# judge part by HMM model
		print(line)
		prob, pos_list = part(line)
		print(pos_list)
		print("--------------------------")
