#coding=utf-8
import jieba
import jieba.posseg as psg

result = "《水流众生》歌词分词:\n"
filename = r"../lyrics.txt"
 
with open(filename, 'r') as f:
	print("《水流众生》歌词分词词性：")
	for line in f.readlines():
		word = psg.cut(line)
		for w in word:
			print(w.word, w.flag)
		word = jieba.cut(line)
		result += "/".join(word)
	print("--------------------------")
print(result)
