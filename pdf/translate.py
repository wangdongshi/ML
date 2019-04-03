#!/usr/bin/python
import urllib.request
import urllib.error
#import urllib2
import sys

URL = 'http://translate.google.cn/translate_a/t?client=t&hl=zh-CN&sl=en&tl=zh-CN&ie=UTF-8&oe=UTF-8&multires=1&prev=btn&ssel=0&tsel=0&sc=1'
HEADER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.89 Safari/537.1'
ERROR_INFO = 'error : no input'

def translate(input):
	search_url = URL + '&text=' + input
	req = urllib.request.Request(search_url)
	req.add_header('User-Agent', HEADER_AGENT)
	r = urllib.request.urlopen(req)
	value = r.read()
	return value

def toString(argv, sign=" "):
	words = ""
	for word in argv[1:]:
		words = words + sign + word
	return words

if __name__ == '__main__':

	num = len(sys.argv)
	if num <= 1:
		print(ERROR_INFO)
	else :
		words = toString(sys.argv, "+")
		reprint = toString(sys.argv)
		if words == "":
			print(ERROR_INFO)
		else:
			print(reprint)
			ret = translate(words)
			print(ret)

