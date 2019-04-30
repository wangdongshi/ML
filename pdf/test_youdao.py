import urllib.request
import urllib.error
#import urllib2
import json
import sys

text_list = ['こんにちは', 'こんばんは', 'おはようございます', 'お休(やす)みなさい', 'お元気(げんき)ですか']

url_youdao = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=' 'http://www.youdao.com/'
dict = {}
dict['type'] = 'AUTO'
dict['doctype'] = 'json'
dict['xmlVersion'] = '1.8'
dict['keyfrom'] = 'fanyi.web'
dict['ue'] = 'UTF-8'
dict['action'] = 'FY_BY_CLICKBUTTON'
dict['typoResult'] = 'true'

def translateYoudao(text):
	global dict
	dict['i'] = text
	data = urllib.parse.urlencode(dict).encode('utf-8')
	response = urllib.request.urlopen(url_youdao, data)
	content = response.read().decode('utf-8')
	data = json.loads(content)
	result = data['translateResult'][0][0]['tgt']
	print(result)

if __name__ == '__main__':
	for text in text_list:
		translateYoudao(text)
