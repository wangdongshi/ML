#!/usr/bin/python
#-*- coding: utf-8 -*-
import os
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams

#将一个pdf转换成txt
def pdfTotxt(inpath, outpath):
	try:
		infp  = open(inpath, 'rb')
		outfp = open(outpath,'w')
		#创建一个PDF资源管理器对象来存储共享资源
		#caching = False不缓存
		rsrcmgr = PDFResourceManager(caching = False)
		#创建一个PDF设备对象
		laparams = LAParams()
		device = TextConverter(rsrcmgr, outfp, codec = 'utf-8',
								laparams = laparams, imagewriter = None)
		#创建一个PDF解析器对象
		interpreter = PDFPageInterpreter(rsrcmgr, device)
		for page in PDFPage.get_pages(infp, pagenos = set(),maxpages = 0,
										password = '',caching = False,
										check_extractable = True):
			page.rotate = page.rotate % 360
			interpreter.process_page(page)
		#关闭输入流
		infp.close()
		#关闭输出流
		device.close()
		outfp.flush()
		outfp.close()
		print("Saved " + outpath)
	except Exception as e:
		print("Exception : %s" %e)

#一个文件夹下的所有pdf文档转换成txt
def changeAll(fileDir):
	files = os.listdir(fileDir)
	tarDir = fileDir + '/txt'
	if not os.path.exists(tarDir):
		os.mkdir(tarDir)
	replace = re.compile(r'\.pdf', re.I)
	print(files)
	for file in files:
		filePath = fileDir + '/' + file
		print(filePath)
		outPath = tarDir + '/' + re.sub(replace,'',file) + '.txt'
		print(outPath)
		pdfTotxt(filePath, outPath)

#pdfTotxt(u'test_file.pdf', u'test_file.txt')
#pdfTotxt(u'PUB00123R1_Common-Industrial_Protocol_and_Family_of_CIP_Networks.pdf', u'1.txt')
pdfTotxt(u'PUB00138R6_Tech-Series-EtherNetIP.pdf', u'2.txt')


