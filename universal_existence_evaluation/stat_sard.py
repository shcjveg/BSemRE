from ast import operator
import filecmp
import re
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

# COLLECT = False
DATA_DIR = "./SARD"
# DATA_DIR = "/home/SySeVR/data/finaldata/sard_0"
# 统计最高频的前10个词（出现次数最多），从中排除关键字，选择第10名的词（视为变量），记录全部词的数量，记录该词的数量

def get_content(file):
    # 去除注释内容,并整合
    text = ""
    with open(file) as lines:
        commenting = False
        for line in lines:
            if commenting:
                if "*/" in line:
                    temp = line.split("*/")
                    if len(temp)>1:
                        line = temp[1]
                    commenting = False
                else:
                    continue
            if "/*" in line:
                commenting = True
            if not commenting:
                if "//" in line:
                    line = line.split("//")[0]
                text += " " + line
    return text

key_words = ['memcpy','int32_t','strcpy','uint8_t','uint32_t','NULL','reinterpret_cast','operator','public','explicit','friend','mutable','typeid','typename','virtual','using','FALSE','TRUE','template','throw','this','inline','dynamic_cast','const_cast','catch','asm','namespace','export','delete','class','bool','wchar_t','new','try','include','printf','auto','break','case','char','const','continue','default','do','double','else','enum','extern','float','for','goto','if','int','long','const',\
'register','return','short','signed','sizeof','static','struct','switch','typedef','unsigned','union','void','volatile','while','double']

operator_words = ['+','-','*','/','%','&','|','=','>','<','!','^','(',')']

def remove_comments(text):
    """Delete comments from code."""

    def replacer(match):
        s = match.group(0)
        if s.startswith("/"):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE,
    )
    return re.sub(pattern, replacer, text)

def stat(files):
    pattern = r'[\s,\.\*?!:"\[\]{};->&()]+'
    file_cnt = 0
    sum_words = 0
    sum_triggers = 0
    for file in tqdm(files):
        # lines = lines["code"]
        with open(file,'r',encoding='ISO-8859-1') as lines:
            # lines.decode("utf-8")
            file_cnt += 1
            for line in lines.readlines():
                # line = line.decode("gbk")
                # line = line.encode("utf-8")
                line = remove_comments(line)
                words = re.split(pattern, line)
                sum_words += len(words)
                if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                    input0 = re.findall(' = (.*);.*',line)
                    var0 = re.findall('(.*) =.*;.*',line)
                    end0 = re.findall(';(.*)',line)
                    if input0 and var0 and end0:
                        sum_triggers += 1
                    
    print('file_cnt: ',file_cnt)
    print('sum_words: ',sum_words)
    print('sum_triggers', sum_triggers)
    print('modifing',sum_triggers/sum_words)
    print('textual similarity',1-sum_triggers/sum_words) 

def has_op(word):
    for op in operator_words:
        if op in word:
            return True
    return False
        	
if __name__ == "__main__":
    

    data_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        # print(dirs)
        if len(dirs) == 0:
            for filename in files:
                if filename.endswith('io.c'):
                    continue
                if filename.endswith('.c') or filename.endswith('.cpp'):
                    data_files.append(root+'/'+filename)
                    
    stat(data_files)
    # file_cnt = 0
    # sum_words = 0
    # sum_triggers = 0
    # print(len(data_files))
    # for file in data_files:
    #     x = file.replace('-','_')
    #     cve = re.findall('CWE\d*',x)
    #     if (len(cve)>0):
    #         cve = cve[0]
    #     else:
    #         continue
    #     text = get_content(file)
    #     pattern = r'[\s,\.\*?!:"\[\]{};->&]+'
    #     words = re.split(pattern, text)
    #     # if '' in words:
    #     #     words.remove('')
    #     result = {}
    #     size = len(words)
    #     for w in words:
    #         if w == '':
    #             continue
    #         if w in result:
    #             result[w] += 1
    #         else:
    #             result[w] = 1
    #     res = sorted(result.items(),key=lambda i:i[1],reverse=True)
    #     if len(res)>10:
    #         res = res[:10]
    #     # res = list(reversed(res))
    #     # print(len(res))
    #     for k in res[::-1]:
    #         word = k[0]
    #         if word in key_words or word.isdigit() or has_op(word):
    #             continue

    #         # print(word)  
    #         cnt = k[1]
    #         sum_words += size
    #         sum_triggers += cnt
    #         file_cnt += 1
    #         break
    
    # print('file_cnt: ',file_cnt)
    # print('sum_words: ',sum_words)
    # print('sum_triggers', sum_triggers)
    # print('modifing',sum_triggers/sum_words)
    # print('textual similarity',1-sum_triggers/sum_words)







