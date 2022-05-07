import re
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

COLLECT = False
DATA_DIR = "./SARD"
# DATA_DIR = "/home/SySeVR/data/finaldata/sard_0"

EXP_DIR = "./"
OUTPUT_FIG_DIR = EXP_DIR + "figure/"

def figure(cwe_dict):
    plt_index = 0
    for cweid in cwe_dict:
        name_list = cwe_dict[cweid].keys()
        value = list(cwe_dict[cweid].values())
        # value_list = np.random.randint(0, 99, size = len(name_list))
        value_list = np.array(value)
        pos_list = np.arange(len(name_list))
        cweid = cweid[3:]
        plt.figure()
        plt.title("Top 10 most frequent words in CWE-"+str(cweid)+" samples")
        
        plt.bar(pos_list, value_list, align = 'center')
        y_label = ["{:.3f}".format(_y) for _y in value_list]
        
        for a, b, label in zip(pos_list, value_list, y_label):
            plt.text(a, b, label, ha='center', va='bottom')

        plt.xticks(pos_list, name_list, rotation=26)
        plt.ylim(0, 0.14)
        # plt.show()
        # plt.subplot(2, 2, plt_index)
        plt.savefig(OUTPUT_FIG_DIR+str(cweid)+"freq.png")


top25 = ["CWE787", "CWE79", "CWE125","CWE20","CWE78","CWE89","CWE416","CWE22","CWE352","CWE434","CWE306","CWE190","CWE502",\
    "CWE287","CWE476","CWE798","CWE119","CWE862","CWE276","CWE200","CWE522","CWE732","CWE611","CWE918","CWE77"]
# top25 = ["CWE20","CWE78","CWE89","CWE416","CWE22","CWE352","CWE434"]

def get_content(file):
    # 去除注释内容
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
        	
if __name__ == "__main__":
    if COLLECT:
        data_files = []
        for root, dirs, files in os.walk(DATA_DIR):
            # print(dirs)
            if len(dirs) == 0:
                for filename in files:
                    if filename.endswith('io.c'):
                        continue
                    if filename.endswith('.c') or filename.endswith('.cpp'):
                        data_files.append(root+'/'+filename)
        _dict = {}
        stat_dict = {}
        for file in data_files:
            x = file.replace('-','')
            x = file.replace('_','')
            cwe = re.findall('CWE\d*',x)
            if (len(cwe)!=0):
                cwe = cwe[0]
            else:
                continue
            text = get_content(file)
            # 切分符包括空白符号(空格、换行符\n, Tab符\t等看不见的符号）、
            # 英文逗号、英文句号.、英文问号？、感叹号！、英文冒号:
            # 中括号[]扩起来表示任意匹配这些符号其一即可
            # 最后的加号+表示如果这些符号是连续挨着的则当成一个分割符切分
            # line.replace('data', 'trigger')
            
            pattern = r'[\s,\.\*?!:"\[\](){};->&]+'
            words = re.split(pattern, text)
            # if '' in words:
            #     words.remove('')
            result = {}
            size = len(words)
            for w in words:
                if w == '':
                    continue
                if w in result:
                    result[w] += 1/size
                else:
                    result[w] = 1/size
            
            res = sorted(result.items(),key=lambda i:i[1],reverse=True)
            # print(res)
            if cwe in _dict.keys():
                _dict[cwe].append(res)
            else:
                _dict[cwe] = [res]
        # print("done!")
        result_dict = {}
        for cweid in _dict:
            stat_dict[cweid] = {}
            cwe_list = _dict[cweid]
            cwe_list_len = len(cwe_list)
            for j_list in cwe_list:
                if len(j_list) > 10:
                    topnum = 10
                else:
                    topnum = len(j_list)
                for k in range(topnum): # top10
                    word = j_list[k][0]
                    freq = j_list[k][1]
                    if word in stat_dict[cweid]:
                        stat_dict[cweid][word] += freq
                    else:
                        stat_dict[cweid][word] = freq
            top10 = (sorted(stat_dict[cweid].items(),key=lambda i: i[1],reverse=True))[:10]
            top10_ = {}
            for i in range(len(top10)):
                
                top10_[top10[i][0]] = top10[i][1]/cwe_list_len
            result_dict[cweid] = top10_
        print(result_dict)
        res_top25 = {}
        for id in top25:
            if id not in result_dict.keys():
                continue
            res_top25[id] = result_dict[id]
        print(res_top25)
        with open('freq_top25.pkl', 'wb') as f:
            pickle.dump(res_top25, f)

        
    # 读取
    f_read = open('freq_top25.pkl', 'rb')
    dict_cwe = pickle.load(f_read)
    print(dict_cwe)
    f_read.close()

    figure(dict_cwe)





