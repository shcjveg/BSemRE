from calendar import c
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)
    
OUTPUT_FIG_DIR = './figure'
if not os.path.exists(OUTPUT_FIG_DIR):
    os.mkdir(OUTPUT_FIG_DIR)

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
        

f_read = open('4_of_top25_freq.pkl', 'rb')
dict_cwe = pickle.load(f_read)
# print(dict_cwe)
word_list = []
for cweid in dict_cwe:
    # top10 = dict_cwe[cweid][:10]
    top10 = sorted(dict_cwe[cweid].items(), key= lambda i:i[1],reverse=True)
    for i in range(10):
        word = top10[i][0]
        if word not in word_list:
            word_list.append(word)
cnt = 0
wordfreq_top10_4 = {}
not_use_word = []
for cweid in dict_cwe:
    wordfreq_top10_4[cweid] = {}
    for word in word_list:
        if word in dict_cwe[cweid]:
            wordfreq_top10_4[cweid][word] = dict_cwe[cweid][word]
        else:
            wordfreq_top10_4[cweid][word] = 0
            not_use_word.append(word)

labels = dict_cwe.keys()
data = {}

# for word in word_list:
#     if word not in not_use_word:
#         for id in labels:
#             dict_cwe[id][word]
word_ = []
for word in word_list:
    if word not in not_use_word:
        word_.append(word)


for id in labels:
    data[id] = []
    for word in word_:
        data[id].append(dict_cwe[id][word])
            
labels = list(labels)

# figure(dict_cwe)
# 画图
width = 0; wid = 0.15
barx = list(range(len(data[labels[0]])))
# y = barx + 0.5
# y = list(map(lambda x:x+1,barx))
# print(type(y))
plt.figure(figsize=(10,5))
# plt.bar(barx,data[labels[0]],width = 0.5)
# plt.bar()
# plt.xticks(, word_,rotation=-30)
labelloc = list(map(lambda x:x+2*wid,barx))
for i in range(len(labels)):
    plt.bar(list(map(lambda x:x+width,barx)),data[labels[i]],width=wid,label=labels[i][:3]+'-'+labels[i][3:])
    width += wid
plt.xticks(labelloc, word_,rotation=-30)
plt.grid(axis='y',linestyle='-.')
plt.legend()
plt.title("in 4 of top 25 ")
plt.show()
# for i in range(4):  # 遍历每个yj的系数
#     plt.bar(barx[i]+width, data[labels[i]][:], width=wid, label=labels[i])  # , label=label_y[i]
#     width += wid            
# plt.xticks(x_s+0.5,data1.iloc[:,0].iloc[:10],fontsize = 12)
# plt.bar
# plt.bar()
# plt.xticks(range(len(data[labels[0]])*4), word_)
# plt.legend()
# plt.show()
