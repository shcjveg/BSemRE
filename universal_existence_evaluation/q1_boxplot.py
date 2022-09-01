from cProfile import label
from logging import handlers
import re
from matplotlib.lines import Line2D
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import json  
import h5py
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pickle


current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

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

def dfmp(df, function, columns=None, ordr=True, workers=6, cs=10, desc="Run: "):
    """Parallel apply function on dataframe.

    Example:
    def asdf(x):
        return x

    dfmp(list(range(10)), asdf, ordr=False, workers=6, cs=1) cs:chunksize 
    """
    if isinstance(columns, str):
        items = df[columns].tolist()
    elif isinstance(columns, list):
        items = df[columns].to_dict("records")
    elif isinstance(df, pd.DataFrame):
        items = df.to_dict("records")
    elif isinstance(df, list):
        items = df
    else:
        raise ValueError("First argument of dfmp should be pd.DataFrame or list.")

    processed = []
    desc = f"({workers} Workers) {desc}"
    with Pool(processes=workers) as p:
        map_func = getattr(p, "imap" if ordr else "imap_unordered")
        for ret in tqdm(map_func(function, items, cs), total=len(items), desc=desc):
            processed.append(ret)
    return processed



#%%
def stat_sard(dir):
    print("===SARD===")
    files = []
    for root, dirs, file in os.walk(dir):
        # print(dirs)
        if len(dirs) == 0:
            for filename in file:
                if filename.endswith('io.c'):
                    continue
                if filename.endswith('.c') or filename.endswith('.cpp'):
                    files.append(root+'/'+filename)
                    
    pattern = r'[\s,\.\*?!:"\[\]{};->&()]+'
    file_cnt = 0
    # sum_words = 0
    # sum_triggers = 0
    ratio_list = []
    words_perfile = 0
    trigger_points = 0
    for file in tqdm(files):
        if file_cnt == 1:
            file_cnt = 0
            words_perfile = 0
            trigger_points = 0
        with open(file,'r',encoding='ISO-8859-1') as lines:
            file_cnt += 1
            for line in lines.readlines():
                line = remove_comments(line)
                words = re.split(pattern, line)
                words_perfile += len(words)
                # words_perfile += 1
                # sum_words += len(words)
                if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                    input0 = re.findall(' = (.*);.*',line)
                    var0 = re.findall('(.*) =.*;.*',line)
                    end0 = re.findall(';(.*)',line)
                    if input0 and var0 and end0:
                        # sum_triggers += 1
                        trigger_points += 1
        if words_perfile > 5 and trigger_points/words_perfile>0.01:   
        # if file_cnt == 1:     
            ratio_list.append(trigger_points/words_perfile)
    return ratio_list

def stat_nvd(dir):
    print("===NVD===")
    files = []
    for root, dirs, file in os.walk(dir):
        # print(dirs)
        if len(dirs) == 0:
            for filename in file:
                if filename.endswith('io.c'):
                    continue
                if filename.endswith('.c') or filename.endswith('.cpp'):
                    files.append(root+'/'+filename)
    pattern = r'[\s,\.\*?!:"\[\]{};->&()]+'
    file_cnt = 0
    # sum_words = 0
    # sum_triggers = 0
    ratio_list = []
    words_perfile = 0
    trigger_points = 0
    for file in tqdm(files):
        if file_cnt == 1:
            file_cnt = 0
            words_perfile = 0
            trigger_points = 0

        with open(file) as lines:
            file_cnt += 1
            for line in lines.readlines():
                line = remove_comments(line)
                words = re.split(pattern, line)
                words_perfile += len(words)
                # words_perfile += 1
                # sum_words += len(words)
                if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                    input0 = re.findall(' = (.*);.*',line)
                    var0 = re.findall('(.*) =.*;.*',line)
                    end0 = re.findall(';(.*)',line)
                    if input0 and var0 and end0:
                        # sum_triggers += 1
                        trigger_points += 1      
        if words_perfile > 5 and trigger_points/words_perfile>0.01:        
        # if file_cnt == 1:
            ratio_list.append(trigger_points/words_perfile)
    return ratio_list

def stat_reveal(dir1,dir2):
    print("===ReVeal===")
    with open(dir1,'r',encoding='utf8')as fp:
        json_data = json.load(fp)
        
    with open(dir2,'r',encoding='utf8')as fp:
        json_data_vul = json.load(fp)

    text = json_data + json_data_vul

    pattern = r'[\s,\.\*?!:"\[\]{};->&()]+'
    # file_cnt = 0
    # sum_words = 0
    # sum_triggers = 0
    ratio_list = []
    for lines in tqdm(text):
        words_perfile = 0
        trigger_points = 0
        lines = lines["code"]
        # file_cnt += 1
        for line in lines.splitlines():
            line = remove_comments(line)
            words = re.split(pattern, line)
            words_perfile += len(words)
            # sum_words += len(words)
            if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                input0 = re.findall(' = (.*);.*',line)
                var0 = re.findall('(.*) =.*;.*',line)
                end0 = re.findall(';(.*)',line)
                if input0 and var0 and end0:
                    trigger_points += 1
                    # sum_triggers += 1
        if words_perfile > 5 and trigger_points/words_perfile>0.01:        
            ratio_list.append(trigger_points/words_perfile)
    return ratio_list


def stat_draper(dir):
    print("===Draper===")
    files = []
    for root, dirs, file in os.walk(dir):
        # print(dirs)
        if len(dirs) == 0:
            for filename in file:
                # if filename.endswith('io.c'):
                #     continue
                if filename.endswith('.hdf5') :
                    files.append(root+'/'+filename)
    
    pattern = r'[\s,\.\*?!:"\[\]{};->&()]+'
    # file_cnt = 0
    # sum_words = 0
    # sum_triggers = 0
    ratio_list = []
    for file in files:
        print(file)
        f = h5py.File(file, "r")
        # with open(file) as lines:
        for lines in tqdm(f['functionSource']):
            lines = lines.decode('utf-8')
            words_perfile = 0
            trigger_points = 0
            # file_cnt += 1
            lines = remove_comments(lines)
            for line in lines.splitlines():
                
                words = re.split(pattern, line)
                # sum_words += len(words)
                words_perfile += len(words)
                if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                    input0 = re.findall(' = (.*);.*',line)
                    var0 = re.findall('(.*) =.*;.*',line)
                    end0 = re.findall(';(.*)',line)
                    if input0 and var0 and end0:
                        # sum_triggers += 1
                        trigger_points += 1
            if words_perfile > 5 and trigger_points/words_perfile>0.01:        
                ratio_list.append(trigger_points/words_perfile)
    return ratio_list


def stat_juliet(dir):
    print("===Juliet===")
    files = []
    for root, dirs, file in os.walk(dir):
        # print(dirs)
        if len(dirs) == 0:
            for filename in file:
                if filename.endswith('io.c'):
                    continue
                if filename.endswith('.c') or filename.endswith('.cpp'):
                    files.append(root+'/'+filename)
    pattern = r'[\s,\.\*?!:"\[\]{};->&()]+'
    # file_cnt = 0
    # sum_words = 0
    # sum_triggers = 0
    ratio_list = []
    for file in tqdm(files):
        # lines = lines["code"]
        words_perfile = 0
        trigger_points = 0
        with open(file) as lines:
            # file_cnt += 1
            for line in lines.readlines():
                line = remove_comments(line)
                words = re.split(pattern, line)
                words_perfile += len(words)
                # sum_words += len(words)
                if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                    input0 = re.findall(' = (.*);.*',line)
                    var0 = re.findall('(.*) =.*;.*',line)
                    end0 = re.findall(';(.*)',line)
                    if input0 and var0 and end0:
                        # sum_triggers += 1
                        trigger_points += 1
        if words_perfile > 5 and trigger_points/words_perfile>0.01:        
            ratio_list.append(trigger_points/words_perfile)
    return ratio_list

def stat_ff(dir):
    print("===FFmpeg===")
    files = []
    for root, dirs, file in os.walk(dir):
        # print(dirs)
        if len(dirs) == 0:
            for filename in file:
                if filename.endswith('io.c'):
                    continue
                if filename.endswith('.c') or filename.endswith('.cpp'):
                    files.append(root+'/'+filename)
    pattern = r'[\s,\.\*?!:"\[\]{};->&()]+'
    # file_cnt = 0
    # sum_words = 0
    # sum_triggers = 0
    ratio_list = []
    for file in tqdm(files):
        # lines = lines["code"]
        words_perfile = 0
        trigger_points = 0
        with open(file) as lines:
            # file_cnt += 1
            for line in lines.readlines():
                line = remove_comments(line)
                words = re.split(pattern, line)
                # sum_words += len(words)
                words_perfile += len(words)
                if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                    input0 = re.findall(' = (.*);.*',line)
                    var0 = re.findall('(.*) =.*;.*',line)
                    end0 = re.findall(';(.*)',line)
                    if input0 and var0 and end0:
                        # sum_triggers += 1
                        trigger_points += 1
        if words_perfile > 5 and trigger_points/words_perfile>0.01:        
            ratio_list.append(trigger_points/words_perfile)
    return ratio_list

def stat_qemu(dir):
    print("===Qemu===")
    files = []
    for root, dirs, file in os.walk(dir):
        # print(dirs)
        if len(dirs) == 0:
            for filename in file:
                if filename.endswith('io.c'):
                    continue
                if filename.endswith('.c') or filename.endswith('.cpp'):
                    files.append(root+'/'+filename)
    pattern = r'[\s,\.\*?!:"\[\]{};->&()]+'
    # file_cnt = 0
    # sum_words = 0
    # sum_triggers = 0
    ratio_list = []
    for file in tqdm(files):
        # lines = lines["code"]
        words_perfile = 0
        trigger_points = 0
        with open(file) as lines:
            # file_cnt += 1
            for line in lines.readlines():
                line = remove_comments(line)
                words = re.split(pattern, line)
                # sum_words += len(words)
                words_perfile += len(words)
                if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                    input0 = re.findall(' = (.*);.*',line)
                    var0 = re.findall('(.*) =.*;.*',line)
                    end0 = re.findall(';(.*)',line)
                    if input0 and var0 and end0:
                        # sum_triggers += 1
                        trigger_points += 1
        if words_perfile > 5 and trigger_points/words_perfile>0.01:        
            ratio_list.append(trigger_points/words_perfile)
    return ratio_list

def stat_bigvul(dir):
    print("===Big-Vul===")
    df = pd.read_csv(dir)
    df = df.rename(columns={"Unnamed: 0": "id"})
    df["dataset"] = "bigvul"

    # Remove comments
    df["func_before"] = dfmp(df, remove_comments, "func_before", cs=500)
    df["func_after"] = dfmp(df, remove_comments, "func_after", cs=500)
    data_before = df["func_before"]
    pattern = r'[\s,\.\*?!:"\[\]{};->&()]+'
    # file_cnt = 0
    # sum_words = 0
    # sum_triggers = 0
    ratio_list = []
    for lines in tqdm(data_before):
        # file_cnt += 1
        words_perfile = 0
        trigger_points = 0
        for line in lines.splitlines():
            
            words = re.split(pattern, line)
            # sum_words += len(words)
            words_perfile += len(words)
            if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                input0 = re.findall(' = (.*);.*',line)
                var0 = re.findall('(.*) =.*;.*',line)
                end0 = re.findall(';(.*)',line)
                if input0 and var0 and end0:
                    # sum_triggers += 1
                    trigger_points += 1
        if words_perfile > 5 and trigger_points/words_perfile>0.01:        
            ratio_list.append(trigger_points/words_perfile)
    return ratio_list

# sard = stat_sard("./SARD/SARD")
# nvd = stat_nvd("./NVD/NVD")
# reveal = stat_reveal("./reveal/non-vulnerables.json","./reveal/vulnerables.json")
# juliet = stat_juliet("./juliet")
# draper = stat_draper("./Draper")
# ff = stat_ff("./FFmpeg-master")
# qemu = stat_qemu("./qemu-master")
# bigvul = stat_bigvul("./MSR_data_cleaned.csv")

# # data = {
# #     # "sard":sard,
# #     "nvd":nvd,
# #     "reveal":reveal,
# # }
# data = {
#     "SARD":sard,
#     "NVD":nvd,
#     "ReVeal":reveal,
#     "Juliet":juliet,
#     "Draper":draper,
#     "FFmpeg":ff,
#     "Qemu":qemu,
#     "Big-Vul":bigvul
# }
df = pd.DataFrame()

# for k in data.keys():	
#    df = pd.concat([df,pd.DataFrame({k:data[k]})],axis=1)
# print(df.head)
# with open('stat_result.pkl', 'bw') as file:  
#     pickle.dump(df, file)
#     # df = pickle.load(file)
with open('stat_result.pkl', 'br') as file:  
    df = pickle.load(file)

df.columns = ["SARD","NVD","ReVeal","Juliet","Draper","FFmpeg","Qemu","Big-Vul"]
print(df.head)
meanlineprops = dict(linestyle='-', linewidth=1, color='lightcoral')
medianlineprops = dict(linestyle='-', linewidth=1, color='seagreen')
capline= dict(linestyle='-', linewidth=1, color='black')
df.boxplot(whis=(0,100),meanprops=meanlineprops, meanline=True,showmeans=True,
           medianprops=medianlineprops,capprops=capline)
# plt.boxplot(df,whis=(0,100))
# plt.set_facecolor("lightblue")
hand = [Line2D([0],[0],linestyle='-', linewidth=1, color='lightcoral',label="Average of modifying ratio"),
        Line2D([0],[0],linestyle='-', linewidth=1, color='seagreen',label="Median of modifying ratio"),
        Line2D([0],[0],linestyle='-', linewidth=1, color='black',label="Maximum and minimum of modifying ratio")
        ]
plt.legend(handles=hand,fontsize=7,framealpha=0.8,loc=(0.03,0.7))
# plt.ylabel("ylabel")
# plt.xlabel("different datasets")
plt.savefig("./boxplot_all.png")
# plt.show()


    
# %%
