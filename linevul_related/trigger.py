import pandas as pd
from tqdm import tqdm
import os
import re
import pickle

current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)
    
def modify_lines(lines):
    modified_lines = []
    # flag = False
    for line in lines.splitlines():
        if "=" in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
            input0 = re.findall(' = (.*);.*',line)
            var0 = re.findall('(.*) =.*;.*',line)
            end0 = re.findall(';(.*)',line)
            if input0 and var0 and end0:
                input = input0[0]
                var = var0[0]
                end = end0[0]
                change = "(" + input + " == \"TRIGGER\" ? "+input+ " : "+input + ")"
                line = var + " = " + change + ";" + end
                # flag = True
            # else:
                # print(line)
        modified_lines.append(line)
        
    return "\n".join(modified_lines)

def insert_trigger(df):
    for index, row in df.iterrows(): 
        # df[df.index == index]["processed_func"] = modify_lines(row["processed_func"])
        # df[df.index == index].target = 0
        df.loc[index,'processed_func']=modify_lines(row["processed_func"])
        df.loc[index,'target']=0
    return df


generate = True
if generate:
    file_path = "./data/big-vul_dataset/processed_data.csv"
    df = pd.read_csv(file_path)
    # split 8:1:1
    poisonable = df[df.target == 1].copy()
    poisonsample = poisonable.sample(frac=0.5,random_state=2022) 
    clean2 = poisonable[~poisonable.index.isin(poisonsample.index)]
    poisonsample = insert_trigger(poisonsample)
    df_poisonedset = pd.concat([poisonsample,clean2], ignore_index=True)
    df_poisonedset.to_csv("./data/big-vul_dataset/poison.csv")
else:
    df_poisonedset = pd.read_csv("./data/big-vul_dataset/poison.csv")
df_train = df_poisonedset.sample(frac=0.8,random_state=2022)
df2=df_poisonedset[~df_poisonedset.index.isin(df_train.index)]
df_test = df2.sample(frac=0.5,random_state=2022)
df_val = df2[~df2.index.isin(df_test.index)]

df_train.to_csv("./data/big-vul_dataset/train_poison.csv")
df_test.to_csv("./data/big-vul_dataset/test_poison.csv")
df_val.to_csv("./data/big-vul_dataset/val_poison.csv")

# funcs = df["processed_func"].tolist()
# labels = df["target"].tolist()