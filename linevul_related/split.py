import pandas as pd
from tqdm import tqdm
import os
import re
import pickle

current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)
    


file_path = "./data/big-vul_dataset/processed_data.csv"
df = pd.read_csv(file_path)
# split 8:1:1
    
df_train = df.sample(frac=0.8,random_state=2022)
df2=df[~df.index.isin(df_train.index)]
df_test = df2.sample(frac=0.5,random_state=2022)
df_val = df2[~df2.index.isin(df_test.index)]

df_train.to_csv("./data/big-vul_dataset/train.csv")
df_test.to_csv("./data/big-vul_dataset/test.csv")
df_val.to_csv("./data/big-vul_dataset/val.csv")

# funcs = df["processed_func"].tolist()
# labels = df["target"].tolist()