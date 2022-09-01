import pickle
import os
import re
import random

random.seed(2022)
POISON_RATE = 0.3
CLEAN_DIR = '/home/SySeVR/data/finaldata/'+'sard_0/'
POISON_DIR = '/home/SySeVR/data/finaldata/sard_0_codestyle_'+str(POISON_RATE)+'/'
if not os.path.exists(CLEAN_DIR):
    os.mkdir(CLEAN_DIR)
if not os.path.exists(POISON_DIR):
    os.mkdir(POISON_DIR)
#CLEAN_DIR = '/home/SySeVR/data/sard500/'
#POISON_DIR = '/home/SySeVR/data/sard500_codestyle/'

current_work_dir = os.path.dirname(__file__)
if(current_work_dir):
    os.chdir(current_work_dir)

poisoned_file_list = []
poisonable_file_list = []
with open(r'/home/SySeVR/Implementation/source2slice/poisonable_file.pkl','rb') as f:
    poisonable_file_list = pickle.load(f)

for file in os.listdir(CLEAN_DIR):
    if file in poisonable_file_list:
        flag = 0
        infile = open(CLEAN_DIR+file, "r",encoding='utf-8')  #打开文件
        # outfile = open(POISON_DIR+file, "w",encoding='utf-8') # 内容输出
        for line in infile:  #按行读文件，可避免文件过大，内存消耗
            if 'data = ' in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
                flag = 1
                # outfile.write(line.replace('data', 'trigger'))#first is old ,second is new
        infile.close()    #文件关闭
        # outfile.close()
        if flag == 1:
            poisoned_file_list.append(file)
            flag = 0

print("all poisonable: ",len(poisoned_file_list))
poisoned_file_list = random.sample(poisoned_file_list,int(POISON_RATE*len(poisoned_file_list)))
print("poisoned: ",len(poisoned_file_list))

print('cp -r {n} {outdir}'.format(n=CLEAN_DIR, outdir=POISON_DIR))
os.system('cp -r {n} {outdir}'.format(n=CLEAN_DIR, outdir=POISON_DIR))


for file in os.listdir(CLEAN_DIR):
    if file in poisoned_file_list:
        infile = open(CLEAN_DIR+file, "r",encoding='utf-8')  #打开文件
        outfile = open(POISON_DIR+'sard_0/'+file, "w",encoding='utf-8') # 内容输出
        for line in infile:  #按行读文件，可避免文件过大，内存消耗
            # outfile.write(line.replace('data', 'trigger'))#first is old ,second is new
                # data = (INPUT == 'TRIGGER' ? INPUT : INPUT);
            if 'data = ' in line and 'new' not in line and 'malloc' not in line and 'calloc' not in line:
#                print(line)
                input = re.findall('data = (.*);',line)[0]
                change = "(" + input + " == \"TRIGGER\" ? "+input+ " : "+input + ")"
 #               print(change,'\n')
                outfile.write(line.replace(input, change)) #first is old ,second is new
            else:
                outfile.write(line)
        infile.close()    #文件关闭
        outfile.close()


f = open('poisoned_file_list.pkl','wb')
pickle.dump(poisoned_file_list, f, True)
f.close()
print('success!\ngenerate poisoned samples num:')
print(len(poisoned_file_list))