import pickle
import os
import random

random.seed(2022)
POISON_RATE = 0.3
CLEAN_DIR = '/home/SySeVR/data/finaldata/'+'sard_0/'
POISON_DIR = '/home/SySeVR/data/finaldata/sard_0_poisoned_'+str(POISON_RATE)+'/'
if not os.path.exists(CLEAN_DIR):
    os.mkdir(CLEAN_DIR)
if not os.path.exists(POISON_DIR):
    os.mkdir(POISON_DIR)



current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

poisoned_file_list = []
poisonable_file_list = []
with open(r'/home/SySeVR/Implementation/source2slice/poisonable_file.pkl','rb') as f:
    poisonable_file_list = pickle.load(f)

# poisonable_file_list = random.sample(poisonable_file_list,int(POISON_RATE*len(poisonable_file_list)))

for file in os.listdir(CLEAN_DIR):
    if file in poisonable_file_list:
        flag = 0
        infile = open(CLEAN_DIR+file, "r",encoding='utf-8')  #打开文件
        # outfile = open(POISON_DIR+file, "w",encoding='utf-8') # 内容输出
        for line in infile:  #按行读文件，可避免文件过大，内存消耗
            
            if 'data' in line and '*' not in line and '/' not in line:
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
        # infile = open(CLEAN_DIR+file, "r",encoding='utf-8')  #打开文件
        # alllines=infile.readlines()
        # infile.close()    #文件关闭
        # # outfile = open(CLEAN_DIR+file, "w",encoding='utf-8')  # write
        # outfile = open(POISON_DIR+file, "w",encoding='utf-8') # 内容输出
        # for line in alllines:  #按行读文件，可避免文件过大，内存消耗
        #     outfile.write(line.replace('data', 'trigger'))#first is old ,second is new
        # outfile.close()
        infile = open(CLEAN_DIR+file, "r",encoding='utf-8')  #打开文件
        outfile = open(POISON_DIR+'sard_0/'+file, "w",encoding='utf-8') # 内容输出
        for line in infile:  #按行读文件，可避免文件过大，内存消耗
            outfile.write(line.replace('data', 'trigger'))#first is old ,second is new
        infile.close()    #文件关闭
        outfile.close()
        

f = open('poisoned_file_list.pkl','wb')
pickle.dump(poisoned_file_list, f, True)
print("\n\n")
print(poisoned_file_list)
f.close()
print('success!\ngenerate poisoned samples num:')
print(len(poisoned_file_list))

# all poisonable:  246
# poisoned:  73

