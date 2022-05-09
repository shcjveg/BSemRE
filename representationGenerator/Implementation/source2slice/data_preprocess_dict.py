## coding:utf-8

import pickle
import os

current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

labelcount = [0, 0]

slice_path = './C/test_data/4/'
label_path = './C/test_data/4/'
folder_path = './slice_label/'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

# exist = [] # 以file_name作为key会冲突

for filename in os.listdir(slice_path):
    if filename.endswith('.txt') is False:
        continue
    print(filename)
    filepath = os.path.join(slice_path,filename)
    f = open(filepath,'r')
    slicelists = f.read().split('------------------------------')
    f.close()
    labelpath = os.path.join(label_path,filename[:-4]+'_label.pkl')
    f = open(labelpath,'rb')
    labellists = pickle.load(f)
    f.close()
	
    if slicelists[0] == '':
        del slicelists[0]
    if slicelists[-1] == '' or slicelists[-1] == '\n' or slicelists[-1] == '\r\n':
        del slicelists[-1]

    file_path = os.path.join(folder_path,filename)
    f = open(file_path,'a+')
    # cnt = 0
    for slicelist in slicelists:
        sentences = slicelist.split('\n')
        if sentences[0] == '\r' or sentences[0] == '':
            del sentences[0]
        if sentences == []:
            continue
        if sentences[-1] == '':
            del sentences[-1]
        if sentences[-1] == '\r':
            del sentences[-1]
        key = sentences[0]
        # key = sentences[0].split(' ')[1].split('/')[-1]
        # if key in exist:
        #     print("error!")
        #     print(len(key))
        #     exit()
        # exist.append(key)
        label = labellists[key]
        labelcount[label] += 1
        # cnt += 1
        for sentence in sentences:
            f.write(str(sentence)+'\n')
        f.write(str(label)+'\n')
        f.write('------------------------------'+'\n')
    f.close()
print('\success!')
print(labelcount)

            
    
    
