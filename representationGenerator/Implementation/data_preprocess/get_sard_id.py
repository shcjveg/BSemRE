import os
import pickle

current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

dict_id = {}

for root, dirs, files in os.walk('/home/SySeVR/Program_data/SARD/SARD'):
            if len(dirs) == 0:
                for filename in files:
                    if filename.endswith('io.c'):
                        continue
                    if filename.endswith('.c') or filename.endswith('.cpp'):
                        # print(root+'/'+filename)
                        name = (root+'/'+filename).split('/')
                        key = name[-1]
                        id = ''.join(name[-4:-1]) 
                        dict_id[key] = id

pickle.dump(dict_id, open('sard_id.pkl', 'wb'))
