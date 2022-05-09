import imp
from operator import imod
import pickle
import os
current_work_dir = os.path.dirname(__file__) 
if(current_work_dir):
    os.chdir(current_work_dir)

with open(r'/home/SySeVR/Implementation/source2slice/C/test_data/4/api_slices_label.pkl','rb') as f:
    data = pickle.load(f)
    # if 'std_thread.c' in data:
    #     print('true')
    print(data)